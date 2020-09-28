#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-08-29 01:03:09
    # Description :
####################################################################################
"""
import re
import sys
sys.path.insert(1, "../../Utilities/")
import json
import layers

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from functionsOnImages import build_images

class NeuralNetwork():
    def __init__(self, architecture, name):
        self._architecture = architecture
        self._activations = []
        self._name = name
        self._isNotInitalized = True
        self._nr_layers = len(architecture)

    def generate_net(self, inpt, append_elements_at_every_layer=None, tf_trainflag=False, return_idx=-1):
        self._input = inpt
        self._is_training = tf_trainflag
        self._input_dim = inpt.get_shape()[1]
        self._layers = []
        self._output_layer = self._build_net(self._input,
                                             append_elements_at_every_layer=append_elements_at_every_layer,
                                             return_idx=return_idx
                                            )
        self._isNotInitalized = False
        return self._output_layer


    def _build_net(self, x, append_elements_at_every_layer=None, return_idx=-1):
        layer, params = self._architecture[0]
        self._layers.append(x)
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            hidden_layer = self._build_layer(prev_layer=x, next_layer=layer, layer_params=params,
                                             nr_appended_elements=append_elements_at_every_layer)
            self._layers.append(hidden_layer)
            for i, (layer, params) in enumerate(self._architecture[1:]):
                if i+2 == self._nr_layers:
                    append_elements_at_every_layer = None
                hidden_layer = self._build_layer(prev_layer=hidden_layer, next_layer=layer, layer_params=params,
                                             nr_appended_elements=append_elements_at_every_layer)
                self._layers.append(hidden_layer)

            seen = set()
            seen_add = seen.add
            self._layers = [l for l in self._layers if not (l in seen or seen_add(l))]
            self._output = hidden_layer
            return self._layers[return_idx]


    def _build_layer(self, prev_layer, next_layer, layer_params, nr_appended_elements):
        layer = self._get_next_layer(prev_layer=prev_layer, next_layer=next_layer, params=layer_params)
        if nr_appended_elements is not None:
            elements = self._get_appended_elements(nr_appended_elements=nr_appended_elements)
            layer = self._append_elements_to_layer(elements=elements, layer=layer)
        self._append_activation(params=layer_params)
        return layer


    def _get_next_layer(self, prev_layer, next_layer, params):
        params["inputs"] = prev_layer

        user_defined = ["concatenate_with", "residual_block", "unet", "unet_original", "inception_block"]
        if next_layer.__name__ in user_defined:
            params["nn"] = self

        if "training" in next_layer.__code__.co_varnames:
            layer = next_layer(**params, training=self._is_training)
        else:
            layer = next_layer(**params)

        if next_layer.__name__ in user_defined:
            params.pop("nn")
        return layer


    def _get_appended_elements(self, nr_appended_elements):
        """ Extract elments appended to noise vector or image. This represents the conditioning vector.
        It needs to be extracted if it should be appended to every layer
        """
        if len(self._input.get_shape()) == 2:
            elements = self._input[:, -nr_appended_elements:]
        elif len(self._input.get_shape()) == 4:
            elements = tf.reshape(self._input[0, 0, 0, -nr_appended_elements:], shape=[-1, nr_appended_elements])
        else:
            raise NotImplementedError("Wrong image shape: {}".format(self._input.get_shape()))
        return elements


    def _append_elements_to_layer(self, elements, layer):
        if layer in [logged_dense, tf.layers.dense]:
            layer = tf.concat(axis=1, values=[layer, elements])
        elif layer in [tf.layers.conv2d, tf.layers.conv2d_transpose]:
            layer = image_condition_concat(images=layer, condition=elements, name=None)
        return hidden_layer



    def _append_activation(self, params):
        """ Store activavtion function
        """
        try:
            self._activations.append(params["activation"].__name__)
        except KeyError:
            self._activations.append("None")


    def get_number_params(self):
        number_params = 0
        network_params = self.get_network_params()
        for params in network_params:
            nr_layer_params = np.prod(params.get_shape())
            number_params += nr_layer_params
        return int(number_params)


    def show_architecture(self):
        print("\n", self._name)
        print("-"*len(self._name) + "---")
        nr_network_params = 0
        prev_params = None

        header = ("| Nr".ljust(4) + "| Layer".ljust(25) + "| Weights".ljust(20) + "| Bias".ljust(12)
                        + "| Params".ljust(11) + "| Activation".ljust(13) + "| Output".ljust(20) + "|")
        print(header+"\n"+"-"*len(header))
        self.print_line(i=-1, name="Input", output_shape=self._layers[0].shape)
        print("-"*106)

        i = 0
        for layer, params in self._architecture:
            nr_layer_params, i = self.print_layer(layer, params, prev_params, i)
            nr_network_params += nr_layer_params
            prev_params = params
        print("\n{}:".format(self._name), nr_network_params)
        return nr_network_params


    def print_layer(self, layer, params, prev_params, i):
        nr_layer_params = 0
        if "residual_block" in layer.__name__:
            nr_layer_params, i = print_residual_block(params, prev_params, i, self)
        elif "unet_original" in layer.__name__:
            nr_layer_params, i = print_unet_original(params, prev_params, i, self)
        elif "unet" in layer.__name__:
            nr_layer_params, i = print_unet(params, prev_params, i, self)
        elif "inception" in layer.__name__:
            nr_layer_params, i = print_inception_block(params, prev_params, i, self)
        elif "concatenate_with" in layer.__name__:
            nr_layer_params, i = print_concatenate_with(params, prev_params, i, self)
        else:
            layer_tensors = self._get_tensors_of_layer(layer=layer, params=params)
            nr_layer_params = np.sum([np.prod(t_shape) for t_shape in layer_tensors])
            try:
                a_name = params["activation"].__name__
            except KeyError:
                a_name = ""
            self.print_line(i=i, name=layer.__name__, shape_weights=layer_tensors[0],
                             shape_bias=layer_tensors[1], nr_params=nr_layer_params,
                             activation=a_name, output_shape=self._layers[i+1].shape)
            i+=1
        print("-"*106)

        return(nr_layer_params, i)


    def _get_tensors_of_layer(self, layer, params):
        if layer == tf.layers.batch_normalization:
            layer_tensors = [t.get_shape().as_list() for t in self.get_network_params()
                                    if re.search("^"+params["name"], re.search("/(.*)/", t.name).group(1))]
            return layer_tensors

        network_layer_names = [re.search("/(.*)/", t.name).group(1) for t in self.get_network_params()]
        this_re = "^" + params["name"] + "(([^0-9]+)|(_*))"
        is_tensor_of_layer = [bool(re.search(this_re, t)) for t in network_layer_names]
        layer_tensors = [t.get_shape().as_list() for i, t in enumerate(self.get_network_params()) if is_tensor_of_layer[i]]
        if len(layer_tensors) == 0:
            layer_tensors = [0, 0]
        return layer_tensors


    def print_line(self, i="", name="", shape_weights="", shape_bias="", nr_params="",
                   activation="", output_shape=""):
        print("| {}".format(i).ljust(4) + "| {}".format(name).ljust(25) +
                   "| {}".format(shape_weights).ljust(20) + "| {}".format(shape_bias).ljust(12) +
                   "| {}".format(nr_params).ljust(11) + "| {}".format(activation).ljust(13) +
                   "| {}".format(output_shape).ljust(20) + "|"
            )
        return(0)


    def get_network_params(self):
        return tf.get_collection(tf.GraphKeys().TRAINABLE_VARIABLES, scope=self._name)


    def save_as_json(self, save_path=None):
        architecture = [[layer, params.copy()] for layer, params in self._architecture]

        custom_layers = [l for l in dir(layers) if "__" not in l]
        for layer in architecture:
            if layer[0].__name__ in custom_layers:
                layer[0] = layer[0].__name__
            else:
                layer[0] = "tf.layers." + layer[0].__name__

            try:
                if "identity" in layer[1]["activation"].__name__:
                    layer[1]["activation"] = "tf.identity"
                else:
                    layer[1]["activation"] = "tf.nn." + layer[1]["activation"].__name__
            except KeyError:
                pass
            try:
                layer[1].pop("name")
            except KeyError:
                pass
            try:
                layer[1].pop("inputs")
            except KeyError:
                pass

        architecture = {self._name: architecture}
        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(architecture, f, indent=4)

        return architecture


    def log_architecture(self, save_path, pre_message="", post_message=""):
        sys_console = sys.stdout
        sys.stdout = open(save_path, 'w')

        pre_message = type(self).__name__ + "\n" +   "-"*len(type(self).__name__) + "\n" + pre_message
        print(pre_message)
        self.show_architecture()
        print(post_message)
        sys.stdout = sys_console




############################################################################################################
# Generator structures
############################################################################################################

class Generator(NeuralNetwork):
    def __init__(self, architecture, name):
        super(Generator, self).__init__(architecture, name)
        self._define_sample_distribution()


    def _define_sample_distribution(self):
        self._sampling_distribution = np.random.normal
        self._sampling_distribution_params = {"loc": 0, "scale": 1}


    def sample_noise(self, n):
        return self._sampling_distribution(**self._sampling_distribution_params, size=[n, self._input_dim])


    def set_sampling_distribution(self, func, params):
        self._sampling_distribution = func
        self._sampling_distribution_params = params


    def get_sampling_distribution(self):
        ret_string = "np.random." + self._sampling_distribution.__name__ + " "  + str(self._sampling_distribution_params)
        return ret_string


    def generate_samples(self, inpt, sess):
        return sess.run(self._output, feed_dict={self._input: inpt, self._is_training: False})


    def plot_samples(self, inpt, sess, image_shape, column_titles=None,
                     row_titles=None, nrows=None, epoch=None, path=None):
        generated_samples = self.inpt_to_image_grid(inpt=inpt, sess=sess, image_shape=image_shape, nrows=nrows)
        generated_samples = [[image.reshape(image_shape[:-1]) for image in row] for row in generated_samples]

        fig, ax = self.build_generated_samples(generated_samples, column_titles=column_titles,
                                               row_titles=row_titles, epoch=epoch, path=path)
        return fig, ax


    def inpt_to_image_grid(self, inpt, sess, image_shape, nrows=None):
        n = len(inpt)
        if nrows is None:
            nrows = int(np.sqrt(n))
        ncols = int(np.ceil(n / nrows))

        dim_x, dim_y = image_shape[1], image_shape[0]
        generated_samples = self.generate_samples(inpt=inpt, sess=sess).reshape([-1, *image_shape])
        generated_samples = [[image for image in generated_samples[r*ncols:(r+1)*ncols]] for r in range(nrows)]
        return generated_samples


    def build_generated_samples(self, samples, column_titles=None, row_titles=None, epoch=None, path=None):
        fig, ax = build_images(samples, column_titles=column_titles, row_titles=row_titles)
        if epoch is not None:
            plt.suptitle("Epoch {}:".format(epoch))

        if path is not None:
            plt.savefig(path)
        return fig, ax


    def generate_samples_n(self, n, sess):
        noise = self.sample_noise(n=n)
        return self.generate_samples(noise, sess=sess)



class ConditionalGenerator(Generator):

    def sample_noise(self, n, y_dim=None):
        if y_dim is None and self._y_dim is None:
            raise ValueError("Dimension of conditional space has to be given at least once.")
        elif self._y_dim is None:
            self._y_dim = y_dim
        return self._sampling_distribution(**self._sampling_distribution_params, size=[n, self._input_dim - self._y_dim])


    def generate_valid_input(self, inpt_y):
        inpt_z = self.sample_noise(n=len(inpt_y))
        inpt = np.concatenate([inpt_z, inpt_y], axis=1)
        return inpt


    def generate_samples_from_labels(self, inpt_y, sess):
        inpt = self.generate_valid_input(inpt_y=inpt_y)
        return self.generate_samples(inpt=inpt, sess=sess)


    def plot_samples_from_labels(self, inpt_y, sess, image_shape, column_titles=None,
                                 row_titles=None, reference=None, nrows=None, epoch=None, path=None):
        inpt = self.generate_valid_input(inpt_y=inpt_y)
        self.plot_samples_from_inpt(inpt=inpt, sess=sess, image_shape=image_shape, column_titles=column_titles,
                                    row_titles=row_titles, reference=reference, nrows=nrows, epoch=epoch, path=path)


    def plot_samples_from_inpt(self, inpt, sess, image_shape, column_titles=None,
                               row_titles=None, reference=None, nrows=None, epoch=None, path=None):
        if reference is None:
            self.plot_samples(inpt=inpt, sess=sess, image_shape=image_shape, column_titles=column_titles,
                              row_titles=row_titles, nrows=nrows, epoch=epoch, path=path)
        else:
            generated_samples = self.inpt_to_image_grid(inpt=inpt, sess=sess, image_shape=image_shape, nrows=nrows)
            assert len(generated_samples) == len(reference), "Different number of logged_image and logged_label"
            generated_samples = [[reference[i], *generated_samples[i]] for i in range(len(reference))]
            generated_samples = [[image.reshape(image_shape[:-1]) for image in row] for row in generated_samples]

            column_titles = ["Gen. {}".format(i+1) for i in range(len(reference))]
            column_titles.insert(0, "Ref.")

            fig, ax = self.build_generated_samples(generated_samples, column_titles=column_titles,
                                                   row_titles=row_titles, epoch=epoch, path=path)
            return fig, ax



class Decoder(Generator):

    def _define_sample_distribution(self):
        self._sampling_distribution = np.random.normal
        self._sampling_distribution_params = {"loc": 0, "scale": 1}


    def set_sampling_distribution(self, func, params):
        raise NotImplementedError("Sampling distribution fixed for Decoder structures.")

    def decode(self, inpt, sess):
        return self.generate_samples(inpt=inpt, sess=sess)



class ConditionalDecoder(ConditionalGenerator, Decoder):

    def _define_sample_distribution(self):
        self._sampling_distribution = np.random.normal
        self._sampling_distribution_params = {"loc": 0, "scale": 1}


    def set_sampling_distribution(self, func, params):
        raise NotImplementedError("Sampling distribution fixed for Decoder structures.")





############################################################################################################
# Discriminator structures
############################################################################################################


class Discriminator(NeuralNetwork):
    def get_accuracy(self, inpt, labels, sess):
        predicted_labels = self.predict_label(inpt, sess)
        accuracy = np.round(np.mean(np.equal(predicted_labels, labels))*100, 2)
        return accuracy


    def predict_label(self, inpt, sess):
        return(np.round(self.predict(inpt, sess)))

    def predict(self, inpt, sess):
        return sess.run(self._output, feed_dict={self._input: inpt, self._is_training: False})



class Critic(NeuralNetwork):
    def predict(self, inpt, sess):
        return sess.run(self._output, feed_dict={self._input: inpt, self._is_training: False})



############################################################################################################
# Encoder structures
############################################################################################################


class Encoder(NeuralNetwork):
    def get_network_params(self):
        network_params = tf.get_collection(tf.GraphKeys().TRAINABLE_VARIABLES, scope=self._name)
        network_params = [variable for variable in network_params if "Std" not in variable.name]
        return network_params

    def encode(self, inpt, sess):
        return sess.run(self._output, feed_dict={self._input: inpt, self._is_training: False})




############################################################################################################
# Utilities structures
############################################################################################################



def print_residual_block(params, prev_params, i, nn):
    nn.print_line(name="residual_block" + "-skips: {}".format(params["skip_layers"]))
    f_prev = int(nn._layers[i].shape[3])

    filters = [params["filters"]]*params["skip_layers"] if isinstance(params["filters"], int) else params["filters"]
    kernel_size = [params["kernel_size"]]*params["skip_layers"] if isinstance(params["kernel_size"], int) else params["kernel_size"]
    activation = [params["activation"]]*params["skip_layers"] if callable(params["activation"]) else params["activation"]
    nr_layer_params = 0
    for f, k, a in zip(filters, kernel_size, activation):
        layer_params = k*k*f*f_prev + f # weights + biases
        nr_layer_params += layer_params
        nn.print_line(i=i, name="conv2d", shape_weights=[k, k, f_prev, f], shape_bias=[f],
                        activation=a.__name__, output_shape=nn._layers[i+1].shape)
        i += 1
        f_prev = f

    nn.print_line(i=i, name="concat", nr_params=nr_layer_params, output_shape=nn._layers[i+1].shape)
    i += 1
    return(nr_layer_params, i)


def print_unet(params, prev_params, i, nn):
    nn.print_line(name="unet" + "-depth: {}".format(params["depth"]))
    nr_layer_params = 0
    f_prev = int(nn._layers[i].shape[3])
    a = params["activation"]
    for ii in range(params["depth"]):
        f = f_prev*2
        unet_layer_params = 2*2*f*f_prev + f # weights + biases
        nr_layer_params += unet_layer_params
        nn.print_line(i=i, name="conv2d", shape_weights=[2, 2, f_prev, f], shape_bias=[f],
                        activation=a.__name__, output_shape=nn._layers[i+1].shape)
        i += 1
        f_prev = f

    nn.print_line()
    f = f_prev
    unet_layer_params = 2*2*f*f_prev + f # weights + biases
    nr_layer_params += unet_layer_params
    nn.print_line(i=i, name="conv2d", shape_weights=[2, 2, f_prev, f], shape_bias=[f],
                    activation=a.__name__, output_shape=nn._layers[i+1].shape)
    i += 1
    unet_layer_params = 2*2*f*f_prev + f # weights + biases
    nr_layer_params += unet_layer_params
    nn.print_line(i=i, name="conv2d", shape_weights=[2, 2, f_prev, f], shape_bias=[f],
                    activation=a.__name__, output_shape=nn._layers[i+1].shape)
    i += 1
    nn.print_line(i=i, name="concat", output_shape=nn._layers[i+1].shape)
    i += 1
    nn.print_line()

    f_prev *= 2
    for ii in range(params["depth"]):
        f = int(f_prev/4)
        unet_layer_params = int(2*2*f*f_prev + f) # weights + biases
        nr_layer_params += unet_layer_params
        nn.print_line(i=i, name="conv2d_transpose", shape_weights=[2, 2, f_prev, f], shape_bias=[f],
                        activation=a.__name__, output_shape=nn._layers[i+1].shape)
        i += 1
        if ii == params["depth"]-1:
            nn.print_line(i=i, name="concat", nr_params=nr_layer_params, output_shape=nn._layers[i+1].shape)
            i += 1
        else:
            nn.print_line(i=i, name="concat", output_shape=nn._layers[i+1].shape)
            i += 1
        f_prev = f*2

    return(nr_layer_params, i)


def print_unet_original(params, prev_params, i, nn):
    nn.print_line(name="unet_original" + "-depth: {}".format(params["depth"]))
    nr_layer_params = 0
    f_prev = int(nn._layers[i].shape[3])
    nr_filters = []
    a = params["activation"]

    ### Downward sampling
    for ii in range(params["depth"]):
        f = params["filters"] if ii == 0 else f_prev*2
        unet_layer_params = 3*3*f*f_prev + f # weights + biases
        nr_layer_params += unet_layer_params
        nn.print_line(i=i, name="conv2d", shape_weights=[3, 3, f_prev, f], shape_bias=[f],
                        activation=a.__name__, output_shape=nn._layers[i+1].shape)
        i += 1
        f_prev = f

        unet_layer_params = 3*3*f*f_prev + f
        nr_layer_params += unet_layer_params
        nn.print_line(i=i, name="conv2d", shape_weights=[3, 3, f_prev, f], shape_bias=[f],
                        activation=a.__name__, output_shape=nn._layers[i+1].shape)
        i += 1

        if ii == params["depth"]-1:
            nn.print_line(i=i, name="Dropout(0.5)", output_shape=nn._layers[i+1].shape)
            i += 1

        nn.print_line(i=i, name="max_pooling2d", output_shape=nn._layers[i+1].shape)
        i += 1
        f_prev = f
        nr_filters.append(f)

    nr_filters = nr_filters[::-1]
    nn.print_line()

    ### Middle part
    f = f_prev
    unet_layer_params = 3*3*f*f_prev + f
    nr_layer_params += unet_layer_params
    nn.print_line(i=i, name="conv2d", shape_weights=[3, 3, f_prev, f], shape_bias=[f],
                    activation=a.__name__, output_shape=nn._layers[i+1].shape)
    i += 1
    nn.print_line(i=i, name="Dropout(0.5)", output_shape=nn._layers[i+1].shape)
    i += 1

    f_prev = f
    unet_layer_params = 3*3*f*f_prev + f
    nr_layer_params += unet_layer_params
    nn.print_line(i=i, name="conv2d", shape_weights=[3, 3, f_prev, f], shape_bias=[f],
                    activation=a.__name__, output_shape=nn._layers[i+1].shape)
    i += 1
    nn.print_line(i=i, name="Dropout(0.5)", output_shape=nn._layers[i+1].shape)
    i += 1
    nn.print_line()

    ### Upward sampling
    for ii in range(params["depth"]):
        nn.print_line(i=i, name="resize", output_shape=nn._layers[i+1].shape)
        i += 1
        f = nr_filters[ii]
        unet_layer_params = int(2*2*f*f_prev + f) # weights + biases
        nr_layer_params += unet_layer_params
        nn.print_line(i=i, name="conv2d", shape_weights=[2, 2, f_prev, f], shape_bias=[f],
                        activation=a.__name__, output_shape=nn._layers[i+1].shape)
        i += 1
        nn.print_line(i=i, name="concat", output_shape=nn._layers[i+1].shape)
        i += 1
        f_prev = f
        unet_layer_params = int(3*3*f*f_prev*2 + f) # weights + biases
        nr_layer_params += unet_layer_params
        nn.print_line(i=i, name="conv2d", shape_weights=[3, 3, f_prev*2, f], shape_bias=[f],
                        activation=a.__name__, output_shape=nn._layers[i+1].shape)
        i += 1
        unet_layer_params = int(3*3*f*f_prev + f) # weights + biases
        nr_layer_params += unet_layer_params
        if ii == params["depth"]-1:
            nn.print_line(i=i, name="conv2d", shape_weights=[3, 3, f_prev, f], shape_bias=[f],
                          nr_params=nr_layer_params, activation=a.__name__, output_shape=nn._layers[i+1].shape)
            i += 1
        else:
            nn.print_line(i=i, name="conv2d", shape_weights=[3, 3, f_prev, f], shape_bias=[f],
                            activation=a.__name__, output_shape=nn._layers[i+1].shape)
            i += 1

    return(nr_layer_params, i)


def print_inception_block(params, prev_params, i, nn):
    nn.print_line(name="inception")
    f_prev = int(nn._layers[i].shape[3])

    nr_layer_params = 0
    f = params["filters"]

    ### Layer 1
    unet_layer_params = 1*1*f*f_prev + f
    nr_layer_params += unet_layer_params
    nn.print_line(i=i, name="conv2d", shape_weights=[1, 1, f_prev, f], shape_bias=[f], nr_params="",
                    activation="relu", output_shape=nn._layers[i+1].shape)
    i += 1
    nn.print_line()

    ### Layer 2
    unet_layer_params = 1*1*f*f_prev + f
    nr_layer_params += unet_layer_params
    nn.print_line(i=i, name="conv2d", shape_weights=[1, 1, f_prev, f], shape_bias=[f], nr_params="",
                    activation="relu", output_shape=nn._layers[i+1].shape)
    i += 1
    unet_layer_params = 3*3*f*f + f
    nr_layer_params += unet_layer_params
    nn.print_line(i=i, name="conv2d", shape_weights=[3, 3, f, f], shape_bias=[f], nr_params="",
                    activation="relu", output_shape=nn._layers[i+1].shape)
    i += 1
    nn.print_line()

    ### Layer 3
    unet_layer_params = 1*1*f*f_prev + f
    nr_layer_params += unet_layer_params
    nn.print_line(i=i, name="conv2d", shape_weights=[1, 1, f_prev, f], shape_bias=[f], nr_params="",
                    activation="relu", output_shape=nn._layers[i+1].shape)
    i += 1
    unet_layer_params = 5*5*f*f + f
    nr_layer_params += unet_layer_params
    nn.print_line(i=i, name="conv2d", shape_weights=[5, 5, f, f], shape_bias=[f], nr_params="",
                    activation="relu", output_shape=nn._layers[i+1].shape)
    i += 1
    nn.print_line()

    ### Layer 4
    nn.print_line(i=i, name="max_pooling2d", shape_weights=[3, 3, f_prev, f], shape_bias=[f], nr_params="",
                    activation="relu", output_shape=nn._layers[i+1].shape)
    i += 1
    unet_layer_params = 1*1*f*f_prev + f
    nr_layer_params += unet_layer_params
    nn.print_line(i=i, name="conv2d", shape_weights=[1, 1, f_prev, f], shape_bias=[f], nr_params="",
                    activation="relu", output_shape=nn._layers[i+1].shape)
    i += 1
    nn.print_line()

    ### Concatenation
    nn.print_line(i=i, name="concat", shape_weights="", shape_bias="", nr_params=nr_layer_params,
                    activation="", output_shape=nn._layers[i+1].shape)
    i += 1
    return(nr_layer_params, i)


def print_concatenate_with(params, prev_params, i, nn):
    nn.print_line(i=i, name="concatenate_with {}".format(params["layer"]), nr_params=0, output_shape=nn._layers[i+1].shape)
    return(0, i+1)