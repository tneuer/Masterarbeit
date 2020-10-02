#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-25 15:41:09
    # Description :
####################################################################################
"""
import os
import sys
sys.path.insert(1, "../../Utilities/")
import json

import numpy as np
import tensorflow as tf

from layers import *
from Dataset import Dataset
from networks import Generator, ConditionalGenerator, NeuralNetwork

class GenerativeModel():
    def __init__(self, x_dim, z_dim, architectures, folder):
        if isinstance(x_dim, int):
            x_dim = [x_dim]
            self._image_shape = None
        else:
            if len(x_dim) != 3:
                raise ValueError("x_dim no integer, so expected three dimensions for x_dim. Given {}.".format(x_dim))
            self._image_shape = x_dim[:]

        self._x_dim = x_dim
        self._z_dim = z_dim
        self._architectures = architectures
        self._folder = folder

        for architecture in self._architectures:
            for i, layer in enumerate(architecture):
                if "name" not in layer[1]:
                    layer[1]["name"] = layer[0].__name__ + str(i)

        with tf.name_scope("Inputs"):
            self._X_input = tf.placeholder(tf.float32, shape=[None, *x_dim], name="x")
            self._Z_input = tf.placeholder(tf.float32, shape=[None, z_dim], name="z")
            self._is_training = tf.placeholder(tf.bool, shape=[], name="is_training")


    def _verify_init(self):
        valid_generators = ["Decoder", "Generator"]
        if hasattr(self, "_y_dim"):
            raise AttributeError("Unconditional generative model must not have attribute self._y_dim")
        if not hasattr(self, "_generator"):
            raise AttributeError("Generative model needs attribute self._generator")
        if type(self._generator).__name__ not in valid_generators:
            raise TypeError("self._generator needs to be one of: {}".format(valid_generators))
        if not hasattr(self, "_X_input"):
            raise AttributeError("Generative model needs attribute self._X_input for true input.")
        if not hasattr(self, "_Z_input"):
            raise AttributeError("Generative model needs attribute self._Z_input for noise.")
        if self._gen_architecture[-1][1]["name"] != "Output":
            raise NameError("Last layer of generator needs to be named 'Output' for loading purposes later.")


    def _init_folders(self):
        if not os.path.exists(self._folder):
            os.mkdir(self._folder)
            os.mkdir(self._folder+"/TFGraphs")
            if self._image_shape is not None:
                os.mkdir(self._folder+"/GeneratedSamples")
        self.save_as_json(save_path=self._folder+"/architecture.json")


    def sample_noise(self, n):
        return self._generator.sample_noise(n=n)


    def _get_vars(self, scope):
        return tf.get_collection(tf.GraphKeys().TRAINABLE_VARIABLES, scope=scope)


    def _summarise(self):
        self._log_time()
        if self._image_shape is not None:
            self._log_images()
        self._merged_summaries = tf.summary.merge_all()


    def _log_time(self):
        with tf.name_scope("Time"):
            self._epoch_time = tf.placeholder(tf.float32, name="Epoch_time")
            tf.summary.scalar("Epoch_time", self._epoch_time)
            self._total_time = tf.Variable(0., name="Total_time")
            self._total_time = self._total_time.assign(self._total_time + self._epoch_time)
            tf.summary.scalar("Total_time", self._total_time)
            self._epoch_nr = tf.placeholder(tf.int32)
            tf.summary.scalar("Epoch", self._epoch_nr)


    def _log_images(self):
        logged_noise = tf.get_variable(name="test_noise", dtype=tf.float32, shape=[10, self._z_dim], initializer=tf.random_uniform_initializer(-1, 1))
        self._testimagegenerator = Generator(self._gen_architecture, name="Generator")
        logged_output_gen = self._generator.generate_net(logged_noise)
        with tf.name_scope("Generator_output"):
            tf.summary.image("Out", tf.reshape(logged_output_gen, [-1, *self._image_shape]), max_outputs=9)


    def _set_up_training(self, log_step=None, gpu_options=None):
        init = tf.global_variables_initializer()
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self._sess.run(init)
        self._writer1 = tf.summary.FileWriter("{}/TensorboardData/Test".format(self._folder), self._sess.graph)
        self._writer2 = tf.summary.FileWriter("{}/TensorboardData/Train".format(self._folder), self._sess.graph)
        self._saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=1)
        self._log_step = log_step


    def _set_up_test_train_sample(self, x_train, x_test):
        if len(self._x_dim) > 1:
            if len(x_train.shape) != 4:
                raise ValueError("Expected 4 image dimensions: (nr_examples, height, width, channels). Given x_train: {}.".format(x_train.shape))
            if len(x_test.shape) != 4:
                raise ValueError("Expected 4 image dimensions: (nr_examples, height, width, channels). Given x_test: {}.".format(x_test.shape))
            if (x_train.shape[1] != self._x_dim[0]) or (x_train.shape[2] != self._x_dim[1]) or (x_train.shape[3] != self._x_dim[2]):
                raise ValueError("Wrong shape for x_train. Expected {}. Given {}.".format(self._x_dim, x_train.shape))
            if (x_test.shape[1] != self._x_dim[0]) or (x_test.shape[2] != self._x_dim[1]) or (x_test.shape[3] != self._x_dim[2]):
                raise ValueError("Wrong shape for x_test. Expected {}. Given {}.".format(self._x_dim, x_test.shape))
        self._trainset = Dataset(x_train)
        if x_test is None:
            nr_samples = 5000 if len(x_train)>5000 else len(x_train)
            self._x_test = self._trainset.sample(n=nr_samples, keep=True)
        else:
            self._x_test = x_test
        self._nr_test = len(self._x_test)
        self._z_test = self._generator.sample_noise(n=len(self._x_test))


    def _log(self, epoch, epoch_time):
        if self._log_step is not None:
            if (epoch-1) % self._log_step == 0:
                self._log_results(epoch, epoch_time)


    def _log_results(self, epoch, epoch_time):
        summary = self._sess.run(self._merged_summaries, feed_dict={self._X_input: self._x_test, self._Z_input: self._z_test,
                                 self._epoch_time: epoch_time, self._is_training: False, self._epoch_nr: epoch})
        self._writer1.add_summary(summary, epoch)
        nr_test = len(self._x_test)
        summary = self._sess.run(self._merged_summaries, feed_dict={self._X_input: self._trainset.get_xdata()[:nr_test],
                                 self._Z_input: self._z_test,
                                 self._epoch_time: epoch_time, self._is_training: False, self._epoch_nr: epoch})
        self._writer2.add_summary(summary, epoch)
        if self._image_shape is not None:
            self._generator.plot_samples(inpt=self.sample_noise(n=100), sess=self._sess, image_shape=self._image_shape,
                                     nrows=10, epoch=epoch, path="{}/GeneratedSamples/result_{}.png".format(self._folder, epoch))
        self.save_model(epoch)
        additional_log = getattr(self, "evaluate", None)
        if callable(additional_log):
            self.evaluate(true=self._x_test, condition=self._y_test, epoch=epoch)
        print("Logged.")


    def save_model(self, epoch=None):
        self._saver.save(self._sess, "{}/TFGraphs/graph".format(self._folder), global_step=epoch)


    def get_number_params(self):
        number_params = 0
        for net in self._nets:
            number_params += net.get_number_params()
        return number_params


    def show_architecture(self):
        parameters = 0
        for net in self._nets:
            parameters += net.show_architecture()
        print("\nTotal parameters:", parameters)
        return parameters


    def log_architecture(self, save_path=None, pre_message="", post_message=""):
        if save_path is None:
            save_path = self._folder + "/architecture_details.txt"
        sys_console = sys.stdout
        sys.stdout = open(save_path, 'w')

        pre_message = type(self).__name__ + "\n" +   "-"*len(type(self).__name__) + "\n" + pre_message
        print(pre_message)
        self.show_architecture()
        print(post_message)
        sys.stdout = sys_console


    def generate_samples(self, inpt):
        return self._generator.generate_samples(inpt, self._sess)


    def plot_samples(self, inpt, column_titles=None,
                        row_titles=None, reference=None, nrows=None, epoch=None, path=None):

        self._generator.plot_samples(
                inpt, self._sess, self._image_shape, column_titles, row_titles, nrows, epoch, path
        )



    def save_as_json(self, save_path=None):
        architecture_dict = {}
        for net in self._nets:
            architecture_dict.update(net.save_as_json(save_path=None))
        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(architecture_dict, f, indent=4)
            print("Architecture saved as .json!")
        return architecture_dict


    @classmethod
    def load_from_json(self, path):
        with open(path) as json_file:
            architectures = json.load(json_file)
            for network_type, network_architecture in architectures.items():
                for layer in network_architecture:
                    layer[0] = eval(layer[0])

                    try:
                        layer[1]["activation"] = eval(layer[1]["activation"])
                    except KeyError:
                        pass

        return architectures


    def set_sampling_distribution(self, func, params):
        self._generator.set_sampling_distribution(func, params)


    def get_sampling_distribution(self):
        return self._generator.get_sampling_distribution()



class ConditionalGenerativeModel(GenerativeModel):

    def __init__(self, x_dim, y_dim, z_dim, architectures, folder, append_y_at_every_layer):
        super(ConditionalGenerativeModel, self).__init__(x_dim, z_dim, architectures, folder)

        self._y_dim = y_dim
        if append_y_at_every_layer:
            self._append_at_every_layer = self._y_dim
        else:
            self._append_at_every_layer = None
        with tf.name_scope("Inputs"):
            self._Y_input = tf.placeholder(tf.float32, shape=[None, y_dim], name="y")
            self._mod_Z_input = tf.concat(axis=1, values=[self._Z_input, self._Y_input], name="mod_z")


    def _verify_init(self):
        valid_generators = ["ConditionalDecoder", "ConditionalGenerator"]
        if not hasattr(self, "_y_dim"):
            raise AttributeError("Conditional generative model needs attribute self._y_dim")
        if not hasattr(self, "_generator"):
            raise AttributeError("Generative model needs attribute self._generator")
        if type(self._generator).__name__ not in valid_generators:
            raise TypeError("self._generator needs to be one of: {}".format(valid_generators))
        if not hasattr(self, "_X_input"):
            raise AttributeError("Generative model needs attribute self._X_input for true input.")
        if not hasattr(self, "_Z_input"):
            raise AttributeError("Generative model needs attribute self._Z_input for noise.")
        if not hasattr(self, "_Y_input"):
            raise AttributeError("Conditional generative model needs attribute self._Y_input for condition.")
        if self._gen_architecture[-1][1]["name"] != "Output":
            raise NameError("Last layer of generator needs to be named 'Output' for loading purposes later.")
        self._generator._y_dim = self._y_dim


    def _summarise(self, logged_images=None, logged_labels=None):
        if (self._image_shape is None) and ((logged_images is not None) or (logged_labels is not None)):
            raise ValueError("Image shape must be given during intitialization of CWGAN if logging is active.")
        elif (logged_images is not None) and (logged_labels is None):
            raise ValueError("If logged_images is given, logged labels has to be given.")

        if logged_images is not None and (logged_images[0].shape[0] != self._image_shape[0] or logged_images[0].shape[1] != self._image_shape[1]):
            raise ValueError("Logged images have wrong shape. Expected {}. Given {}.".format(self._image_shape, logged_images.shape))
        self._logged_images = logged_images
        self._logged_labels = logged_labels
        super(ConditionalGenerativeModel, self)._summarise()


    def _log_images(self):
        if self._logged_images is not None:
            if self._logged_labels is not None:
                logged_noise = tf.get_variable(
                                               name="test_noise", dtype=tf.float32,
                                               shape=[len(self._logged_labels), self._z_dim],
                                               initializer=tf.random_uniform_initializer(-1, 1)
                                               )
                logged_labels = tf.constant(name="TestLabel", value=self._logged_labels, dtype=tf.float32)
                logged_noise = tf.concat(axis=1, values=[logged_noise, logged_labels], name="TestImages")
                self._testimagegenerator = ConditionalGenerator(self._gen_architecture, name="Generator")
                logged_output_gen = self._testimagegenerator.generate_net(inpt=logged_noise, append_elements_at_every_layer=self._append_at_every_layer)
                with tf.name_scope("Generator_output"):
                    tf.summary.image("Out", tf.reshape(logged_output_gen, [-1, *self._image_shape]), max_outputs=9)
            else:
                super(ConditionalGenerativeModel, self)._log_images()


    def _set_up_test_train_sample(self, x_train, y_train, x_test, y_test):
        if len(self._x_dim) > 1:
            if len(x_train.shape) != 4:
                raise ValueError("Expected 4 image dimensions: (nr_examples, height, width, channels). Given x_train: {}.".format(x_train.shape))
            if len(x_test.shape) != 4:
                raise ValueError("Expected 4 image dimensions: (nr_examples, height, width, channels). Given x_test: {}.".format(x_test.shape))
            if (x_train.shape[1] != self._x_dim[0]) or (x_train.shape[2] != self._x_dim[1]) or (x_train.shape[3] != self._x_dim[2]):
                raise ValueError("Wrong shape for x_train. Expected {}. Given {}.".format(self._x_dim, x_train.shape))
            if (x_test.shape[1] != self._x_dim[0]) or (x_test.shape[2] != self._x_dim[1]) or (x_test.shape[3] != self._x_dim[2]):
                raise ValueError("Wrong shape for x_test. Expected {}. Given {}.".format(self._x_dim, x_test.shape))
        self._trainset = Dataset(x_train, y_train)
        if x_test is None or y_test is None:
            nr_samples = 5000 if len(x_train)>5000 else len(x_train)
            self._x_test, self._y_test = self._trainset.sample(n=nr_samples, keep=True)
        else:
            self._x_test, self._y_test = x_test, y_test
        self._nr_test = len(self._x_test)
        self._z_test = self._generator.sample_noise(n=len(self._x_test))


    def _log_results(self, epoch, epoch_time):
        summary = self._sess.run(self._merged_summaries, feed_dict={self._X_input: self._x_test, self._Y_input: self._y_test,
                                 self._Z_input: self._z_test, self._epoch_time: epoch_time, self._is_training: False, self._epoch_nr: epoch})
        self._writer1.add_summary(summary, epoch)
        nr_test = len(self._x_test)
        summary = self._sess.run(self._merged_summaries, feed_dict={self._X_input: self._trainset.get_xdata()[:nr_test],
                                 self._Z_input: self._z_test, self._Y_input: self._trainset.get_ydata()[:nr_test],
                                 self._epoch_time: epoch_time, self._is_training: False, self._epoch_nr: epoch})
        self._writer2.add_summary(summary, epoch)
        if self._image_shape is not None:
            nr_reps = 10
            inpt_y = np.repeat(self._logged_labels, nr_reps, axis=0)
            nrow = int(len(inpt_y)/nr_reps)
            self._generator.plot_samples_from_labels(
                                        inpt_y=inpt_y, sess=self._sess, image_shape=self._image_shape,
                                        reference=self._logged_images, nrows=nrow,
                                        path="{}/GeneratedSamples/Samples_{}.png".format(self._folder, epoch)
            )
        self.save_model(epoch)
        additional_log = getattr(self, "evaluate", None)
        if callable(additional_log):
            self.evaluate(true=self._x_test, condition=self._y_test, epoch=epoch)
        print("Logged.")


    def generate_samples(self, inpt):
        return self._generator.generate_samples_from_labels(inpt, self._sess)


    def plot_samples(self, inpt, column_titles=None,
                        row_titles=None, reference=None, nrows=None, epoch=None, path=None):
        self._generator.plot_samples_from_labels(
                inpt, self._sess, self._image_shape, column_titles, row_titles, reference, nrows, epoch, path
            )



class Image2ImageGenerativeModel(GenerativeModel):
    def __init__(self, x_dim, y_dim, architectures, folder):
        if isinstance(x_dim, int):
            x_dim = [x_dim]
        else:
            if len(x_dim) != 3:
                raise ValueError("x_dim no integer, so expected three dimensions for x_dim. Given {}.".format(x_dim))
            self._image_shape = x_dim[:]

        self._x_dim = x_dim
        self._y_dim = y_dim
        self._architectures = architectures
        self._folder = folder

        for architecture in self._architectures:
            for i, layer in enumerate(architecture):
                if "name" not in layer[1]:
                    layer[1]["name"] = layer[0].__name__ + str(i)

        with tf.name_scope("Inputs"):
            self._X_input = tf.placeholder(tf.float32, shape=[None, *x_dim], name="x")
            self._Y_input = tf.placeholder(tf.float32, shape=[None, *y_dim], name="y")
            self._is_training = tf.placeholder(tf.bool, shape=[], name="is_training")


    def _verify_init(self):
        if not hasattr(self, "_X_input"):
            raise AttributeError("Generative model needs attribute self._X_input for true input.")
        if not hasattr(self, "_Y_input"):
            raise AttributeError("Image-to-Image model needs attribute self._Y_input for noise.")


    def sample_noise(self, n):
        raise AttributeError("sample_noise not implemented for CycleGAN.")


    def _summarise(self):
        self._log_time()
        self._merged_summaries = tf.summary.merge_all()


    def _set_up_test_train_sample(self, x_train, y_train, x_test, y_test):
        if len(x_train.shape) != 4:
            raise ValueError("Expected 4 image dimensions: (nr_examples, height, width, channels). Given x_train: {}.".format(x_train.shape))
        if len(x_test.shape) != 4:
            raise ValueError("Expected 4 image dimensions: (nr_examples, height, width, channels). Given x_test: {}.".format(x_test.shape))
        if (x_train.shape[1] != self._x_dim[0]) or (x_train.shape[2] != self._x_dim[1]) or (x_train.shape[3] != self._x_dim[2]):
            raise ValueError("Wrong shape for x_train. Expected {}. Given {}.".format(self._x_dim, x_train.shape))
        if (x_test.shape[1] != self._x_dim[0]) or (x_test.shape[2] != self._x_dim[1]) or (x_test.shape[3] != self._x_dim[2]):
            raise ValueError("Wrong shape for x_test. Expected {}. Given {}.".format(self._x_dim, x_test.shape))
        self._trainset = Dataset(x_train, y_train)
        if x_test is None or y_test is None:
            nr_samples = 5000 if len(x_train)>5000 else len(x_train)
            self._x_test, self._y_test = self._trainset.sample(n=nr_samples, keep=True)
        else:
            self._x_test, self._y_test = x_test, y_test
        self._nr_test = len(self._x_test)


    def _log_results(self, epoch, epoch_time):
        summary = self._sess.run(self._merged_summaries, feed_dict={self._X_input: self._x_test, self._Y_input: self._y_test,
                                 self._epoch_time: epoch_time, self._is_training: False, self._epoch_nr: epoch})
        self._writer1.add_summary(summary, epoch)
        nr_test = len(self._x_test)
        summary = self._sess.run(self._merged_summaries, feed_dict={self._X_input: self._trainset.get_xdata()[:nr_test],
                                 self._Y_input: self._trainset.get_ydata()[:nr_test],
                                 self._epoch_time: epoch_time, self._is_training: False, self._epoch_nr: epoch})
        self._writer2.add_summary(summary, epoch)
        if self._image_shape is not None:
            self.plot_samples(inpt_x=self._x_test[:10], inpt_y=self._y_test[:10], sess=self._sess, image_shape=self._image_shape,
                                epoch=epoch, path="{}/GeneratedSamples/result_{}.png".format(self._folder, epoch))
        self.save_model(epoch)
        additional_log = getattr(self, "evaluate", None)
        if callable(additional_log):
            self.evaluate(true=self._x_test, condition=self._y_test, epoch=epoch)
        print("Logged.")


    def plot_samples(self, inpt_x, inpt_y, sess, image_shape, epoch, path):
        outpt_xy = sess.run(self._output_gen, feed_dict={self._X_input: inpt_x, self._is_training: False})

        image_matrix = np.array([
            [
                x.reshape(image_shape[0], image_shape[1]), y.reshape(image_shape[0], image_shape[1]),
                np.zeros(shape=(32, 32)),
                xy.reshape(image_shape[0], image_shape[1])
            ]
                for x, y, xy in zip(inpt_x, inpt_y, outpt_xy)
        ])
        self._generator.build_generated_samples(image_matrix,
                                                   column_titles=["True X", "True Y", "", "Gen_XY", "Gen_XYX",
                                                   "Gen_YX", "Gen_YXY", "Gen_XY_YX", "Gen_YX_XY"],
                                                    epoch=epoch, path=path)


    def get_sampling_distribution(self):
        return None



class CyclicGenerativeModel(Image2ImageGenerativeModel):

    def plot_samples(self, inpt_x, inpt_y, sess, image_shape, epoch, path):
        outpt_xy = sess.run(self._output_gen_xy, feed_dict={self._X_input: inpt_x, self._is_training: False})
        outpt_xyx = sess.run(self._output_gen_xyx, feed_dict={self._X_input: inpt_x, self._is_training: False})
        outpt_yx = sess.run(self._output_gen_yx, feed_dict={self._Y_input: inpt_y, self._is_training: False})
        outpt_yxy = sess.run(self._output_gen_yxy, feed_dict={self._Y_input: inpt_y, self._is_training: False})

        outpt_xy_yx = sess.run(self._output_gen_yx, feed_dict={self._Y_input: outpt_xy, self._is_training: False})
        outpt_yx_xy = sess.run(self._output_gen_xy, feed_dict={self._X_input: outpt_yx, self._is_training: False})

        image_matrix = np.array([
            [
                x.reshape(image_shape[0], image_shape[1]), y.reshape(image_shape[0], image_shape[1]),
                np.zeros(shape=(32, 32)),
                xy.reshape(image_shape[0], image_shape[1]), xyx.reshape(image_shape[0], image_shape[1]),
                yx.reshape(image_shape[0], image_shape[1]), yxy.reshape(image_shape[0], image_shape[1]),
                xy_yx.reshape(image_shape[0], image_shape[1]), yx_xy.reshape(image_shape[0], image_shape[1]),
            ]
                for x, y, xy, xyx, yx, yxy, xy_yx, yx_xy in zip(inpt_x, inpt_y, outpt_xy, outpt_xyx, outpt_yx, outpt_yxy, outpt_xy_yx, outpt_yx_xy)
        ])
        self._generator_xy.build_generated_samples(image_matrix,
                                                   column_titles=["True X", "True Y", "", "Gen_XY", "Gen_XYX",
                                                   "Gen_YX", "Gen_YXY", "Gen_XY_YX", "Gen_YX_XY"],
                                                    epoch=epoch, path=path)



