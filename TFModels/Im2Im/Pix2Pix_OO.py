#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-08-10 11:09:04
    # Description :
####################################################################################
"""
import os
import sys
import time
import json
import copy
sys.path.insert(1, "../..")
sys.path.insert(1, "../building_blocks")
sys.path.insert(1, "../../Utilities")
import grid_search

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from layers import logged_dense, conv2d_logged, conv2d_transpose_logged, reshape_layer
from layers import concatenate_with, residual_block, unet, unet_original, inception_block
from networks import Generator, Discriminator
from generativeModels import Image2ImageGenerativeModel, GenerativeModel
from functionsOnImages import padding_zeros
import Preprocessing.initialization as init
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, crop_images


class Pix2PixGAN(Image2ImageGenerativeModel):
    def __init__(self, x_dim, y_dim, gen_architecture, disc_architecture,
                    folder="./Pix2Pix", is_patchgan=False, is_wasserstein=False
        ):
        super(Pix2PixGAN, self).__init__(x_dim, y_dim, [gen_architecture, disc_architecture], folder)

        self._gen_architecture = self._architectures[0]
        self._disc_architecture = self._architectures[1]
        self._is_patchgan = is_patchgan
        self._is_wasserstein = is_wasserstein

        ################# Define architecture

        if self._is_patchgan:
            f_xy = self._disc_architecture[-1][-1]["filters"]
            assert f_xy == 1, "If is_patchgan, last layer of Discriminator_XY needs 1 filter. Given: {}.".format(f_xy)

            a_xy = self._disc_architecture[-1][-1]["activation"]
            if self._is_wasserstein:
                assert a_xy == tf.identity, "If is_patchgan, last layer of Discriminator_XY needs tf.nn.identity. Given: {}.".format(a_xy)
            else:
                assert a_xy == tf.nn.sigmoid, "If is_patchgan, last layer of Discriminator_XY needs tf.nn.sigmoid. Given: {}.".format(a_xy)
        else:
            self._disc_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            if self._is_wasserstein:
                self._disc_architecture.append([logged_dense, {"units": 1, "activation": tf.identity, "name": "Output"}])
            else:
                self._disc_architecture.append([logged_dense, {"units": 1, "activation": tf.nn.sigmoid, "name": "Output"}])

        self._gen_architecture[-1][1]["name"] = "Output"

        self._generator = Generator(self._gen_architecture, name="Generator")
        self._discriminator = Discriminator(self._disc_architecture, name="Discriminator")

        self._nets = [self._generator, self._discriminator]

        ################# Connect inputs and networks
        self._output_gen = self._generator.generate_net(self._X_input, tf_trainflag=self._is_training)

        with tf.name_scope("InputsMod"):
            gen_out_shape = self._output_gen.get_shape()
            x_shape = self._X_input.get_shape()
            y_shape = self._Y_input.get_shape()
            assert gen_out_shape[1] == x_shape[1] and gen_out_shape[2] == x_shape[2], (
                "Wrong shapes: Generator output has {}, while X input has {}.".format(gen_out_shape, x_shape)
            )
            assert y_shape[1] == x_shape[1] and y_shape[2] == x_shape[2], (
                "Wrong shapes: Y input has {}, while X input has {}.".format(y_shape, x_shape)
            )
            self._output_gen_mod = tf.concat(values=[self._output_gen, self._X_input], axis=3, name="mod_gen")
            self._Y_input_mod = tf.concat(values=[self._Y_input, self._X_input], axis=3, name="mod_y")

        self._output_disc_real = self._discriminator.generate_net(self._Y_input_mod, tf_trainflag=self._is_training)
        self._output_disc_fake = self._discriminator.generate_net(self._output_gen_mod, tf_trainflag=self._is_training)

        if self._is_patchgan:
            print("PATCHGAN chosen with output: {}.".format(self._output_disc_real.shape))

        ################# Finalize

        self._init_folders()
        self._verify_init()


    def compile(self, learning_rate=0.0005, learning_rate_gen=None, learning_rate_disc=None,
                    optimizer=tf.train.AdamOptimizer, lmbda=10, loss="cross-entropy", label_smoothing=1):
        if self._is_wasserstein and loss != "wasserstein":
            raise ValueError("If is_wasserstein is true in Constructor, loss needs to be wasserstein.")
        if not self._is_wasserstein and loss == "wasserstein":
            raise ValueError("If loss is wasserstein, is_wasserstein needs to be true in constructor.")

        if learning_rate_gen is None:
            learning_rate_gen = learning_rate
        if learning_rate_disc is None:
            learning_rate_disc = learning_rate
        self._define_loss(lmbda=lmbda, loss=loss, label_smoothing=label_smoothing)
        with tf.name_scope("Optimizer"):
            self._gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_optimizer_op = self._gen_optimizer.minimize(self._gen_loss,
                                                         var_list=self._get_vars(scope="Generator"),
                                                         name="Generator")
            self._disc_optimizer = optimizer(learning_rate=learning_rate_disc)
            self._disc_optimizer_op= self._disc_optimizer.minimize(self._disc_loss,
                                                            var_list=self._get_vars(scope="Discriminator"),
                                                            name="Discriminator")
        self._summarise()


    def _define_loss(self, lmbda, loss, label_smoothing):
        possible_losses = ["cross-entropy", "L2", "wasserstein"]
        def get_labels_one(tensor):
            return tf.ones_like(tensor)*label_smoothing
        def get_labels_zero(tensor):
            return tf.zeros_like(tensor) + 1 - label_smoothing
        eps = 1e-7
        self._label_smoothing = label_smoothing
        if loss == "wasserstein":
            self._generator_loss = -tf.reduce_mean(self._output_disc_fake)
            self._disc_loss = (-(tf.reduce_mean(self._output_disc_real) -
                                    tf.reduce_mean(self._output_disc_fake)) +
                                    10*self._define_gradient_penalty()
            )
        elif loss == "cross-entropy":
            self._logits_real = tf.math.log( self._output_disc_real / (1+eps - self._output_disc_real) + eps)
            self._logits_fake = tf.math.log( self._output_disc_fake / (1+eps - self._output_disc_fake) + eps)
            self._generator_loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.ones_like(self._logits_fake), logits=self._logits_fake
                                    )
            )
            self._disc_loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_one(self._logits_real), logits=self._logits_real
                                    ) +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_zero(self._logits_fake), logits=self._logits_fake
                                    )
            )

        elif loss == "L2":
            self._generator_loss = tf.reduce_mean(tf.square(self._output_disc_fake - tf.ones_like(self._output_disc_fake)))
            self._disc_loss = (
                        tf.reduce_mean(
                                        tf.square(self._output_disc_real - get_labels_one(self._output_disc_real)) +
                                        tf.square(self._output_disc_fake)
                        )
            ) / 2.0
        elif loss == "KL":
            self._logits_real = tf.math.log( self._output_disc_real / (1+eps - self._output_disc_real) + eps)
            self._logits_fake = tf.math.log( self._output_disc_fake / (1+eps - self._output_disc_fake) + eps)
            self._generator_loss = -tf.reduce_mean(self._logits_fake)
            self._disc_loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_one(self._logits_real), logits=self._logits_real
                                    ) +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_zero(self._logits_fake), logits=self._logits_fake
                                    )
            )
        else:
            raise ValueError("Loss not implemented. Choose from {}. Given: {}.".format(possible_losses, loss))

        with tf.name_scope("Loss") as scope:
            tf.summary.scalar("Generator_vanilla_loss", self._generator_loss)
            self._recon_loss = lmbda*tf.reduce_mean(tf.abs(self._Y_input - self._output_gen))
            tf.summary.scalar("Generator_recon_loss", self._recon_loss)
            self._gen_loss = self._generator_loss + self._recon_loss
            tf.summary.scalar("Generator_total_loss", self._gen_loss)
            tf.summary.scalar("Discriminator_loss", self._disc_loss)


    def _define_gradient_penalty(self):
        alpha = tf.random_uniform(shape=tf.shape(self._Y_input), minval=0., maxval=1.)
        differences = self._output_gen - self._Y_input
        interpolates = self._Y_input + (alpha * differences)
        interpolates = tf.concat(values=[interpolates, self._X_input], axis=3)
        gradients = tf.gradients(self._discriminator.generate_net(interpolates, tf_trainflag=self._is_training), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        with tf.name_scope("Loss") as scope:
            self._gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            tf.summary.scalar("Gradient_penalty", self._gradient_penalty)
        return self._gradient_penalty


    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=100, batch_size=64,
              gen_steps=1, disc_steps=5, steps=None, log_step=3, batch_log_step=None, gpu_options=None):
        if steps is not None:
            gen_steps = 1
            disc_steps = steps
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        self._set_up_test_train_sample(x_train, y_train, x_test, y_test)
        self._prepare_monitoring()
        self._log_results(epoch=0, epoch_time=0)
        nr_batches = np.floor(len(x_train) / batch_size)

        for epoch in range(epochs):
            batch_nr = 0
            disc_loss_epoch = 0
            gen_loss_epoch = 0
            start = time.clock()
            trained_examples = 0

            while trained_examples < len(x_train):
                disc_loss_batch, gen_loss_batch = self._optimize(self._trainset, batch_size, disc_steps, gen_steps)
                trained_examples += batch_size
                disc_loss_epoch += disc_loss_batch
                gen_loss_epoch += gen_loss_batch

                if (batch_log_step is not None) and (batch_nr % batch_log_step == 0):
                    self._count_batches += batch_log_step
                    batch_train_time = (time.clock() - start)/60
                    self._log(self._count_batches, batch_train_time)
                batch_nr += 1

            disc_loss_epoch /= nr_batches
            gen_loss_epoch /= nr_batches

            epoch_train_time = (time.clock() - start)/60
            disc_loss_epoch = np.round(disc_loss_epoch, 7)
            gen_loss_epoch = np.round(gen_loss_epoch, 7)

            print("Epoch {}: Discrimiantor: {} \n\t\t\tGenerator: {}.".format(epoch+1, disc_loss_epoch, gen_loss_epoch))

            if batch_log_step is None and (log_step is not None) and (epoch % log_step == 0):
                self._log(epoch+1, epoch_train_time)


    def _optimize(self, dataset, batch_size, disc_steps, gen_steps):
        d_loss = 0
        for i in range(disc_steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            _, disc_loss_batch = self._sess.run([
                                            self._disc_optimizer_op, self._disc_loss
                                            ],
                                            feed_dict={self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                            self._is_training: True})
            d_loss += disc_loss_batch
        d_loss /= disc_steps

        g_loss = 0
        for _ in range(gen_steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            _, gen_loss_batch = self._sess.run([self._gen_optimizer_op, self._gen_loss],
                                               feed_dict={self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                               self._is_training: True})
            g_loss += gen_loss_batch
        g_loss /= gen_steps
        return d_loss, g_loss


    def _prepare_monitoring(self):
        self._total_train_time = 0
        self._total_log_time = 0
        self._count_batches = 0
        self._batches = []

        self._max_allowed_failed_checks = 20
        self._gen_grads_and_vars = self._gen_optimizer.compute_gradients(self._gen_loss, var_list=self._get_vars("Generator"))
        self._disc_grads_and_vars = self._disc_optimizer.compute_gradients(self._disc_loss, var_list=self._get_vars("Discriminator"))

        self._monitor_dict = {
            "Gradients": [
                [self._gen_grads_and_vars, self._disc_grads_and_vars],
                ["Generator", "Adversarial"],
                [[] for i in range(6)]
            ],
            "Losses": [
                    [self._gen_loss, self._disc_loss, self._generator_loss, self._recon_loss],
                    ["Generator (V+R)", "Adversarial", "Vanilla_Generator", "Reconstruction"],
                    [[] for i in range(4)]
            ],
            "Output Adversarial": [
                    [self._output_disc_fake, self._output_disc_real],
                    ["Fake", "Real"],
                    [[] for i in range(3)],
                    [np.mean]
            ]
        }

        self._check_dict = {
            "Dominating Discriminator": {
                "Tensors": [self._output_disc_real, self._output_disc_fake],
                "OPonTensors": [np.mean, np.mean],
                "Relation": [">", "<"], "Threshold": [self._label_smoothing*0.95, (1-self._label_smoothing)*1.05],
                "TensorRelation": np.logical_and
            },
            "Generator outputs zeros": {
                "Tensors": [self._output_gen],
                "OPonTensors": [np.max],
                "Relation": ["<"], "Threshold": [0.05]
            }
        }
        self._check_count = [0 for key in self._check_dict]


    def evaluate(self, true, condition, epoch):
        log_start = time.clock()
        self._batches.append(epoch)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))
        axs = np.ravel(axs)
        if "Gradients" in self._monitor_dict:
            colors = ["green", "blue", "red"]
            axy_min = np.inf
            axy_max = -np.inf
            for go, gradient_ops in enumerate(self._monitor_dict["Gradients"][0]):
                grads = [self._sess.run(gv[0], feed_dict={
                            self._X_input: self._x_test, self._Y_input: self._y_test, self._is_training: False
                            }) for gv in gradient_ops]

                for op_idx, op in enumerate([np.mean, np.max, np.min]):
                    self._monitor_dict["Gradients"][2][go*3+op_idx].append(op([op(grad) for grad in grads]))
                    vals = self._monitor_dict["Gradients"][2][go*3+op_idx]
                    if op_idx == 0:
                        axs[0].plot(self._batches, vals, label=self._monitor_dict["Gradients"][1][go], color=colors[go])
                    else:
                        axs[0].plot(self._batches, vals, linewidth=0.5, linestyle="--", color=colors[go])
                        upper = np.mean(vals)
                        lower = np.mean(vals)
                        if upper > axy_max:
                            axy_max = upper
                        if lower < axy_min:
                            axy_min = lower
        axs[0].set_title("Gradients")
        axs[0].legend()
        axs[0].set_ylim([axy_min, axy_max])

        current_batch_x, current_batch_y = self._trainset.get_next_batch(batch_size)

        print("Batch ", epoch)
        colors = ["green", "blue", "red", "orange", "purple", "brown"]
        for k, key in enumerate(self._monitor_dict):
            if key == "Gradients":
                continue
            key_results = self._sess.run(
                self._monitor_dict[key][0],
                feed_dict={
                    self._X_input: current_batch_x, self._Y_input: current_batch_y, self._is_training: True
            })
            for kr, key_result in enumerate(key_results):
                try:
                    self._monitor_dict[key][2][kr].append(self._monitor_dict[key][3][0](key_result))
                except IndexError:
                    self._monitor_dict[key][2][kr].append(key_result)
                axs[k].plot(self._batches, self._monitor_dict[key][2][kr], label=self._monitor_dict[key][1][kr], color=colors[kr])
            axs[k].legend()
            axs[k].set_title(key)
            print("; ".join([
                "{}: {}".format(name, round(float(val[-1]), 5))
                    for name, val in zip(self._monitor_dict[key][1], self._monitor_dict[key][2])
            ]))

        gen_samples = self._sess.run([self._output_gen],
            feed_dict={
                self._X_input: current_batch_x, self._is_training: False
        })
        axs[-1].hist([np.ravel(gen_samples), np.ravel(current_batch_y)], label=["Generated", "True"])
        axs[-1].set_title("Pixel distribution")
        axs[-1].legend()

        for check_idx, check_key in enumerate(self._check_dict):
            result_bools_of_check = []
            check = self._check_dict[check_key]
            for tensor_idx in range(len(check["Tensors"])):
                tensor_ = self._sess.run(check["Tensors"][tensor_idx], feed_dict={
                    self._X_input: self._x_test, self._Y_input: self._y_test, self._is_training: False
                })
                tensor_op = check["OPonTensors"][tensor_idx](tensor_)
                if eval(str(tensor_op) + check["Relation"][tensor_idx] + str(check["Threshold"][tensor_idx])):
                    result_bools_of_check.append(True)
                else:
                    result_bools_of_check.append(False)
            if ( tensor_idx > 0 and check["TensorRelation"](*result_bools_of_check) ) or ( result_bools_of_check[0] ):
                self._check_count[check_idx] += 1
                if self._check_count[check_idx] == self._max_allowed_failed_checks:
                    raise GeneratorExit(check_key)
            else:
                self._check_count[check_idx] = 0

        self._total_log_time += (time.clock() - log_start)
        fig.suptitle("Train {} / Log {} / Fails {}".format(np.round(self._total_train_time, 2), np.round(self._total_log_time, 2),
                                                            self._check_count))
        plt.savefig(self._folder+"/TrainStatistics.png")


if __name__ == '__main__':
    ########### Image input
    if "lhcb_data2" in os.getcwd():
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        gpu_frac = 0.3
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
        print("1 GPU limited to {}% memory.".format(np.round(gpu_frac*100)))
    else:
        gpu_options = None

    import pickle
    if "lhcb_data2" in os.getcwd():
        data_path = "../../../Data/fashion_mnist"
    else:
        data_path = "/home/tneuer/Backup/Algorithmen/0TestData/image_to_image/fashion_mnist"
    with open("{}/train_images.pickle".format(data_path), "rb") as f:
        x_train_orig = (pickle.load(f)[0].reshape(-1, 28, 28, 1) / 255)[:20000]
        x_test_orig = x_train_orig[-100:]
        x_train_orig = x_train_orig[:-100]
    with open("{}/train_images_rotated.pickle".format(data_path), "rb") as f:
        y_train_orig = (pickle.load(f).reshape(-1, 28, 28, 1) / 255)[:20000]
        y_test_orig = y_train_orig[-100:]
        y_train_orig = y_train_orig[:-100]


    ########### Architecture

    x_train_orig = padding_zeros(x_train_orig, top=2, bottom=2, left=2, right=2)
    x_test_orig = padding_zeros(x_test_orig, top=2, bottom=2, left=2, right=2)
    y_train_orig = padding_zeros(y_train_orig, top=2, bottom=2, left=2, right=2)
    y_test_orig = padding_zeros(y_test_orig, top=2, bottom=2, left=2, right=2)
    inpt_dim = x_train_orig[0].shape
    opt_dim = y_train_orig[0].shape

    param_dict = {
            "disc_steps": [1, 5],
            "architecture": ["keraslike"],
            "batch_size": [8, 16, 32],
            "gen_steps": [1],
            "is_patchgan": [True, False],
            "invert_images": [True, False],
            "lmbda": [0.1, 1, 10],
            "loss": ["cross-entropy", "KL", "L2"],
            "lr": [0.0001, 0.00005, 0.00001],
            "lr_disc": [0.000005],
            "optimizer": [tf.train.RMSPropOptimizer, tf.train.AdamOptimizer],
    }
    sampled_params = grid_search.get_parameter_grid(param_dict=param_dict, n=50, allow_repetition=True)

    for params in sampled_params:

        disc_steps = int(params["disc_steps"])
        architecture_path = "../../../Architectures/Im2Im/Pix2Pix/{}.json".format(params["architecture"])
        batch_size = int(params["batch_size"])
        learning_rate = float(params["lr"])
        learning_rate_disc = float(params["lr_disc"])
        lmbda = int(params["lmbda"])
        optimizer = params["optimizer"]
        loss = str(params["loss"])
        is_patchGAN = bool(params["is_patchgan"])
        invert_images = bool(params["invert_images"])

        x_train = x_train_orig
        x_test = x_test_orig
        y_train = y_train_orig
        y_test = y_test_orig
        if invert_images:
            x_train = 1-x_train_orig
            x_test = 1-x_test_orig
            y_train = 1-y_train_orig
            y_test = 1-y_test_orig

        epochs = 15
        batch_log_step = int(len(x_train) / batch_size / 20)
        gen_steps = 1
        label_smoothing = 0.95

        architectures = GenerativeModel.load_from_json(architecture_path)
        gen_architecture = architectures["Generator"]
        disc_architecture = architectures["Discriminator"]

        if is_patchGAN:
            disc_architecture[-1][1]["activation"] = tf.nn.sigmoid
            disc_architecture[-1][1]["filters"] = 1
        else:
            disc_architecture[-1][1]["activation"] = tf.nn.leaky_relu


        path_saving = "../../../Results/Test"
        path_saving = init.initialize_folder(algorithm="Pix2Pix_", base_folder=path_saving)

        init_params = {
            "x_dim": inpt_dim, "y_dim": opt_dim, "gen_architecture": gen_architecture,
            "disc_architecture": disc_architecture, "folder": path_saving,
            "is_patchgan": is_patchGAN
        }
        compile_params = {
            "loss": loss, "learning_rate": learning_rate, "learning_rate_disc": learning_rate_disc,
            "optimizer": optimizer, "lmbda": lmbda, "label_smoothing": label_smoothing

        }
        train_params = {
            "x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test":y_test, "epochs": epochs, "batch_size": batch_size,
            "disc_steps": disc_steps, "gen_steps": gen_steps, "log_step": 1, "batch_log_step": batch_log_step,
            "gpu_options": gpu_options
        }

        config_data = copy.deepcopy(init_params); config_data.update(compile_params); config_data.update(train_params)
        config_data["nr_train"] = len(x_test)
        config_data["optimizer"] = config_data["optimizer"].__name__
        for key in ["x_train", "y_train", "x_test", "y_test", "gpu_options"]:
            config_data.pop(key)
        config_data = init.function_to_string(config_data)

        grid_search.run_agorithm(Pix2PixGAN, init_params=init_params, compile_params=compile_params, train_params=train_params,
                                 path_saving=path_saving, config_data=config_data)
        tf.reset_default_graph()
