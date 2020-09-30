#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-08-20 16:50:44
    # Description :
####################################################################################
"""
import os
import sys
import time
import json
sys.path.insert(1, "../..")
sys.path.insert(1, "../building_blocks")
sys.path.insert(1, "../../Utilities")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from layers import logged_dense, conv2d_logged, conv2d_transpose_logged, reshape_layer, image_condition_concat
from layers import concatenate_with, residual_block, unet, unet_original, inception_block
from networks import Generator, Discriminator, Encoder
from generativeModels import Image2ImageGenerativeModel
from functionsOnImages import padding_zeros
import Preprocessing.initialization as init
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, crop_images

class BiCycleGAN(Image2ImageGenerativeModel):
    def __init__(self, x_dim, y_dim, z_dim, enc_architecture, gen_architecture, adversarial_architecture, folder="./BiCycleGAN",
                 is_patchgan=False, is_wasserstein=False):
        super(BiCycleGAN, self).__init__(x_dim, y_dim, [enc_architecture, gen_architecture, adversarial_architecture], folder)

        self._z_dim = z_dim
        with tf.name_scope("Inputs"):
            self._Z_input = tf.placeholder(tf.float32, shape=[None, z_dim], name="z")

        self._enc_architecture = self._architectures[0]
        self._gen_architecture = self._architectures[1]
        self._adv_architecture = self._architectures[2]

        self._is_patchgan = is_patchgan
        self._is_wasserstein = is_wasserstein
        self._is_feature_matching = False

        ################# Define architecture
        if self._is_patchgan:
            f_xy = self._adv_architecture[-1][-1]["filters"]
            assert f_xy == 1, "If is PatchGAN, last layer of adversarial_XY needs 1 filter. Given: {}.".format(f_xy)

            a_xy = self._adv_architecture[-1][-1]["activation"]
            if self._is_wasserstein:
                assert a_xy == tf.identity, "If is PatchGAN, last layer of adversarial needs tf.identity. Given: {}.".format(a_xy)
            else:
                assert a_xy == tf.nn.sigmoid, "If is PatchGAN, last layer of adversarial needs tf.nn.sigmoid. Given: {}.".format(a_xy)
        else:
            self._adv_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            if self._is_wasserstein:
                self._adv_architecture.append([logged_dense, {"units": 1, "activation": tf.identity, "name": "Output"}])
            else:
                self._adv_architecture.append([logged_dense, {"units": 1, "activation": tf.nn.sigmoid, "name": "Output"}])

        last_layers_mean = [
            [tf.layers.flatten, {"name": "flatten"}],
            [logged_dense, {"units": z_dim, "activation": tf.identity, "name": "Mean"}]
        ]
        self._encoder_mean = Encoder(self._enc_architecture + last_layers_mean, name="Encoder")
        last_layers_std = [
            [tf.layers.flatten, {"name": "flatten"}],
            [logged_dense, {"units": z_dim, "activation": tf.identity, "name": "Std"}]
        ]
        self._encoder_std = Encoder(self._enc_architecture + last_layers_std, name="Encoder")

        self._gen_architecture[-1][1]["name"] = "Output"
        self._generator = Generator(self._gen_architecture, name="Generator")

        self._adversarial = Discriminator(self._adv_architecture, name="Adversarial")

        self._nets = [self._encoder_mean, self._generator, self._adversarial]

        ################# Connect inputs and networks
        self._mean_layer = self._encoder_mean.generate_net(self._Y_input)
        self._std_layer = self._encoder_std.generate_net(self._Y_input)

        self._output_enc_with_noise = self._mean_layer + tf.exp(0.5*self._std_layer)*self._Z_input
        self._gen_input = image_condition_concat(inputs=self._X_input, condition=self._output_enc_with_noise, name="real")
        self._gen_input_from_encoding = image_condition_concat(inputs=self._X_input, condition=self._Z_input, name="real")
        self._output_gen = self._generator.generate_net(self._gen_input)
        self._output_gen_from_encoding = self._generator.generate_net(self._gen_input_from_encoding)

        assert self._output_gen.get_shape()[1:] == y_dim, (
            "Generator output must have shape of y_dim. Given: {}. Expected: {}.".format(self._output_gen.get_shape(), x_dim)
        )

        with tf.name_scope("InputsAdversarial"):
            self._input_real = tf.concat(values=[self._Y_input, self._X_input], axis=3)
            self._input_fake_from_real = tf.concat(values=[self._output_gen, self._X_input], axis=3)
            self._input_fake_from_latent = tf.concat(values=[self._output_gen_from_encoding, self._X_input], axis=3)

        self._output_adv_real = self._adversarial.generate_net(self._input_real)
        self._output_adv_fake_from_real = self._adversarial.generate_net(self._input_fake_from_real)
        self._output_adv_fake_from_latent = self._adversarial.generate_net(self._input_fake_from_latent)

        ################# Finalize
        self._init_folders()
        self._verify_init()

        if self._is_patchgan:
            print("PATCHGAN chosen with output: {}.".format(self._output_adv_real.shape))


    def compile(self, loss, optimizer, learning_rate=None, learning_rate_enc=None, learning_rate_gen=None, learning_rate_adv=None,
                label_smoothing=1, lmbda=1, feature_matching=False):

        if self._is_wasserstein and loss != "wasserstein":
            raise ValueError("If is_wasserstein is true in Constructor, loss needs to be wasserstein.")
        if not self._is_wasserstein and loss == "wasserstein":
            raise ValueError("If loss is wasserstein, is_wasserstein needs to be true in constructor.")

        if np.all([lr is None for lr in [learning_rate, learning_rate_enc, learning_rate_gen, learning_rate_adv]]):
            raise ValueError("Need learning_rate.")
        if learning_rate is not None and learning_rate_enc is None:
            learning_rate_enc = learning_rate
        if learning_rate is not None and learning_rate_gen is None:
            learning_rate_gen = learning_rate
        if learning_rate is not None and learning_rate_adv is None:
            learning_rate_adv = learning_rate

        self._define_loss(loss=loss, label_smoothing=label_smoothing, lmbda=lmbda, feature_matching=feature_matching)
        with tf.name_scope("Optimizer"):
            enc_optimizer = optimizer(learning_rate=learning_rate_enc)
            self._enc_optimizer = enc_optimizer.minimize(self._enc_loss, var_list=self._get_vars("Encoder"), name="Encoder")
            gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars("Generator"), name="Generator")
            adv_optimizer = optimizer(learning_rate=learning_rate_adv)
            self._adv_optimizer = adv_optimizer.minimize(self._adv_loss, var_list=self._get_vars("Adversarial"), name="Adversarial")

            self._enc_grads_and_vars = enc_optimizer.compute_gradients(self._enc_loss, var_list=self._get_vars("Encoder"))
            self._gen_grads_and_vars = gen_optimizer.compute_gradients(self._gen_loss, var_list=self._get_vars("Generator"))
            self._adv_grads_and_vars = adv_optimizer.compute_gradients(self._adv_loss, var_list=self._get_vars("Adversarial"))

        self._summarise()


    def _define_loss(self, loss, label_smoothing, lmbda, feature_matching):
        def get_labels_one(tensor):
            return tf.ones_like(tensor)*label_smoothing
        def get_labels_zero(tensor):
            return tf.zeros_like(tensor) + 1 - label_smoothing
        eps = 1e-7
        ## Kullback-Leibler divergence
        self._KLdiv = 0.5*(tf.square(self._mean_layer) + tf.exp(self._std_layer) - self._std_layer - 1)
        self._KLdiv = tf.reduce_mean(self._KLdiv)

        ## L1 loss
        self._recon_loss = lmbda*tf.reduce_mean(tf.abs(self._Y_input - self._output_gen))

        ## Adversarial loss
        if loss == "cross-entropy":
            self._logits_real = tf.math.log( self._output_adv_real / (1+eps - self._output_adv_real) + eps)
            self._logits_fake_from_real = tf.math.log( self._output_adv_fake_from_real / (1+eps - self._output_adv_fake_from_real) + eps)
            self._logits_fake_from_latent = tf.math.log( self._output_adv_fake_from_latent / (1+eps - self._output_adv_fake_from_latent) + eps)
            self._generator_loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.ones_like(self._logits_fake_from_real), logits=self._logits_fake_from_real
                                    )  +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.ones_like(self._logits_fake_from_latent), logits=self._logits_fake_from_latent
                                    )
            )
            self._adversarial_loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_one(self._logits_real), logits=self._logits_real
                                    ) +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_zero(self._logits_fake_from_real), logits=self._logits_fake_from_real
                                    )  +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_zero(self._logits_fake_from_latent), logits=self._logits_fake_from_latent
                                    )
            )

        elif loss == "L1":
            self._generator_loss = tf.reduce_mean(
                tf.abs(self._output_adv_fake_from_real - tf.ones_like(self._output_adv_fake_from_real)) +
                tf.abs(self._output_adv_fake_from_latent - tf.ones_like(self._output_adv_fake_from_latent))
            ) / 2
            self._adversarial_loss = (
                tf.reduce_mean(
                    tf.abs(self._output_adv_real - get_labels_one(self._output_adv_real)) +
                    tf.abs(self._output_adv_fake_from_real - get_labels_zero(self._output_adv_fake_from_real)) +
                    tf.abs(self._output_adv_fake_from_latent - get_labels_zero(self._output_adv_fake_from_real))
                )
            ) / 3.0
        elif loss == "L2":
            self._generator_loss = tf.reduce_mean(
                tf.square(self._output_adv_fake_from_real - tf.ones_like(self._output_adv_fake_from_real)) +
                tf.square(self._output_adv_fake_from_latent - tf.ones_like(self._output_adv_fake_from_latent))
            ) / 2
            self._adversarial_loss = (
                tf.reduce_mean(
                    tf.square(self._output_adv_real - get_labels_one(self._output_adv_real)) +
                    tf.square(self._output_adv_fake_from_real - get_labels_zero(self._output_adv_fake_from_real)) +
                    tf.square(self._output_adv_fake_from_latent - get_labels_zero(self._output_adv_fake_from_real))
                )
            ) / 3.0
        elif loss == "wasserstein":
            self._generator_loss = -tf.reduce_mean(self._output_adv_fake_from_real) - tf.reduce_mean(self._output_adv_fake_from_latent)
            self._adversarial_loss = (
                -(tf.reduce_mean(self._output_adv_real) -
                    tf.reduce_mean(self._output_adv_fake_from_real) -
                    tf.reduce_mean(self._output_adv_fake_from_latent)) +
                    10*self._define_gradient_penalty()
            )
        elif loss == "KL":
            self._logits_real = tf.math.log( self._output_adv_real / (1+eps - self._output_adv_real) + eps)
            self._logits_fake_from_real = tf.math.log( self._output_adv_fake_from_real / (1+eps - self._output_adv_fake_from_real) + eps)
            self._logits_fake_from_latent = tf.math.log( self._output_adv_fake_from_latent / (1+eps - self._output_adv_fake_from_latent) + eps)
            self._generator_loss = (-tf.reduce_mean(self._logits_fake_from_real) - tf.reduce_mean(self._logits_fake_from_latent))/2
            self._adversarial_loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_one(self._logits_real), logits=self._logits_real
                                    ) +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_zero(self._logits_fake_from_real), logits=self._logits_fake_from_real
                                    )  +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_zero(self._logits_fake_from_latent), logits=self._logits_fake_from_latent
                                    )
            )
        else:
            raise ValueError("Loss not implemented. Choose from {}. Given: {}.".format(possible_losses, loss))


        if feature_matching:
            self._is_feature_matching = True
            otp_adv_real = self._adversarial.generate_net(self._input_real, tf_trainflag=self._is_training, return_idx=-2)
            otp_adv_fake = self._adversarial.generate_net(self._input_fake_from_real, tf_trainflag=self._is_training, return_idx=-2)
            self._generator_loss = tf.reduce_mean(tf.square(otp_adv_real - otp_adv_fake))

        with tf.name_scope("Loss") as scope:

            self._enc_loss = self._KLdiv + self._recon_loss + self._generator_loss
            self._gen_loss = self._recon_loss + self._generator_loss
            self._adv_loss = self._adversarial_loss

            tf.summary.scalar("Kullback-Leibler", self._KLdiv)
            tf.summary.scalar("Reconstruction", self._recon_loss)
            tf.summary.scalar("Vanilla_Generator", self._generator_loss)

            tf.summary.scalar("Encoder", self._enc_loss)
            tf.summary.scalar("Generator", self._gen_loss)
            tf.summary.scalar("Adversarial", self._adv_loss)


    def _define_gradient_penalty(self):
        alpha = tf.random_uniform(shape=tf.shape(self._input_real), minval=0., maxval=1.)
        differences = self._input_fake_from_real - self._input_real
        interpolates = self._input_real + (alpha * differences)
        gradients = tf.gradients(self._adversarial.generate_net(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        with tf.name_scope("Loss") as scope:
            self._gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            tf.summary.scalar("Gradient_penalty", self._gradient_penalty)
        return self._gradient_penalty


    def train(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=64, adv_steps=5, gen_steps=1,
              log_step=3, gpu_options=None, batch_log_step=None):
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        self._set_up_test_train_sample(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        self._z_test = self._generator.sample_noise(n=len(self._x_test))
        nr_batches = np.floor(len(x_train) / batch_size)
        self._total_train_time = 0
        self._total_log_time = 0
        count_batches = 0

        self._batches = []
        self._monitor_dict = {
            "Gradients": [
                [self._enc_grads_and_vars, self._gen_grads_and_vars, self._adv_grads_and_vars],
                ["Encoder", "Generator", "Adversarial"],
                [[] for i in range(9)]
            ],
            "Losses": [
                    [self._enc_loss, self._gen_loss, self._adversarial_loss, self._generator_loss, self._recon_loss, self._KLdiv],
                    ["Encoder (V+R+K)", "Generator (V+R)", "Adversarial", "Vanilla_Generator", "Reconstruction", "Kullback-Leibler"],
                    [[] for i in range(6)]
            ],
            "Output Adversarial": [
                    [self._output_adv_fake_from_real, self._output_adv_fake_from_latent, self._output_adv_real],
                    ["Fake_from_real", "Fake_from_latent", "Real"],
                    [[] for i in range(3)],
                    [np.mean]
            ]
        }

        for epoch in range(epochs):
            batch_nr = 0
            adv_loss_epoch = 0
            gen_loss_epoch = 0
            enc_loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            ii = 0

            while trained_examples < len(x_train):
                batch_train_start = time.clock()
                adv_loss_batch, gen_loss_batch, enc_loss_batch = self._optimize(self._trainset, batch_size, adv_steps, gen_steps)
                trained_examples += batch_size
                adv_loss_epoch += adv_loss_batch
                gen_loss_epoch += gen_loss_batch
                enc_loss_epoch += enc_loss_batch
                self._total_train_time += (time.clock() - batch_train_start)

                if (batch_log_step is not None) and (ii % batch_log_step == 0):
                    count_batches += batch_log_step
                    batch_train_time = (time.clock() - start)/60
                    self._log(count_batches, batch_train_time)
                ii += 1

            epoch_train_time = (time.clock() - start)/60
            adv_loss_epoch = np.round(adv_loss_epoch, 2)
            gen_loss_epoch = np.round(gen_loss_epoch, 2)
            enc_loss_epoch = np.round(enc_loss_epoch, 2)

            print("\nEpoch {}: D: {}; G: {}; E: {}.".format(epoch, adv_loss_epoch, gen_loss_epoch, enc_loss_epoch))

            if log_step is not None and batch_log_step is None:
                self._log(epoch+1, epoch_train_time)


    def _optimize(self, dataset, batch_size, adv_steps, gen_steps):
        for i in range(adv_steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            Z_noise = self._generator.sample_noise(n=len(current_batch_x))
            _, adv_loss_batch = self._sess.run([self._adv_optimizer, self._adv_loss], feed_dict={
                self._X_input: current_batch_x, self._Y_input: current_batch_y, self._Z_input: Z_noise, self._is_training: True
            })

        for i in range(gen_steps):
            Z_noise = self._generator.sample_noise(n=len(current_batch_x))
            _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss], feed_dict={
                self._X_input: current_batch_x, self._Y_input: current_batch_y, self._Z_input: Z_noise, self._is_training: True
            })
            _, enc_loss_batch = self._sess.run([self._enc_optimizer, self._enc_loss], feed_dict={
                self._X_input: current_batch_x, self._Y_input: current_batch_y, self._Z_input: Z_noise, self._is_training: True
            })

        return adv_loss_batch, gen_loss_batch, enc_loss_batch


    def _log_results(self, epoch, epoch_time):
        summary = self._sess.run(self._merged_summaries, feed_dict={self._X_input: self._x_test, self._Y_input: self._y_test,
                                 self._Z_input: self._z_test,
                                 self._epoch_time: epoch_time, self._is_training: False, self._epoch_nr: epoch})
        self._writer1.add_summary(summary, epoch)
        nr_test = len(self._x_test)
        summary = self._sess.run(self._merged_summaries, feed_dict={self._X_input: self._trainset.get_xdata()[:nr_test],
                                 self._Z_input: self._z_test,
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
        outpt_xy = sess.run(self._output_gen_from_encoding, feed_dict={self._X_input: inpt_x, self._Z_input: self._z_test[:len(inpt_x)],
                            self._is_training: False})

        image_matrix = np.array([
            [
                x.reshape(self._x_dim[0], self._x_dim[1]), y.reshape(self._y_dim[0], self._y_dim[1]),
                np.zeros(shape=(self._x_dim[0], self._x_dim[1])),
                xy.reshape(self._y_dim[0], self._y_dim[1])
            ]
                for x, y, xy in zip(inpt_x, inpt_y, outpt_xy)
        ])
        self._generator.build_generated_samples(image_matrix, column_titles=["True X", "True Y", "", "Gen_XY"], epoch=epoch, path=path)


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
                            self._X_input: self._x_test, self._Y_input: self._y_test,
                            self._Z_input: self._z_test, self._is_training: False
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
        Z_noise = self._generator.sample_noise(n=len(current_batch_x))

        print("Batch ", epoch)
        colors = ["green", "blue", "red", "orange", "purple", "brown"]
        for k, key in enumerate(self._monitor_dict):
            if key == "Gradients":
                continue
            key_results = self._sess.run(
                self._monitor_dict[key][0],
                feed_dict={
                    self._X_input: current_batch_x, self._Y_input: current_batch_y,
                    self._Z_input: Z_noise, self._is_training: True
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

        gen_samples = self._sess.run([self._output_gen_from_encoding],
            feed_dict={
                self._X_input: current_batch_x, self._Z_input: Z_noise, self._is_training: False
        })
        axs[-1].hist([np.ravel(gen_samples), np.ravel(current_batch_y)], label=["Generated", "True"])
        axs[-1].set_title("Pixel distribution")
        axs[-1].legend()

        self._total_log_time += (time.clock() - log_start)
        fig.suptitle("Train {} / Log {}".format(np.round(self._total_train_time, 2), np.round(self._total_log_time, 2)))
        plt.savefig(self._folder+"/TrainStatistics.png")
        plt.close("all")


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
        x_train = pickle.load(f)[0].reshape(-1, 28, 28, 1) / 255
        x_test = x_train[-100:]
        x_train = x_train[:-100]
    with open("{}/train_images_rotated.pickle".format(data_path), "rb") as f:
        y_train = pickle.load(f).reshape(-1, 28, 28, 1) / 255
        y_test = y_train[-100:]
        y_train = y_train[:-100]

    x_train = 1-x_train
    x_test = 1-x_test
    y_train = 1-y_train
    y_test = 1-y_test


    ########### Architecture

    x_train = padding_zeros(x_train, top=2, bottom=2, left=2, right=2)
    x_test = padding_zeros(x_test, top=2, bottom=2, left=2, right=2)
    y_train = padding_zeros(y_train, top=2, bottom=2, left=2, right=2)
    y_test = padding_zeros(y_test, top=2, bottom=2, left=2, right=2)
    enc_architecture = [
                        [conv2d_logged, {"filters": 8, "kernel_size": 5, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 16, "kernel_size": 5, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 32, "kernel_size": 5, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [tf.layers.flatten, {}],
                        [tf.layers.dense, {"units": 64, "activation": tf.nn.leaky_relu}],
                        [tf.layers.dense, {"units": 32, "activation": tf.nn.leaky_relu}],
                        ]
    gen_architecture = [
                        [conv2d_logged, {"filters": 32, "kernel_size": 5, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 64, "kernel_size": 5, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_transpose_logged, {"filters": 64, "kernel_size": 5, "strides": 2,
                                                    "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_transpose_logged, {"filters": 32, "kernel_size": 5, "strides": 2,
                                                    "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 32, "kernel_size": 5, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 16, "kernel_size": 5, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 1, "kernel_size": 5, "activation": tf.nn.sigmoid, "padding": "same"}]
                        ]
    adversarial_architecture = [
                        [conv2d_logged, {"filters": 4, "kernel_size": 5, "strides": 4, "activation": tf.nn.tanh, "padding": "same"}],
                        [conv2d_logged, {"filters": 8, "kernel_size": 5, "strides": 4, "activation": tf.nn.tanh, "padding": "same"}],
                        [conv2d_logged, {"filters": 16, "kernel_size": 5, "strides": 2, "activation": tf.nn.tanh, "padding": "same"}],
                        ]


    learning_rate = 0.00005
    learning_rate_adv = 0.000005
    batch_size = 16
    epochs = 30
    gen_steps = 1
    adv_steps = 1
    z_dim = 32
    lmbda = 10
    optimizer = tf.train.AdamOptimizer
    batch_log_step = 100 #int(len(x_train) / batch_size / 10)
    feature_matching = True
    loss = "cross-entropy"
    is_wasserstein = loss == "wasserstein"
    is_patchGAN = False
    label_smoothing = 0.95

    if is_patchGAN and is_wasserstein:
        adversarial_architecture[-1][1]["activation"] = tf.identity
    elif is_patchGAN:
        adversarial_architecture[-1][1]["activation"] = tf.nn.sigmoid


    inpt_dim = x_train[0].shape
    opt_dim = y_train[0].shape

    config_data = init.create_config_file(globals())
    save_folder = "../../../Results/Test"
    save_folder = init.initialize_folder(algorithm="BiCycleGAN_", base_folder=save_folder)

    bicyclegan = BiCycleGAN(x_dim=inpt_dim, y_dim=opt_dim, z_dim=z_dim, enc_architecture=enc_architecture,
                      gen_architecture=gen_architecture, adversarial_architecture=adversarial_architecture, folder=save_folder,
                      is_patchgan=is_patchGAN, is_wasserstein=is_wasserstein)
    print(bicyclegan._generator.get_number_params())
    print(bicyclegan._adversarial.get_number_params())
    print(bicyclegan._encoder_mean.get_number_params())
    nr_params = bicyclegan.get_number_params()
    print(bicyclegan.show_architecture())
    with open(save_folder+"/config.json", "w") as f:
        json.dump(config_data, f, indent=4)
    bicyclegan.log_architecture()
    bicyclegan.compile(loss=loss, learning_rate=learning_rate, learning_rate_adv=learning_rate_adv, optimizer=tf.train.AdamOptimizer,
                    lmbda=lmbda, feature_matching=feature_matching, label_smoothing=label_smoothing)
    bicyclegan.train(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size,
                   adv_steps=adv_steps, gen_steps=gen_steps, log_step=1, batch_log_step=batch_log_step, gpu_options=gpu_options)


