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

class BiCylceGAN(Image2ImageGenerativeModel):
    def __init__(self, x_dim, y_dim, z_dim, enc_architecture, gen_architecture, disc_architecture, folder="./BiCylceGAN"):
        super(BiCylceGAN, self).__init__(x_dim, y_dim, [enc_architecture, gen_architecture, disc_architecture], folder)

        self._z_dim = z_dim
        with tf.name_scope("Inputs"):
            self._Z_input = tf.placeholder(tf.float32, shape=[None, z_dim], name="z")

        self._enc_architecture = self._architectures[0]
        self._gen_architecture = self._architectures[1]
        self._disc_architecture = self._architectures[2]

        ################# Define architecture
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

        self._disc_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
        self._disc_architecture.append([logged_dense, {"units": 1, "activation": tf.nn.sigmoid, "name": "Output"}])
        self._discriminator = Discriminator(self._disc_architecture, name="Discriminator")

        self._nets = [self._encoder_mean, self._generator, self._discriminator]

        ################# Connect inputs and networks
        self._mean_layer = self._encoder_mean.generate_net(self._Y_input)
        self._std_layer = self._encoder_std.generate_net(self._Y_input)

        self._output_enc_with_noise = self._mean_layer + tf.exp(0.5*self._std_layer)*self._Z_input
        self._gen_input = image_condition_concat(inputs=self._X_input, condition=self._output_enc_with_noise, name="real")
        self._gen_input_from_encoding = image_condition_concat(inputs=self._X_input, condition=self._Z_input, name="real")
        self._output_gen = self._generator.generate_net(self._gen_input)
        self._output_gen_from_encoding = self._generator.generate_net(self._gen_input_from_encoding)

        assert self._output_gen.get_shape()[1:] == x_dim, (
            "Generator output must have shape of x_dim. Given: {}. Expected: {}.".format(self._output_gen.get_shape(), x_dim)
        )

        self._output_disc_real = self._discriminator.generate_net(self._X_input)
        self._output_disc_fake = self._discriminator.generate_net(self._output_gen)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, learning_rate=0.0001, optimizer=tf.train.AdamOptimizer, label_smoothing=1, lmbda=1):
        self._define_loss(label_smoothing=label_smoothing, lmbda=lmbda)
        with tf.name_scope("Optimizer"):
            enc_optimizer = optimizer(learning_rate=learning_rate)
            self._enc_optimizer = enc_optimizer.minimize(self._enc_loss, var_list=self._get_vars("Encoder"), name="Encoder")
            gen_optimizer = optimizer(learning_rate=learning_rate)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars("Generator"), name="Generator")
            disc_optimizer = optimizer(learning_rate=learning_rate)
            self._disc_optimizer = disc_optimizer.minimize(self._disc_loss, var_list=self._get_vars("Discriminator"), name="Discriminator")
        self._summarise()


    def _define_loss(self, label_smoothing, lmbda):
        def get_labels_one(tensor):
            return tf.ones_like(tensor)*label_smoothing
        eps = 1e-7
        ## Kullback-Leibler divergence
        self._KLdiv = 0.5*(tf.square(self._mean_layer) + tf.exp(self._std_layer) - self._std_layer - 1)
        self._KLdiv = tf.reduce_mean(self._KLdiv)

        ## L1 loss
        self._recon_loss = lmbda*tf.reduce_mean(tf.abs(self._Y_input - self._output_gen))

        ## Discriminator loss
        self._logits_real = tf.math.log( self._output_disc_real / (1+eps - self._output_disc_real) + eps)
        self._logits_fake = tf.math.log( self._output_disc_fake / (1+eps - self._output_disc_fake) + eps)
        self._generator_loss = tf.reduce_mean(
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=get_labels_one(self._logits_fake), logits=self._logits_fake
                                )
        )
        self._discriminator_loss = tf.reduce_mean(
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=get_labels_one(self._logits_real), logits=self._logits_real
                                ) +
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=tf.zeros_like(self._logits_fake), logits=self._logits_fake
                                )
        )

        with tf.name_scope("Loss") as scope:

            self._enc_loss = self._KLdiv + self._recon_loss + self._generator_loss
            self._gen_loss = self._recon_loss + self._generator_loss
            self._disc_loss = self._discriminator_loss

            tf.summary.scalar("Encoder", self._enc_loss)
            tf.summary.scalar("Generator", self._gen_loss)
            tf.summary.scalar("Discriminator", self._disc_loss)


    def train(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=64, disc_steps=5, gen_steps=1,
              log_step=3, gpu_options=None, batch_log_step=None):
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        self._set_up_test_train_sample(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        self._z_test = self._generator.sample_noise(n=len(self._x_test))
        nr_batches = np.floor(len(x_train) / batch_size)

        for epoch in range(epochs):
            batch_nr = 0
            disc_loss_epoch = 0
            gen_loss_epoch = 0
            enc_loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            ii = 0

            while trained_examples < len(x_train):
                disc_loss_batch, gen_loss_batch, enc_loss_batch = self._optimize(self._trainset, batch_size, disc_steps, gen_steps)
                trained_examples += batch_size
                disc_loss_epoch += disc_loss_batch
                gen_loss_epoch += gen_loss_batch
                enc_loss_epoch += enc_loss_batch

                if (batch_log_step is not None) and (ii % batch_log_step == 0):
                    batch_train_time = (time.clock() - start)/60
                    self._log(int(epoch*nr_batches+ii), batch_train_time)
                    print("Batch {}: D: {}; G: {}; E: {}.".format(ii, disc_loss_batch, gen_loss_batch, enc_loss_batch))

                ii += 1

            epoch_train_time = (time.clock() - start)/60
            disc_loss_epoch = np.round(disc_loss_epoch, 2)
            gen_loss_epoch = np.round(gen_loss_epoch, 2)
            enc_loss_epoch = np.round(enc_loss_epoch, 2)

            print("Epoch {}: D: {}; G: {}; E: {}.".format(epoch, disc_loss_epoch, gen_loss_epoch, enc_loss_epoch))

            if log_step is not None:
                self._log(epoch, epoch_train_time)


    def _optimize(self, dataset, batch_size, disc_steps, gen_steps):
        for i in range(disc_steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            Z_noise = self._generator.sample_noise(n=len(current_batch_x))
            _, disc_loss_batch = self._sess.run([self._disc_optimizer, self._disc_loss], feed_dict={
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

        return disc_loss_batch, gen_loss_batch, enc_loss_batch


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
        data_path = "../../../Data/mnist"
    else:
        data_path = "/home/tneuer/Backup/Algorithmen/0TestData/image_to_image/mnist"
    with open("{}/train_images.pickle".format(data_path), "rb") as f:
        x_train = pickle.load(f)[0].reshape(-1, 28, 28, 1) / 255
        x_test = x_train[-100:]
        x_train = x_train[:-100]
    with open("{}/train_images_rotated.pickle".format(data_path), "rb") as f:
        y_train = pickle.load(f).reshape(-1, 28, 28, 1) / 255
        y_test = y_train[-100:]
        y_train = y_train[:-100]


    ########### Architecture

    x_train = padding_zeros(x_train, top=2, bottom=2, left=2, right=2)
    x_test = padding_zeros(x_test, top=2, bottom=2, left=2, right=2)
    y_train = padding_zeros(y_train, top=2, bottom=2, left=2, right=2)
    y_test = padding_zeros(y_test, top=2, bottom=2, left=2, right=2)
    enc_architecture = [
                        [conv2d_logged, {"filters": 128, "kernel_size": 5, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 64, "kernel_size": 4, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 16, "kernel_size": 4, "strides": 2, "activation": tf.nn.sigmoid, "padding": "same"}],
                        ]
    gen_architecture = [
                        # [unet_original, {"depth": 2, "filters": 32, "activation": tf.nn.leaky_relu}],
                        [conv2d_logged, {"filters": 32, "kernel_size": 4, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 64, "kernel_size": 4, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 128, "kernel_size": 4, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_transpose_logged, {"filters": 64, "kernel_size": 4, "strides": 2,
                                                    "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_transpose_logged, {"filters": 32, "kernel_size": 4, "strides": 2,
                                                    "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_transpose_logged, {"filters": 1, "kernel_size": 4, "strides": 2,
                                                    "activation": tf.nn.sigmoid, "padding": "same"}],
                        ]
    disc_architecture = [
                        [conv2d_logged, {"filters": 16, "kernel_size": 4, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 32, "kernel_size": 4, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 64, "kernel_size": 4, "strides": 2, "activation": tf.nn.sigmoid, "padding": "same"}],
                        ]


    learning_rate = 0.001
    batch_size = 16
    epochs = 30
    gen_steps = 1
    disc_steps = 5
    z_dim = 32
    lmbda = 1
    optimizer = tf.train.AdamOptimizer

    inpt_dim = x_train[0].shape
    opt_dim = y_train[0].shape

    config_data = init.create_config_file(globals())
    save_folder = "../../../Results/Test"
    save_folder = init.initialize_folder(algorithm="BiCylceGAN_", base_folder=save_folder)

    bicyclegan = BiCylceGAN(x_dim=inpt_dim, y_dim=opt_dim, z_dim=z_dim, enc_architecture=enc_architecture,
                      gen_architecture=gen_architecture, disc_architecture=disc_architecture, folder=save_folder)
    print(bicyclegan._generator.get_number_params())
    print(bicyclegan._discriminator.get_number_params())
    print(bicyclegan._encoder_mean.get_number_params())
    nr_params = bicyclegan.get_number_params()
    print(bicyclegan.show_architecture())
    with open(save_folder+"/config.json", "w") as f:
        json.dump(config_data, f, indent=4)
    bicyclegan.log_architecture()
    bicyclegan.compile(learning_rate=learning_rate, optimizer=tf.train.AdamOptimizer, lmbda=lmbda)
    bicyclegan.train(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size,
                   disc_steps=disc_steps, gen_steps=gen_steps, log_step=1, batch_log_step=2000, gpu_options=gpu_options)


