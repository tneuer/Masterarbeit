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

from layers import logged_dense, conv2d_logged, conv2d_transpose_logged, reshape_layer
from layers import concatenate_with, residual_block, unet, unet_original, inception_block
from networks import Generator, Discriminator, Encoder
from generativeModels import GenerativeModel
from functionsOnImages import padding_zeros
import Preprocessing.initialization as init
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, crop_images

class VAEGAN(GenerativeModel):
    def __init__(self, x_dim, z_dim, enc_architecture, gen_architecture, disc_architecture, folder="./VAEGAN"):
        super(VAEGAN, self).__init__(x_dim, z_dim, [enc_architecture, gen_architecture, disc_architecture], folder)

        self._enc_architecture = self._architectures[0]
        self._gen_architecture = self._architectures[1]
        self._disc_architecture = self._architectures[2]

        ################# Define architecture
        last_layer_mean = [logged_dense, {"units": z_dim, "activation": tf.identity, "name": "Mean"}]
        self._encoder_mean = Encoder(self._enc_architecture + [last_layer_mean], name="Encoder")
        last_layer_std = [logged_dense, {"units": z_dim, "activation": tf.identity, "name": "Std"}]
        self._encoder_std = Encoder(self._enc_architecture + [last_layer_std], name="Encoder")

        self._gen_architecture[-1][1]["name"] = "Output"
        self._generator = Generator(self._gen_architecture, name="Generator")

        self._disc_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
        self._disc_architecture.append([logged_dense, {"units": 1, "activation": tf.nn.sigmoid, "name": "Output"}])
        self._discriminator = Discriminator(self._disc_architecture, name="Discriminator")

        self._nets = [self._encoder_mean, self._generator, self._discriminator]

        ################# Connect inputs and networks
        self._mean_layer = self._encoder_mean.generate_net(self._X_input)
        self._std_layer = self._encoder_std.generate_net(self._X_input)

        self._output_enc_with_noise = self._mean_layer + tf.exp(0.5*self._std_layer)*self._Z_input

        self._output_gen = self._generator.generate_net(self._output_enc_with_noise)
        self._output_gen_from_encoding = self._generator.generate_net(self._Z_input)

        assert self._output_gen.get_shape()[1:] == x_dim, (
            "Generator output must have shape of x_dim. Given: {}. Expected: {}.".format(self._output_gen.get_shape(), x_dim)
        )

        self._output_disc_real = self._discriminator.generate_net(self._X_input)
        self._output_disc_fake_from_real = self._discriminator.generate_net(self._output_gen)
        self._output_disc_fake_from_latent = self._discriminator.generate_net(self._output_gen_from_encoding)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, learning_rate=0.0001, optimizer=tf.train.AdamOptimizer, label_smoothing=1, gamma=1):
        self._define_loss(label_smoothing=label_smoothing, gamma=gamma)
        with tf.name_scope("Optimizer"):
            enc_optimizer = optimizer(learning_rate=learning_rate)
            self._enc_optimizer = enc_optimizer.minimize(self._enc_loss, var_list=self._get_vars("Encoder"), name="Encoder")
            gen_optimizer = optimizer(learning_rate=learning_rate)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars("Generator"), name="Generator")
            disc_optimizer = optimizer(learning_rate=learning_rate)
            self._disc_optimizer = disc_optimizer.minimize(self._disc_loss, var_list=self._get_vars("Discriminator"), name="Discriminator")
        self._summarise()


    def _define_loss(self, label_smoothing, gamma):
        def get_labels_one(tensor):
            return tf.ones_like(tensor)*label_smoothing
        eps = 1e-7
        ## Kullback-Leibler divergence
        self._KLdiv = 0.5*(tf.square(self._mean_layer) + tf.exp(self._std_layer) - self._std_layer - 1)
        self._KLdiv = tf.reduce_mean(self._KLdiv)

        ## Feature matching loss
        otp_disc_real = self._discriminator.generate_net(self._X_input, tf_trainflag=self._is_training, return_idx=-2)
        otp_disc_fake = self._discriminator.generate_net(self._output_gen, tf_trainflag=self._is_training, return_idx=-2)
        self._feature_loss = tf.reduce_mean(tf.square(otp_disc_real - otp_disc_fake))

        ## Discriminator loss
        self._logits_real = tf.math.log( self._output_disc_real / (1+eps - self._output_disc_real) + eps)
        self._logits_fake_from_real = tf.math.log( self._output_disc_fake_from_real / (1+eps - self._output_disc_fake_from_real) + eps)
        self._logits_fake_from_latent = tf.math.log( self._output_disc_fake_from_latent / (1+eps - self._output_disc_fake_from_latent) + eps)
        self._generator_loss = tf.reduce_mean(
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=get_labels_one(self._logits_fake_from_real), logits=self._logits_fake_from_real
                                )  +
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=get_labels_one(self._logits_fake_from_latent), logits=self._logits_fake_from_latent
                                )
        )
        self._discriminator_loss = tf.reduce_mean(
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=get_labels_one(self._logits_real), logits=self._logits_real
                                ) +
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=tf.zeros_like(self._logits_fake_from_real), logits=self._logits_fake_from_real
                                )  +
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=tf.zeros_like(self._logits_fake_from_latent), logits=self._logits_fake_from_latent
                                )
        )

        with tf.name_scope("Loss") as scope:

            self._enc_loss = self._KLdiv + self._feature_loss
            self._gen_loss = self._feature_loss + self._generator_loss
            self._disc_loss = self._discriminator_loss

            tf.summary.scalar("Encoder", self._enc_loss)
            tf.summary.scalar("Generator", self._gen_loss)
            tf.summary.scalar("Discriminator", self._disc_loss)


    def train(self, x_train, x_test, epochs=100, batch_size=64, disc_steps=5, gen_steps=1, log_step=3):
        self._set_up_training(log_step=log_step)
        self._set_up_test_train_sample(x_train, x_test)
        for epoch in range(epochs):
            batch_nr = 0
            disc_loss_epoch = 0
            gen_loss_epoch = 0
            enc_loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            while trained_examples < len(x_train):
                disc_loss_batch, gen_loss_batch, enc_loss_batch = self._optimize(self._trainset, batch_size, disc_steps, gen_steps)
                trained_examples += batch_size
                disc_loss_epoch += disc_loss_batch
                gen_loss_epoch += gen_loss_batch
                enc_loss_epoch += enc_loss_batch

            epoch_train_time = (time.clock() - start)/60
            disc_loss_epoch = np.round(disc_loss_epoch, 2)
            gen_loss_epoch = np.round(gen_loss_epoch, 2)
            enc_loss_epoch = np.round(enc_loss_epoch, 2)

            print("Epoch {}: D: {}; G: {}; E: {}.".format(epoch, disc_loss_epoch, gen_loss_epoch, enc_loss_epoch))

            if log_step is not None:
                self._log(epoch, epoch_train_time)


    def _optimize(self, dataset, batch_size, disc_steps, gen_steps):
        for i in range(disc_steps):
            current_batch_x = dataset.get_next_batch(batch_size)
            Z_noise = self._generator.sample_noise(n=len(current_batch_x))
            _, disc_loss_batch = self._sess.run([self._disc_optimizer, self._disc_loss], feed_dict={self._X_input: current_batch_x, self._Z_input: Z_noise})

        for i in range(gen_steps):
            Z_noise = self._generator.sample_noise(n=len(current_batch_x))
            _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss], feed_dict={self._X_input: current_batch_x, self._Z_input: Z_noise})
            _, enc_loss_batch = self._sess.run([self._enc_optimizer, self._enc_loss], feed_dict={self._X_input: current_batch_x, self._Z_input: Z_noise})

        return disc_loss_batch, gen_loss_batch, enc_loss_batch


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train/255.
    x_test = x_test/255.
    x_train = x_train.reshape([len(x_train), 784])[:20]
    x_test = x_test.reshape([len(x_test), 784])[:20]

    enc_architecture = [
                        [logged_dense, {"units": 256, "activation": tf.nn.relu}],
                        ]
    gen_architecture = [
                        [logged_dense, {"units": 256, "activation": tf.nn.relu}],
                        [logged_dense, {"units": 784, "activation": tf.nn.relu}],
                        ]
    disc_architecture = [
                        [logged_dense, {"units": 256, "activation": tf.nn.relu}],
                        ]
    inpt_dim = 784
    z_dim = 64
    label_smoothing = 0.9
    gamma = 0.9

    vaegan = VAEGAN(x_dim=inpt_dim, z_dim=z_dim, enc_architecture=enc_architecture,
                    gen_architecture=gen_architecture, disc_architecture=disc_architecture,
                  folder="../../../Results/Test/VAEGAN")
    vaegan.log_architecture()
    print(vaegan.show_architecture())
    vaegan.compile(label_smoothing=label_smoothing, gamma=gamma)
    vaegan.train(x_train, x_test, epochs=100)

