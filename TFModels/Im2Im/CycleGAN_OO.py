#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-06-13 10:26:33
    # Description :
####################################################################################
"""
import os
import sys
import time
import json
sys.path.insert(1, "./building_blocks")
sys.path.insert(1, "../Utilities")
sys.path.insert(1, "../")

import numpy as np
import tensorflow as tf

from building_blocks.layers import logged_dense, conv2d_logged, conv2d_transpose_logged, reshape_layer
from building_blocks.layers import residual_block, unet, unet_original, inception_block
from building_blocks.networks import Generator, Discriminator
from building_blocks.generativeModels import CyclicGenerativeModel
from functionsOnImages import padding_zeros
import Preprocessing.initialization as init


class CycleGAN(CyclicGenerativeModel):
    def __init__(self, x_dim, y_dim, gen_xy_architecture, disc_xy_architecture,
                    gen_yx_architecture, disc_yx_architecture,
                    folder="./CycleGAN", PatchGAN=False, is_wasserstein=False
        ):
        super(CycleGAN, self).__init__(x_dim, y_dim,
                                [gen_xy_architecture, disc_xy_architecture, gen_yx_architecture, disc_yx_architecture],
                                folder, y_dim)

        self._gen_xy_architecture = self._architectures[0]
        self._disc_xy_architecture = self._architectures[1]
        self._gen_yx_architecture = self._architectures[2]
        self._disc_yx_architecture = self._architectures[3]
        self._is_patchgan = PatchGAN
        self._is_wasserstein = is_wasserstein

        ################# Define architecture

        if self._is_patchgan:
            f_xy = self._disc_xy_architecture[-1][-1]["filters"]
            assert f_xy == 1, "If is PatchGAN, last layer of Discriminator_XY needs 1 filter. Given: {}.".format(f_xy)
            f_yx = self._disc_yx_architecture[-1][-1]["filters"]
            assert f_yx == 1, "If is PatchGAN, last layer of Discriminator_YX needs 1 filter. Given: {}.".format(f_yx)

            a_xy = self._disc_xy_architecture[-1][-1]["activation"]
            assert a_xy == tf.nn.sigmoid, "If is PatchGAN, last layer of Discriminator_XY needs tf.nn.sigmoid. Given: {}.".format(a_xy)
            a_yx = self._disc_yx_architecture[-1][-1]["activation"]
            assert a_yx == tf.nn.sigmoid, "If is PatchGAN, last layer of Discriminator_YX needs tf.nn.sigmoid. Given: {}.".format(a_yx)
        else:
            self._disc_xy_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            if self._is_wasserstein:
                self._disc_xy_architecture.append([logged_dense, {"units": 1, "activation": tf.identity, "name": "Output"}])
            else:
                self._disc_xy_architecture.append([logged_dense, {"units": 1, "activation": tf.sigmoid, "name": "Output"}])

            self._disc_yx_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            if self._is_wasserstein:
                self._disc_yx_architecture.append([logged_dense, {"units": 1, "activation": tf.identity, "name": "Output"}])
            else:
                self._disc_yx_architecture.append([logged_dense, {"units": 1, "activation": tf.sigmoid, "name": "Output"}])


        self._gen_xy_architecture[-1][1]["name"] = "Output_XY"
        self._gen_yx_architecture[-1][1]["name"] = "Output_YX"


        self._generator_xy = Generator(self._gen_xy_architecture, name="Generator_XY")
        self._discriminator_xy = Discriminator(self._disc_xy_architecture, name="Discriminator_XY")
        self._generator_yx = Generator(self._gen_yx_architecture, name="Generator_YX")
        self._discriminator_yx = Discriminator(self._disc_yx_architecture, name="Discriminator_YX")

        self._nets = [self._generator_xy, self._discriminator_xy, self._generator_yx, self._discriminator_yx]

        ################# Connect inputs and networks
        self._output_gen_xy = self._generator_xy.generate_net(self._X_input, tf_trainflag=self._is_training)
        self._output_gen_yx = self._generator_yx.generate_net(self._Y_input, tf_trainflag=self._is_training)

        self._output_disc_xy_real = self._discriminator_xy.generate_net(self._Y_input, tf_trainflag=self._is_training)
        self._output_disc_xy_fake = self._discriminator_xy.generate_net(self._output_gen_xy, tf_trainflag=self._is_training)

        self._output_disc_yx_real = self._discriminator_yx.generate_net(self._X_input, tf_trainflag=self._is_training)
        self._output_disc_yx_fake = self._discriminator_yx.generate_net(self._output_gen_yx, tf_trainflag=self._is_training)

        self._output_gen_xyx = self._generator_yx.generate_net(self._output_gen_xy, tf_trainflag=self._is_training)
        self._output_gen_yxy = self._generator_xy.generate_net(self._output_gen_yx, tf_trainflag=self._is_training)

        if self._is_patchgan:
            print("PATCHGAN chosen with output: {}.".format(self._output_disc_xy_real.shape))

        ################# Finalize

        self._init_folders()
        self._verify_init()


    def compile(self, learning_rate=0.00005, learning_rate_gen=None, learning_rate_disc=None,
                    optimizer=tf.train.AdamOptimizer, lmbda=10, loss="cross-entropy"):
        if self._is_wasserstein and loss != "wasserstein":
            raise ValueError("If is_wasserstein is true in Constructor, loss needs to be wasserstein.")
        if not self._is_wasserstein and loss == "wasserstein":
            raise ValueError("If loss is wasserstein, is_wasserstein needs to be true in constructor.")

        if learning_rate_gen is None:
            learning_rate_gen = learning_rate
        if learning_rate_disc is None:
            learning_rate_disc = learning_rate
        self._define_loss(lmbda, loss)
        with tf.name_scope("Optimizer"):
            gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss,
                                                         var_list=self._get_vars(scope="Generator_XY")+self._get_vars(scope="Generator_YX"),
                                                         name="Generator")
            disc_optimizer = optimizer(learning_rate=learning_rate_disc)
            self._disc_optimizer = disc_optimizer.minimize(self._disc_loss,
                                                            var_list=self._get_vars(scope="Discriminator_XY")+self._get_vars(scope="Discriminator_YX"),
                                                            name="Discriminator")
        self._summarise()


    def _define_loss(self, lmbda, loss):
        possible_losses = ["cross-entropy", "L1", "L2", "wasserstein"]
        if loss == "wasserstein":
            self._gen_loss_xy = -tf.reduce_mean(self._output_disc_xy_fake)
            self._gen_loss_yx = -tf.reduce_mean(self._output_disc_yx_fake)

            self._disc_loss_xy = (-(tf.reduce_mean(self._output_disc_xy_real) -
                                    tf.reduce_mean(self._output_disc_xy_fake)) +
                                    10*self._define_gradient_penalty_xy()
            )
            self._disc_loss_yx = (-(tf.reduce_mean(self._output_disc_yx_real) -
                                    tf.reduce_mean(self._output_disc_yx_fake)) +
                                    10*self._define_gradient_penalty_yx()
            )
        elif loss == "cross-entropy":
            logits_xy_real = tf.math.log(self._output_disc_xy_real / (1 - self._output_disc_xy_real))
            logits_yx_real = tf.math.log(self._output_disc_yx_real / (1 - self._output_disc_yx_real))
            logits_xy_fake = tf.math.log(self._output_disc_xy_fake / (1 - self._output_disc_xy_fake))
            logits_yx_fake = tf.math.log(self._output_disc_yx_fake / (1 - self._output_disc_yx_fake))

            self._gen_loss_xy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.ones_like(logits_xy_fake), logits=logits_xy_fake
            ))

            self._gen_loss_yx = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.ones_like(logits_yx_fake), logits=logits_yx_fake
            ))

            self._disc_loss_xy = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.ones_like(logits_xy_real), logits=logits_xy_real
                                    ) +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.zeros_like(logits_xy_fake), logits=logits_xy_fake
                                    )
            )
            self._disc_loss_yx = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.ones_like(logits_yx_real), logits=logits_yx_real
                                    ) +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.zeros_like(logits_yx_fake), logits=logits_yx_fake
                                    )
            )

        elif loss == "L1":
            self._gen_loss_xy = tf.reduce_mean(tf.abs(self._output_disc_xy_fake - tf.ones_like(self._output_disc_xy_fake)))
            self._gen_loss_yx = tf.reduce_mean(tf.abs(self._output_disc_yx_fake - tf.ones_like(self._output_disc_yx_fake)))

            self._disc_loss_xy = (
                        tf.reduce_mean(
                                        tf.abs(self._output_disc_xy_real - tf.ones_like(self._output_disc_xy_real)) +
                                        tf.abs(self._output_disc_xy_fake)
                        )
            ) / 2.0
            self._disc_loss_yx = (
                        tf.reduce_mean(
                                        tf.abs(self._output_disc_yx_real - tf.ones_like(self._output_disc_yx_real)) +
                                        tf.abs(self._output_disc_yx_fake)
                        )
            ) / 2.0

        elif loss == "L2":
            self._gen_loss_xy = tf.reduce_mean(tf.square(self._output_disc_xy_fake - tf.ones_like(self._output_disc_xy_fake)))
            self._gen_loss_yx = tf.reduce_mean(tf.square(self._output_disc_yx_fake - tf.ones_like(self._output_disc_yx_fake)))

            self._disc_loss_xy = (
                        tf.reduce_mean(
                                        tf.square(self._output_disc_xy_real - tf.ones_like(self._output_disc_xy_real)) +
                                        tf.square(self._output_disc_xy_fake)
                        )
            ) / 2.0
            self._disc_loss_yx = (
                        tf.reduce_mean(
                                        tf.square(self._output_disc_yx_real - tf.ones_like(self._output_disc_yx_real)) +
                                        tf.square(self._output_disc_yx_fake)
                        )
            ) / 2.0

        else:
            raise ValueError("Loss not implemented. Choose from {}. Given: {}.".format(possible_losses, loss))

        with tf.name_scope("Loss") as scope:
            self._gen_loss_vanilla = self._gen_loss_xy + self._gen_loss_yx
            tf.summary.scalar("Generator_vanilla_loss_xy", self._gen_loss_xy)
            tf.summary.scalar("Generator_vanilla_loss_yx", self._gen_loss_yx)
            tf.summary.scalar("Generator_vanilla_loss", self._gen_loss_vanilla)

            self._recon_loss_xyx = lmbda*tf.reduce_mean(tf.abs(self._X_input - self._output_gen_xyx))
            self._recon_loss_yxy = lmbda*tf.reduce_mean(tf.abs(self._Y_input - self._output_gen_yxy))
            self._recon_loss = self._recon_loss_xyx + self._recon_loss_yxy
            tf.summary.scalar("Generator_recon_loss_xyx", self._recon_loss_xyx)
            tf.summary.scalar("Generator_recon_loss_yxy", self._recon_loss_yxy)
            tf.summary.scalar("Generator_recon_loss", self._recon_loss)

            self._gen_loss = self._gen_loss_vanilla + self._recon_loss
            tf.summary.scalar("Generator_total_loss", self._gen_loss)

            self._disc_loss = self._disc_loss_xy + self._disc_loss_yx
            tf.summary.scalar("Discriminator_loss_xy", self._disc_loss_xy)
            tf.summary.scalar("Discriminator_loss_yx", self._disc_loss_yx)
            tf.summary.scalar("Discriminator_loss", self._disc_loss)


        self._losses = {name: [] for name in ["Gen_XY", "Gen_YX", "Gen_Vanilla",
                                                "Gen_XYX", "Gen_YXY", "Gen_Recon", "Generator",
                                                "Disc_XY", "Disc_YX", "Discriminator"]}


    def _define_gradient_penalty_xy(self):
        alpha = tf.random_uniform(shape=tf.shape(self._Y_input), minval=0., maxval=1.)
        differences = self._output_gen_xy - self._Y_input
        interpolates = self._Y_input + (alpha * differences)
        gradients = tf.gradients(self._discriminator_xy.generate_net(interpolates, tf_trainflag=self._is_training), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        with tf.name_scope("Loss") as scope:
            self._gradient_penalty_xy = tf.reduce_mean((slopes-1.)**2)
            tf.summary.scalar("Gradient_penalty_xy", self._gradient_penalty_xy)
        return self._gradient_penalty_xy


    def _define_gradient_penalty_yx(self):
        alpha = tf.random_uniform(shape=tf.shape(self._X_input), minval=0., maxval=1.)
        differences = self._output_gen_yx - self._X_input
        interpolates = self._X_input + (alpha * differences)
        gradients = tf.gradients(self._discriminator_yx.generate_net(interpolates, tf_trainflag=self._is_training), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        with tf.name_scope("Loss") as scope:
            self._gradient_penalty_yx = tf.reduce_mean((slopes-1.)**2)
            tf.summary.scalar("Gradient_penalty_yx", self._gradient_penalty_yx)
        return self._gradient_penalty_yx


    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=100, batch_size=64,
              gen_steps=1, disc_steps=5, steps=None, log_step=3, gpu_options=None, shuffle=True):
        if steps is not None:
            gen_steps = 1
            disc_steps = steps
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        self._set_up_test_train_sample(x_train, y_train, x_test, y_test)
        self._log_results(epoch=0, epoch_time=0)
        nr_batches = np.floor(len(x_train) / batch_size)

        for epoch in range(epochs):
            batch_nr = 0
            disc_loss_epoch = 0
            gen_loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            while trained_examples < len(x_train):
                disc_loss_batch, gen_loss_batch = self._optimize(self._trainset, batch_size, disc_steps, gen_steps, shuffle)
                trained_examples += batch_size

                disc_loss_epoch += disc_loss_batch
                gen_loss_epoch += gen_loss_batch

            disc_loss_epoch /= nr_batches
            gen_loss_epoch /= nr_batches
            print("D, D_XY, D_YX: ", self._sess.run([self._disc_loss, self._disc_loss_xy, self._disc_loss_yx],
                                                     feed_dict={self._X_input: x_test, self._Y_input: y_test, self._is_training: False}))
            print("G, G_XY, G_YX: ", self._sess.run([self._gen_loss_vanilla, self._gen_loss_xy, self._gen_loss_yx],
                                                     feed_dict={self._X_input: x_test, self._Y_input: y_test, self._is_training: False}))
            print("R, R_XYX, R_YXY: ", self._sess.run([self._recon_loss, self._recon_loss_xyx, self._recon_loss_yxy],
                                                     feed_dict={self._X_input: x_test, self._Y_input: y_test, self._is_training: False}))
            print("G Total: ", self._sess.run([self._gen_loss],
                                                     feed_dict={self._X_input: x_test, self._Y_input: y_test, self._is_training: False}))

            epoch_train_time = (time.clock() - start)/60
            disc_loss_epoch = np.round(disc_loss_epoch, 7)
            gen_loss_epoch = np.round(gen_loss_epoch, 7)

            print("Epoch {}: Discrimiantor: {} \n\t\t\tGenerator: {}.".format(epoch+1, disc_loss_epoch, gen_loss_epoch))

            if self._log_step is not None:
                self._log(epoch+1, epoch_train_time)


    def _optimize(self, dataset, batch_size, disc_steps, gen_steps, shuffle):
        d_loss = 0
        for i in range(disc_steps):
            if shuffle:
                current_batch_x, _ = dataset._sample_xy(n=batch_size)
                _, current_batch_y = dataset._sample_xy(n=batch_size)
            else:
                current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            _, disc_loss_batch = self._sess.run([
                                            self._disc_optimizer, self._disc_loss
                                            ],
                                            feed_dict={self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                            self._is_training: True})
            d_loss += disc_loss_batch
        d_loss /= disc_steps

        g_loss = 0
        for _ in range(gen_steps):
            if shuffle:
                current_batch_x, _ = dataset._sample_xy(n=batch_size)
                _, current_batch_y = dataset._sample_xy(n=batch_size)
            else:
                current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss],
                                               feed_dict={self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                               self._is_training: True})
            g_loss += gen_loss_batch
        g_loss /= gen_steps
        return d_loss, g_loss


if __name__ == '__main__':
    ########### Image input
    import pickle
    if "lhcb_data2" in os.getcwd():
        data_path = "../../Data/fashion_mnist"
    else:
        data_path = "/home/tneuer/Backup/Algorithmen/0TestData/image_to_image/fashion_mnist"
    with open("{}/train_images.pickle".format(data_path), "rb") as f:
        x_train = pickle.load(f)[0].reshape(-1, 28, 28, 1) / 255
    with open("{}/train_images_rotated.pickle".format(data_path), "rb") as f:
        y_train = pickle.load(f).reshape(-1, 28, 28, 1) / 255


    ########### Architecture

    x_train = padding_zeros(x_train, top=2, bottom=2, left=2, right=2)
    y_train = padding_zeros(y_train, top=2, bottom=2, left=2, right=2)
    gen_xy_architecture = [
                        [conv2d_logged, {"filters": 32, "kernel_size": 5, "strides": 1, "padding": "same", "activation": tf.nn.leaky_relu}],
                        [tf.layers.max_pooling2d, {"pool_size": 2, "strides": 2}],
                        # [unet_original, {"depth": 2, "activation": tf.nn.leaky_relu}],
                        # [inception_block, {"filters": 32}],
                        [conv2d_transpose_logged, {"filters": 32, "kernel_size": 2, "strides": 2, "activation": tf.nn.leaky_relu}],
                        [conv2d_logged, {"filters": 1, "kernel_size": 2, "strides": 1, "padding":"same", "activation": tf.nn.sigmoid}]
                        ]
    disc_xy_architecture = [
                        # [tf.layers.conv2d, {"filters": 32, "kernel_size": 2, "strides": 2, "activation": tf.nn.leaky_relu}],
                        # [residual_block, {"filters": 32, "kernel_size": 2, "activation": tf.nn.leaky_relu, "skip_layers": 3}],
                        # [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": tf.nn.leaky_relu}],
                        # [residual_block, {"filters": 128, "kernel_size": 2, "activation": tf.nn.leaky_relu, "skip_layers": 3}],
                        # [inception_block, {"filters": 32}],
                        # [residual_block, {"filters": 128, "kernel_size": 2, "activation": tf.nn.leaky_relu, "skip_layers": 3}],
                        [conv2d_logged, {"filters": 1, "kernel_size": 2, "strides": 1, "activation": tf.nn.relu}],
                        ]
    gen_yx_architecture = [
                        [conv2d_logged, {"filters": 32, "kernel_size": 5, "strides": 1, "padding": "same", "activation": tf.nn.leaky_relu}],
                        [tf.layers.max_pooling2d, {"pool_size": 2, "strides": 2}],
                        # [unet_original, {"depth": 2, "activation": tf.nn.leaky_relu}],
                        # [inception_block, {"filters": 32}],
                        [conv2d_transpose_logged, {"filters": 32, "kernel_size": 2, "strides": 2, "activation": tf.nn.leaky_relu}],
                        [conv2d_logged, {"filters": 1, "kernel_size": 2, "strides": 1, "padding":"same", "activation": tf.nn.sigmoid}]
                        ]
    disc_yx_architecture = [
                        # [tf.layers.conv2d, {"filters": 32, "kernel_size": 2, "strides": 2, "activation": tf.nn.leaky_relu}],
                        # [residual_block, {"filters": 32, "kernel_size": 2, "activation": tf.nn.leaky_relu, "skip_layers": 3}],
                        # [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": tf.nn.leaky_relu}],
                        # [residual_block, {"filters": 128, "kernel_size": 2, "activation": tf.nn.leaky_relu, "skip_layers": 3}],
                        # [inception_block, {"filters": 32}],
                        # [residual_block, {"filters": 128, "kernel_size": 2, "activation": tf.nn.leaky_relu, "skip_layers": 3}],
                        [conv2d_logged, {"filters": 1, "kernel_size": 2, "strides": 1, "activation": tf.nn.relu}],
                        ]
    # gen_xy_architecture = [
    #                     [tf.layers.flatten, {}],
    #                     [tf.layers.dense, {"units": 128, "activation": tf.nn.relu}],
    #                     [tf.layers.dense, {"units": 32*32, "activation": tf.nn.sigmoid}],
    #                     [reshape_layer, {"shape": [32, 32, 1]}],
    #                     ]
    # disc_xy_architecture = [
    #                     [tf.layers.dense, {"units": 128, "activation": tf.nn.relu}],
    #                     ]
    # gen_yx_architecture = [
    #                     [tf.layers.flatten, {}],
    #                     [tf.layers.dense, {"units": 128, "activation": tf.nn.relu}],
    #                     [tf.layers.dense, {"units": 32*32, "activation": tf.nn.sigmoid}],
    #                     [reshape_layer, {"shape": [32, 32, 1]}],
    #                     ]
    # disc_yx_architecture = [
    #                     [tf.layers.dense, {"units": 128, "activation": tf.nn.relu}],
    #                     ]

    x_test = x_train[-100:]
    x_train = x_train[:-100]
    y_test = y_train[-100:]
    y_train = y_train[:-100]

    lmbda = 10
    learning_rate = 0.0005
    batch_size = 32
    loss = "wasserstein"
    is_patchgan = False
    epochs = 100
    activation = tf.nn.relu
    shuffle = True

    is_wasserstein = True if loss == "wasserstein" else False

    inpt_dim = x_train[0].shape
    opt_dim = y_train[0].shape
    print(inpt_dim)
    print(opt_dim)

    print(np.max(x_train), np.max(y_train))

    config_data = init.create_config_file(globals())

    cyclegan = CycleGAN(x_dim=inpt_dim, y_dim=opt_dim, gen_xy_architecture=gen_xy_architecture,
                     disc_xy_architecture=disc_xy_architecture, gen_yx_architecture=gen_yx_architecture,
                     disc_yx_architecture=disc_yx_architecture,
                    folder="../../Results/Test/CycleGAN", PatchGAN=is_patchgan, is_wasserstein=is_wasserstein)
    print(cyclegan._generator_xy.get_number_params())
    print(cyclegan._discriminator_xy.get_number_params())
    nr_params = cyclegan.get_number_params()
    print(cyclegan.show_architecture())
    with open("../../Results/Test/CycleGAN/config.json", "w") as f:
        json.dump(config_data, f, indent=4)
    cyclegan.log_architecture()
    cyclegan.compile(learning_rate=learning_rate, lmbda=lmbda, loss=loss)
    cyclegan.train(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size,
                   disc_steps=1, gen_steps=5, log_step=1, shuffle=shuffle)