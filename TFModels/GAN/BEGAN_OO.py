#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-07 17:34:25
    # Description :
####################################################################################
"""
import os
import sys
sys.path.insert(1, "./building_blocks")
sys.path.insert(1, "../Utilities")
import time

import numpy as np
import tensorflow as tf

from building_blocks.layers import reshape_layer, logged_dense, resize_layer
from building_blocks.networks import Generator, Encoder, Decoder
from building_blocks.generativeModels import GenerativeModel
from functionsOnImages import padding_zeros


class BEGAN(GenerativeModel):
    def __init__(self, x_dim, z_dim, enc_dim, gen_architecture, enc_architecture, dec_architecture,
                 last_layer_activation, folder="./BEGAN", image_shape=None
        ):
        super(BEGAN, self).__init__(x_dim, z_dim, [gen_architecture, enc_architecture, dec_architecture],
                                   last_layer_activation, folder, image_shape)

        self._gen_architecture = self._architectures[0]
        self._enc_architecture = self._architectures[1]
        self._dec_architecture = self._architectures[2]

        self._enc_dim = enc_dim
        self._kt = 0
        self._gamma = 1.
        self._lambda = 0.001

        ################# Define architecture
        if len(self._x_dim) == 1:
            self._gen_architecture.append([logged_dense, {"units": x_dim, "activation": self._last_layer_activation, "name": "Output"}])
            self._dec_architecture.append([logged_dense, {"units": x_dim, "activation": self._last_layer_activation, "name": "Output"}])
        else:
            self._enc_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            self._gen_architecture[-1][1]["name"] = "Output"
        self._enc_architecture.append([logged_dense, {"units": enc_dim, "activation": tf.nn.tanh, "name": "Output"}])

        self._generator = Generator(self._gen_architecture, name="Generator")
        self._encoder = Encoder(self._enc_architecture, name="Encoder")
        self._decoder = Encoder(self._dec_architecture, name="Decoder")

        self._nets = [self._generator, self._encoder, self._decoder]

        ################# Connect inputs and networks
        self._output_gen = self._generator.generate_net(self._Z_input, tf_trainflag=self._is_training)


        self._output_encoder_real = self._encoder.generate_net(self._X_input, tf_trainflag=self._is_training)
        self._output_encoder_fake = self._encoder.generate_net(self._output_gen, tf_trainflag=self._is_training)

        self._output_decoder_real = self._decoder.generate_net(self._output_encoder_real, tf_trainflag=self._is_training)
        self._output_decoder_fake = self._decoder.generate_net(self._output_encoder_fake, tf_trainflag=self._is_training)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, learning_rate_gen=0.0001, learning_rate_ae=0.0001, optimizer=tf.train.AdamOptimizer):
        self._define_loss()
        with tf.name_scope("Optimizer"):
            gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars("Generator"), name="Generator")
            ae_optimizer = optimizer(learning_rate=learning_rate_ae)
            self._ae_optimizer = ae_optimizer.minimize(self._ae_loss, var_list=self._get_vars("Encoder")+ self._get_vars("Decoder"), name="Autoencoder")
        self._summarise()


    def _define_loss(self):
        with tf.name_scope("Loss") as scope:
            self._loss_real = tf.reduce_mean(tf.abs(self._X_input - self._output_decoder_real))
            self._loss_fake = tf.reduce_mean(tf.abs(self._output_gen - self._output_decoder_fake))
            self._gen_loss = self._loss_fake
            tf.summary.scalar("Generator_loss", self._gen_loss)
            self._ae_loss = self._loss_real - self._kt*self._loss_fake
            tf.summary.scalar("Autoencoder_loss", self._ae_loss)
            self._global_loss = self._loss_real + tf.abs(self._gamma*self._loss_real - self._loss_fake)
            tf.summary.scalar("Global_loss", self._global_loss)


    def train(self, x_train, x_test=None, epochs=100, batch_size=64, log_step=3, steps=None, gpu_options=None):
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        self._set_up_test_train_sample(x_train, x_test)
        self._log_results(epoch=0, epoch_time=0)
        for epoch in range(epochs):
            batch_nr = 0
            ae_loss_epoch = 0
            gen_loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            while trained_examples < len(x_train):
                ae_loss_batch, gen_loss_batch, loss_real_batch, loss_fake_batch = self._optimize(self._trainset, batch_size)
                trained_examples += batch_size

                ae_loss_epoch += ae_loss_batch
                gen_loss_epoch += gen_loss_batch

                self._kt = np.maximum(np.minimum(1., self._kt + self._lambda * (self._gamma * loss_real_batch - loss_fake_batch)), 0.)

            epoch_train_time = (time.clock() - start)/60
            ae_loss_epoch = np.round(ae_loss_epoch, 2)
            gen_loss_epoch = np.round(gen_loss_epoch, 2)

            print("Epoch {}: Autoencoder: {} \n\t\t\tGenerator: {}.".format(epoch+1, ae_loss_epoch, gen_loss_epoch))

            if self._log_step is not None:
                self._log(epoch+1, epoch_train_time)


    def _optimize(self, dataset, batch_size):
        current_batch_x = dataset.get_next_batch(batch_size)
        Z_noise = self.sample_noise(n=len(current_batch_x))
        _, ae_loss_batch, loss_real_batch, loss_fake_batch = self._sess.run([
                                        self._ae_optimizer, self._ae_loss, self._loss_real, self._loss_fake
                                        ],
                                        feed_dict={self._X_input: current_batch_x, self._Z_input: Z_noise,
                                        self._is_training: True
        })

        Z_noise = self._generator.sample_noise(n=len(current_batch_x))
        _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss],
                                           feed_dict={self._Z_input: Z_noise, self._is_training: True})

        return ae_loss_batch, gen_loss_batch, loss_real_batch, loss_fake_batch






if __name__ == '__main__':
    from sklearn.preprocessing import OneHotEncoder
    nr_examples = 60000
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train = x_train[:nr_examples], y_train[:nr_examples]
    x_test, y_test = x_train[:500], y_train[:500]
    x_train = x_train/255.

    y_train_log = np.identity(10)
    enc = OneHotEncoder(sparse=False)


    ########### Flattened input
    # x_train_log = np.array([x_train[y_train.tolist().index(i)] for i in range(10)])
    # x_train_log = np.reshape(x_train_log, newshape=(-1, 28, 28, 1))
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # y_train = enc.fit_transform(y_train.reshape(-1, 1))
    # y_test = enc.transform(y_test.reshape(-1, 1))
    # gen_architecture = [
    #                     [tf.layers.dense, {"units": 256, "activation": tf.nn.elu}],
    #                     [tf.layers.dense, {"units": 512, "activation": tf.nn.elu}],
    #                     ]
    # enc_architecture = [
    #                     [tf.layers.dense, {"units": 512, "activation": tf.nn.elu}],
    #                     [tf.layers.dense, {"units": 256, "activation": tf.nn.elu}],
    #                     ]
    # dec_architecture = [
    #                     [tf.layers.dense, {"units": 256, "activation": tf.nn.elu}],
    #                     [tf.layers.dense, {"units": 512, "activation": tf.nn.elu}],
    #                     ]
    # inpt_dim = 784
    # image_shape=[28, 28, 1]



    ########### Image input
    x_train = padding_zeros(x_train, top=2, bottom=2, left=2, right=2)
    x_train = np.reshape(x_train, newshape=(-1, 32, 32, 1))
    x_test = padding_zeros(x_test, top=2, bottom=2, left=2, right=2)
    x_test = np.reshape(x_test, newshape=(-1, 32, 32, 1))
    x_train_log = [x_train[y_train.tolist().index(i)] for i in range(10)]
    x_train_log = np.reshape(x_train_log, newshape=(-1, 32, 32, 1))

    y_train = enc.fit_transform(y_train.reshape(-1, 1))
    y_test = enc.transform(y_test.reshape(-1, 1))

    gen_architecture = [
                        [tf.layers.dense, {"units": 8*8*128, "activation": tf.nn.relu}],
                        [reshape_layer, {"shape": [8, 8, 128]}],
                        [tf.layers.conv2d, {"filters": 128, "kernel_size": 3, "strides": 1, "activation": tf.nn.relu, "padding": "SAME"}],
                        [resize_layer, {"size": 16}],
                        [tf.layers.conv2d, {"filters": 64, "kernel_size": 3, "strides": 1, "activation": tf.nn.relu, "padding": "SAME"}],
                        [resize_layer, {"size": 32}],
                        [tf.layers.conv2d, {"filters": 1, "kernel_size": 3, "strides": 1, "activation": tf.nn.sigmoid, "padding": "SAME"}]
                        ]
    enc_architecture = [
                        [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        ]
    dec_architecture = [
                        [tf.layers.dense, {"units": 8*8*128, "activation": tf.nn.relu}],
                        [reshape_layer, {"shape": [8, 8, 128]}],
                        [tf.layers.conv2d, {"filters": 128, "kernel_size": 3, "strides": 1, "activation": tf.nn.relu, "padding": "SAME"}],
                        [resize_layer, {"size": 16}],
                        [tf.layers.conv2d, {"filters": 64, "kernel_size": 3, "strides": 1, "activation": tf.nn.relu, "padding": "SAME"}],
                        [resize_layer, {"size": 32}],
                        [tf.layers.conv2d, {"filters": 1, "kernel_size": 3, "strides": 1, "activation": tf.identity, "padding": "SAME"}]
                        ]

    inpt_dim = x_train[0].shape
    image_shape=[32, 32, 1]




    z_dim = 128
    enc_dim = z_dim
    began = BEGAN(x_dim=inpt_dim, z_dim=z_dim, enc_dim=enc_dim, last_layer_activation=tf.nn.sigmoid,
                  gen_architecture=gen_architecture, enc_architecture=enc_architecture, dec_architecture=dec_architecture,
                  folder="../../Results/Test/BEGAN_conv", image_shape=image_shape)
    print(began.show_architecture())
    began.log_architecture()
    began.compile()
    began.train(x_train, x_test, epochs=100, log_step=3, gpu_options=None)