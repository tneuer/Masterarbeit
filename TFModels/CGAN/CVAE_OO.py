#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-21 14:21:44
    # Description :
####################################################################################
"""
import time
import sys
sys.path.insert(1, "./building_blocks")
sys.path.insert(1, "../Utilities")

import numpy as np
import tensorflow as tf

from building_blocks.layers import logged_dense, reshape_layer, image_condition_concat
from building_blocks.networks import Encoder, ConditionalDecoder
from building_blocks.generativeModels import ConditionalGenerativeModel
from functionsOnImages import padding_zeros


class CVAE(ConditionalGenerativeModel):
    def __init__(self, x_dim, y_dim, z_dim, dec_architecture, enc_architecture, last_layer_activation,
                 folder="./VAE", image_shape=None, append_y_at_every_layer=None
        ):
        super(CVAE, self).__init__(x_dim, y_dim, z_dim, [dec_architecture, enc_architecture],
                                   last_layer_activation, folder, image_shape, append_y_at_every_layer)

        self._gen_architecture = self._architectures[0]
        self._enc_architecture = self._architectures[1]

        ################# Define architecture
        if len(self._x_dim) == 1:
            self._gen_architecture.append([logged_dense, {"units": x_dim, "activation": self._last_layer_activation, "name": "Output"}])
        else:
            self._enc_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            self._gen_architecture[-1][1]["name"] = "Output"
        last_layer_mean = [logged_dense, {"units": z_dim, "activation": tf.identity, "name": "Mean"}]
        last_layer_std = [logged_dense, {"units": z_dim, "activation": tf.identity, "name": "Std"}]

        self._encoder_mean = Encoder(self._enc_architecture + [last_layer_mean], name="Encoder")
        self._encoder_std = Encoder(self._enc_architecture + [last_layer_std], name="Encoder")
        self._generator = ConditionalDecoder(self._gen_architecture, name="Generator")

        self._nets = [self._generator, self._encoder_mean]

        ################# Connect inputs and networks
        with tf.name_scope("InputsEncoder"):
            if len(self._x_dim) == 1:
                self._mod_X_input = tf.concat(axis=1, values=[self._X_input, self._Y_input], name="real")
            else:
                self._mod_X_input = image_condition_concat(inputs=self._X_input, condition=self._Y_input, name="real")

        self._mean_layer = self._encoder_mean.generate_net(self._mod_X_input,
                                                           append_elements_at_every_layer=self._append_at_every_layer,
                                                           tf_trainflag=self._is_training)
        self._std_layer = self._encoder_std.generate_net(self._mod_X_input,
                                                         append_elements_at_every_layer=self._append_at_every_layer,
                                                         tf_trainflag=self._is_training)
        self._output_enc = self._mean_layer + tf.exp(0.5*self._std_layer)*self._Z_input

        with tf.name_scope("InputsGenerator"):
            self._dec_input = tf.concat(axis=1, values=[self._output_enc, self._Y_input], name="latent")

        self._output_dec = self._generator.generate_net(self._dec_input,
                                                        append_elements_at_every_layer=self._append_at_every_layer,
                                                        tf_trainflag=self._is_training)
        self._output_dec_from_encoding = self._generator.generate_net(self._mod_Z_input,
                                                                      append_elements_at_every_layer=self._append_at_every_layer,
                                                                      tf_trainflag=self._is_training)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, logged_images=None, logged_labels=None, learning_rate=0.0001, optimizer=tf.train.AdamOptimizer):
        self._define_loss()
        with tf.name_scope("Optimizer"):
            vae_optimizer = optimizer(learning_rate=learning_rate)
            self._vae_optimizer = vae_optimizer.minimize(self._vae_loss, name="VAE")
        self._summarise(logged_images=logged_images, logged_labels=logged_labels)


    def _define_loss(self):
        with tf.name_scope("Loss") as scope:
            self._data_fidelity_loss = self._X_input*tf.log(1e-10 + self._output_dec) + (1 - self._X_input)*tf.log(1e-10 + 1 - self._output_dec)
            self._data_fidelity_loss = -tf.reduce_sum(self._data_fidelity_loss, 1)

            self._KLdiv = 1 + self._std_layer - tf.square(self._mean_layer) - tf.exp(self._std_layer)
            self._KLdiv = -0.5*tf.reduce_sum(self._KLdiv, 1)

            self._vae_loss = tf.reduce_mean(self._data_fidelity_loss + self._KLdiv)
            tf.summary.scalar("Loss", self._vae_loss)



    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=100, batch_size=64, steps=5, log_step=3, gpu_options=None):
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        self._set_up_test_train_sample(x_train, y_train, x_test, y_test)
        self._log_results(epoch=0, epoch_time=0)
        for epoch in range(epochs):
            batch_nr = 0
            loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            while trained_examples < len(x_train):
                loss_batch = self._optimize(self._trainset, batch_size, steps)
                trained_examples += batch_size
                loss_epoch += loss_batch

            epoch_train_time = (time.clock() - start)/60
            loss_epoch = np.round(loss_epoch, 2)

            print("Epoch {}: Loss: {}.".format(epoch, loss_epoch))

            if log_step is not None:
                self._log(epoch+1, epoch_train_time)


    def _optimize(self, dataset, batch_size, steps):
        for i in range(steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            Z_noise = self.sample_noise(n=len(current_batch_x))
            _, loss_batch = self._sess.run([self._vae_optimizer, self._vae_loss],
                                           feed_dict={self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                           self._Z_input: Z_noise, self._is_training: True
            })
        return loss_batch


    def decode(self, inpt_x, is_encoded):
        if not is_encoded:
            inpt_x = self._encoder_mean.encode(noise=inpt_x, sess=self._sess)
        return self._generator.decode(inpt_x, self._sess)






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
    # y_train = enc.fit_transform(y_train.reshape([-1, 1]))
    # y_test = enc.fit_transform(y_test.reshape([-1, 1]))
    # decoder_architecture = [
    #                     [logged_dense, {"units": 256, "activation": tf.nn.relu}],
    #                     [logged_dense, {"units": 512, "activation": tf.nn.relu}],
    #                     ]
    # encoder_architecture = [
    #                     [logged_dense, {"units": 512, "activation": tf.nn.relu}],
    #                     [logged_dense, {"units": 256, "activation": tf.nn.relu}],
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

    decoder_architecture = [
                        [logged_dense, {"units": 4*4*512, "activation": tf.nn.relu}],
                        [reshape_layer, {"shape": [4, 4, 512]}],

                        [tf.layers.conv2d_transpose, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

                        [tf.layers.conv2d_transpose, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

                        [tf.layers.conv2d_transpose, {"filters": 1, "kernel_size": 2, "strides": 2, "activation": tf.nn.sigmoid}]
                        ]
    encoder_architecture = [
                        [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

                        [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        # [tf.layers.batch_normalization, {}],
                        # [tf.layers.dropout, {}],

                        [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        ]
    inpt_dim = x_train[0].shape
    image_shape=[32, 32, 1]




    z_dim = 64
    label_dim = 10
    cvae = CVAE(x_dim=inpt_dim, y_dim=label_dim, z_dim=z_dim, last_layer_activation=tf.nn.sigmoid,
                  enc_architecture=encoder_architecture, dec_architecture=decoder_architecture,
                  folder="../../Results/Test/CVAE_conv2", image_shape=image_shape, append_y_at_every_layer=True)
    cvae.log_architecture()
    print(cvae.show_architecture())
    cvae.compile(logged_images=x_train_log, logged_labels=y_train_log)
    cvae.train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=100, steps=3)

