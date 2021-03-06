#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-16 22:45:25
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

from building_blocks.layers import reshape_layer, image_condition_concat
from building_blocks.networks import Critic, ConditionalGenerator
from building_blocks.generativeModels import ConditionalGenerativeModel
from functionsOnImages import padding_zeros


class CLSGAN(ConditionalGenerativeModel):
    def __init__(self, x_dim, y_dim, z_dim, gen_architecture, critic_architecture, last_layer_activation,
                 folder="./CLSGAN", image_shape=None, append_y_at_every_layer=None
        ):
        super(CLSGAN, self).__init__(x_dim, y_dim, z_dim, [gen_architecture, critic_architecture],
                                   last_layer_activation, folder, image_shape, append_y_at_every_layer)

        self._gen_architecture = self._architectures[0]
        self._critic_architecture = self._architectures[1]

        ################# Define architecture
        if len(self._x_dim) == 1:
            self._gen_architecture.append([tf.layers.dense, {"units": x_dim, "activation": self._last_layer_activation, "name": "Output"}])
        else:
            self._critic_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            self._gen_architecture[-1][1]["name"] = "Output"
        self._critic_architecture.append([tf.layers.dense, {"units": 1, "activation": tf.identity, "name": "Output"}])

        self._generator = ConditionalGenerator(self._gen_architecture, name="Generator")
        self._critic = Critic(self._critic_architecture, name="Critic")

        self._nets = [self._generator, self._critic]

        ################# Connect inputs and networks
        self._output_gen = self._generator.generate_net(self._mod_Z_input,
                                                        append_elements_at_every_layer=self._append_at_every_layer,
                                                        tf_trainflag=self._is_training)

        with tf.name_scope("InputsCritic"):
            if len(self._x_dim) == 1:
                self._input_real = tf.concat(axis=1, values=[self._X_input, self._Y_input], name="real")
                self._input_fake = tf.concat(axis=1, values=[self._output_gen, self._Y_input], name="fake")
            else:
                self._input_real = image_condition_concat(inputs=self._X_input, condition=self._Y_input, name="real")
                self._input_fake = image_condition_concat(inputs=self._output_gen, condition=self._Y_input, name="fake")

        self._output_critic_real = self._critic.generate_net(self._input_real, tf_trainflag=self._is_training)
        self._output_critic_fake = self._critic.generate_net(self._input_fake, tf_trainflag=self._is_training)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, logged_images=None, logged_labels=None, learning_rate=0.00005,
     learning_rate_gen=None, learning_rate_critic=None, optimizer=tf.train.RMSPropOptimizer):
        if learning_rate_gen is None:
            learning_rate_gen = learning_rate
        if learning_rate_critic is None:
            learning_rate_critic = learning_rate
        self._define_loss()
        with tf.name_scope("Optimizer"):
            gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars("Generator"), name="Generator")
            critic_optimizer = optimizer(learning_rate=learning_rate_critic)
            self._critic_optimizer = critic_optimizer.minimize(self._critic_loss, var_list=self._get_vars("Critic"), name="Critic")
        self._summarise(logged_images=logged_images, logged_labels=logged_labels)


    def _define_loss(self):
        with tf.name_scope("Loss") as scope:
            self._gen_loss = tf.reduce_sum(tf.square(self._output_critic_fake-1))/2
            tf.summary.scalar("Generator_loss", self._gen_loss)
            self._critic_loss = tf.reduce_sum(tf.square(self._output_critic_real-1) + tf.square(self._output_critic_fake))/2
            tf.summary.scalar("Critic_loss_penalized", self._critic_loss)


    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=100, batch_size=64,
              gen_steps=1, critic_steps=5, log_step=3, steps=None, gpu_options=None):
        if steps is not None:
            gen_steps = 1
            critic_steps = steps
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        self._set_up_test_train_sample(x_train, y_train, x_test, y_test)
        self._log_results(epoch=0, epoch_time=0)
        for epoch in range(epochs):
            batch_nr = 0
            critic_loss_epoch = 0
            gen_loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            while trained_examples < len(x_train):
                critic_loss_batch, gen_loss_batch = self._optimize(self._trainset, batch_size, critic_steps, gen_steps)
                trained_examples += batch_size

                critic_loss_epoch += critic_loss_batch
                gen_loss_epoch += gen_loss_batch


            epoch_train_time = (time.clock() - start)/60
            critic_loss_epoch = np.round(critic_loss_epoch, 2)
            gen_loss_epoch = np.round(gen_loss_epoch, 2)

            print("Epoch {}: Critic: {} \n\t\t\tGenerator: {}.".format(epoch+1, critic_loss_epoch, gen_loss_epoch))

            if self._log_step is not None:
                self._log(epoch+1, epoch_train_time)


    def _optimize(self, dataset, batch_size, critic_steps, gen_steps):
        for i in range(critic_steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            Z_noise = self.sample_noise(n=len(current_batch_x))
            _, critic_loss_batch = self._sess.run([
                                            self._critic_optimizer, self._critic_loss
                                            ],
                                            feed_dict={self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                            self._Z_input: Z_noise, self._is_training: True
            })

        for _ in range(gen_steps):
            Z_noise = self._generator.sample_noise(n=len(current_batch_x))
            _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss],
                                               feed_dict={self._Z_input: Z_noise, self._Y_input: current_batch_y,
                                               self._is_training: True})

        return critic_loss_batch, gen_loss_batch


    def predict(self, inpt_x, inpt_y, is_encoded):
        if is_encoded:
            inpt_x = self._sess.run(self._mod_Z_input, feed_dict={self._Z_input: inpt_x, self._Y_input_oneHot: inpt_y, self._is_training: True})
            inpt_x = self._generator.generate_samples(noise=inpt_x, sess=self._sess)
        inpt = self._sess.run(self._input_real, feed_dict={self._X_input: inpt_x, self._Y_input: inpt_y, self._is_training: True})
        return self._critic.predict(inpt, self._sess)





if __name__ == '__main__':
    from sklearn.preprocessing import OneHotEncoder
    nr_examples = 60000
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train = x_train[:nr_examples], y_train[:nr_examples]
    x_test, y_test = x_train[:500], y_train[:500]
    x_train, x_test = x_train/255., x_test/255.

    y_train_log = np.identity(10)
    enc = OneHotEncoder(sparse=False)


    ########### Flattened input
    x_train_log = np.array([x_train[y_train.tolist().index(i)] for i in range(10)])
    x_train_log = np.reshape(x_train_log, newshape=(-1, 28, 28, 1))
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_train = enc.fit_transform(y_train.reshape(-1, 1))
    y_test = enc.transform(y_test.reshape(-1, 1))
    gen_architecture = [
                        [tf.layers.dense, {"units": 128, "activation": tf.nn.relu}],
                        [tf.layers.dense, {"units": 256, "activation": tf.nn.relu}],
                        [tf.layers.dense, {"units": 512, "activation": tf.nn.relu}],
                        ]
    critic_architecture = [
                        [tf.layers.dense, {"units": 512, "activation": tf.nn.relu}],
                        [tf.layers.dense, {"units": 256, "activation": tf.nn.relu}],
                        [tf.layers.dense, {"units": 128, "activation": tf.nn.relu}],
                        ]
    inpt_dim = 784
    image_shape=[28, 28, 1]



    ########### Image input
    # x_train = padding_zeros(x_train, top=2, bottom=2, left=2, right=2)
    # x_train = np.reshape(x_train, newshape=(-1, 32, 32, 1))
    # x_test = padding_zeros(x_test, top=2, bottom=2, left=2, right=2)
    # x_test = np.reshape(x_test, newshape=(-1, 32, 32, 1))
    # x_train_log = [x_train[y_train.tolist().index(i)] for i in range(10)]
    # x_train_log = np.reshape(x_train_log, newshape=(-1, 32, 32, 1))

    # y_train = enc.fit_transform(y_train.reshape(-1, 1))
    # y_test = enc.transform(y_test.reshape(-1, 1))

    # gen_architecture = [
    #                     [tf.layers.dense, {"units": 4*4*512, "activation": tf.nn.relu}],
    #                     [reshape_layer, {"shape": [4, 4, 512]}],

    #                     [tf.layers.conv2d_transpose, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
    #                     # [tf.layers.batch_normalization, {}],
    #                     # [tf.layers.dropout, {}],

    #                     [tf.layers.conv2d_transpose, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

    #                     [tf.layers.conv2d_transpose, {"filters": 1, "kernel_size": 2, "strides": 2, "activation": tf.nn.sigmoid}]
    #                     ]
    # critic_architecture = [
    #                     [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
    #                     # [tf.layers.batch_normalization, {}],
    #                     # [tf.layers.dropout, {}],

    #                     [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

    #                     [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
    #                     ]
    # inpt_dim = x_train[0].shape
    # image_shape=[32, 32, 1]




    z_dim = 128
    label_dim = 10
    gpu_options = None
    clsgan = CLSGAN(x_dim=inpt_dim, y_dim=label_dim, z_dim=z_dim, last_layer_activation=tf.nn.sigmoid,
                  gen_architecture=gen_architecture, critic_architecture=critic_architecture,
                  folder="../../Results/Test/CLSGAN", image_shape=image_shape, append_y_at_every_layer=False)
    print(clsgan.show_architecture())
    clsgan.log_architecture()
    clsgan.compile(logged_images=x_train_log, logged_labels=y_train_log)
    clsgan.train(x_train, y_train, x_test, y_test, epochs=100, critic_steps=5, gen_steps=1, log_step=3,
                  gpu_options=gpu_options)