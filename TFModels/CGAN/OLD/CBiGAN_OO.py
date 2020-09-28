#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-25-11 18:21:18
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
from building_blocks.networks import ConditionalGenerator, Discriminator, Encoder
from building_blocks.generativeModels import ConditionalGenerativeModel
from functionsOnImages import padding_zeros

class CBiGAN(ConditionalGenerativeModel):
    def __init__(self, x_dim, y_dim, z_dim, gen_architecture, disc_architecture, enc_architecture, last_layer_activation,
                 folder="./Results/CBiGAN_log", image_shape=None, append_y_at_every_layer=None
        ):
        super(CBiGAN, self).__init__(x_dim, y_dim, z_dim, [gen_architecture, disc_architecture, enc_architecture],
                                   last_layer_activation, folder, image_shape, append_y_at_every_layer)

        self._gen_architecture = self._architectures[0]
        self._disc_architecture = self._architectures[1]
        self._enc_architecture = self._architectures[2]

        ################# Define architecture
        if len(self._x_dim) == 1:
            self._gen_architecture.append([logged_dense, {"units": x_dim, "activation": self._last_layer_activation, "name": "Output"}])
        else:
            self._disc_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            self._enc_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            self._gen_architecture[-1][1]["name"] = "Output"
        self._disc_architecture.append([logged_dense, {"units": 1, "activation": tf.sigmoid, "name": "Output"}])
        self._enc_architecture.append([logged_dense, {"units": z_dim, "activation": tf.identity, "name": "Output"}])

        self._generator = ConditionalGenerator(self._gen_architecture, name="Generator")
        self._discriminator = Discriminator(self._disc_architecture, name="Discriminator")
        self._encoder = Encoder(self._enc_architecture, name="Encoder")

        self._nets = [self._generator, self._discriminator, self._encoder]

        ################# Connect inputs and networks
        with tf.name_scope("InputsEncoder"):
            if len(self._x_dim) == 1:
                self._mod_X_input = tf.concat(axis=1, values=[self._X_input, self._Y_input], name="modified_x")
            else:
                self._mod_X_input = image_condition_concat(images=self._X_input, condition=self._Y_input, name="fake")


        self._output_gen = self._generator.generate_net(self._mod_Z_input,
                                                        append_elements_at_every_layer=self._append_at_every_layer,
                                                        tf_trainflag=self._is_training)
        self._output_enc = self._encoder.generate_net(self._mod_X_input,
                                                        append_elements_at_every_layer=self._append_at_every_layer,
                                                        tf_trainflag=self._is_training)


        with tf.name_scope("InputsCritic"):
            self._mod_output_enc = tf.concat(axis=1, values=[self._output_enc, self._Y_input], name="modified_encoder")
            if len(self._x_dim) == 1:
                self._disc_input_fake = tf.concat(axis=1, values=[self._output_gen, self._mod_Z_input], name="fake")
                self._disc_input_real = tf.concat(axis=1, values=[self._X_input, self._mod_output_enc], name="real")
            else:
                self._disc_input_fake = image_condition_concat(inputs=self._output_gen, condition=self._mod_Z_input, name="fake")
                self._disc_input_real = image_condition_concat(inputs=self._X_input, condition=self._mod_output_enc, name="real")

        self._output_disc_fake = self._discriminator.generate_net(self._disc_input_fake,
                                                        append_elements_at_every_layer=self._append_at_every_layer,
                                                        tf_trainflag=self._is_training)
        self._output_disc_real = self._discriminator.generate_net(self._disc_input_real,
                                                        append_elements_at_every_layer=self._append_at_every_layer,
                                                        tf_trainflag=self._is_training)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, logged_images=None, logged_labels=None,
                learning_rate=0.0003, learning_rate_gen=None, learning_rate_disc=None, optimizer=tf.train.AdamOptimizer):
        self._define_loss()
        if learning_rate_gen is None:
            learning_rate_gen = learning_rate
        if learning_rate_disc is None:
            learning_rate_disc = learning_rate
        with tf.name_scope("Optimizer"):
            gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars(scope="Generator") + self._get_vars(scope="Encoder"), name="Generator")
            disc_optimizer = optimizer(learning_rate=learning_rate_disc)
            self._disc_optimizer = disc_optimizer.minimize(self._disc_loss, var_list=self._get_vars(scope="Discriminator"), name="Discriminator")
        self._summarise(logged_images=logged_images, logged_labels=logged_labels)


    def _define_loss(self):
        with tf.name_scope("Loss") as scope:
            self._gen_loss = -tf.reduce_mean( tf.log(self._output_disc_fake+0.00001) + tf.log(1.0-self._output_disc_real+0.00001) )
            tf.summary.scalar("Generator_loss", self._gen_loss)
            self._disc_loss = -tf.reduce_mean( tf.log(self._output_disc_real+0.00001) + tf.log(1.0-self._output_disc_fake+0.00001) )
            tf.summary.scalar("Discriminator_loss", self._disc_loss)


    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=100, batch_size=64,
              gen_steps=1, disc_steps=1, steps=None, log_step=3, gpu_options=None):
        if steps is not None:
            gen_steps = 1
            disc_steps = steps
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        self._set_up_test_train_sample(x_train, y_train, x_test, y_test)
        self._log_results(epoch=0, epoch_time=0)
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

            epoch_train_time = (time.clock() - start)/60
            disc_loss_epoch = np.round(disc_loss_epoch, 2)
            gen_loss_epoch = np.round(gen_loss_epoch, 2)

            acc_real = self.get_accuracy(inpt=self._x_test, inpt_y=self._y_test, labels=np.ones(len(self._x_test)), is_encoded=False)
            acc_fake = self.get_accuracy(inpt=self._z_test, inpt_y=self._y_test, labels=np.zeros(len(self._z_test)), is_encoded=True)
            print("Epoch {}: Discriminator: {} ({})\n\t\t\tGenerator: {} ({}).".format(epoch, disc_loss_epoch, acc_real, gen_loss_epoch, acc_fake))

            if log_step is not None:
                self._log(epoch+1, epoch_train_time)


    def _optimize(self, dataset, batch_size, disc_steps, gen_steps):
        for i in range(disc_steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            Z_noise = self.sample_noise(n=len(current_batch_x))
            _, disc_loss_batch = self._sess.run([self._disc_optimizer, self._disc_loss],
                                                feed_dict={
                                                    self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                                    self._Z_input: Z_noise, self._is_training: True
                                                })

        for _ in range(gen_steps):
            Z_noise = self.sample_noise(n=len(current_batch_x))
            _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss],
                                               feed_dict={
                                                    self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                                    self._Z_input: Z_noise, self._is_training: True
                                                })
        return disc_loss_batch, gen_loss_batch


    def get_accuracy(self, inpt, inpt_y, labels, is_encoded):
        if not is_encoded:
            inpt = self._sess.run(self._disc_input_real, feed_dict={self._X_input: inpt, self._Y_input: inpt_y, self._is_training: False})
        else:
            inpt = self._sess.run(self._disc_input_fake, feed_dict={self._Z_input: inpt, self._Y_input: inpt_y, self._is_training: False})
        return self._discriminator.get_accuracy(inpt, labels, self._sess)


    def predict(self, inpt_x, inpt_y, is_encoded):
        if not is_encoded:
            inpt_x = self._encoder.encode(inpt=inpt_x, sess=self._sess)
        inpt_x = self._sess.run(self._mod_X_input, feed_dict={self._X_input: inpt_x, self._Y_input: inpt_y, self._is_training: False})
        return self._discriminator.predict(inpt_x, self._sess)


    def generate_image_from_noise(self, n):
        noise = self.sample_noise(n=n)
        return self.generate_samples(noise, self._sess)





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
    x_train_log = np.array([x_train[y_train.tolist().index(i)] for i in range(10)])
    x_train_log = np.reshape(x_train_log, newshape=(-1, 28, 28, 1))
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_train = enc.fit_transform(y_train.reshape(-1, 1))
    y_test = enc.transform(y_test.reshape(-1, 1))
    gen_architecture = [
                        [logged_dense, {"units": 256, "activation": tf.nn.relu}],
                        [logged_dense, {"units": 512, "activation": tf.nn.relu}],
                        ]
    disc_architecture = [
                        [logged_dense, {"units": 512, "activation": tf.nn.relu}],
                        [logged_dense, {"units": 256, "activation": tf.nn.relu}],
                        ]
    enc_architecture = [
                        [logged_dense, {"units": 512, "activation": tf.nn.relu}],
                        [logged_dense, {"units": 256, "activation": tf.nn.relu}],
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
    #                     [logged_dense, {"units": 4*4*512, "activation": tf.nn.relu}],
    #                     [reshape_layer, {"shape": [4, 4, 512]}],

    #                     [tf.layers.conv2d_transpose, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
    #                     # [tf.layers.batch_normalization, {}],
    #                     # [tf.layers.dropout, {}],

    #                     [tf.layers.conv2d_transpose, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

    #                     [tf.layers.conv2d_transpose, {"filters": 1, "kernel_size": 2, "strides": 2, "activation": tf.nn.sigmoid}]
    #                     ]
    # disc_architecture = [
    #                     [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
    #                     # [tf.layers.batch_normalization, {}],
    #                     # [tf.layers.dropout, {}],

    #                     [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

    #                     [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
    #                     ]
    # enc_architecture = [
    #                     [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
    #                     # [tf.layers.batch_normalization, {}],
    #                     # [tf.layers.dropout, {}],

    #                     [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

    #                     [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
    #                     ]
    # inpt_dim = x_train[0].shape
    # image_shape=[32, 32, 1]

    z_dim = 64
    label_dim = 10
    cbigan = CBiGAN(x_dim=inpt_dim, y_dim=label_dim, z_dim=z_dim, last_layer_activation=tf.nn.sigmoid,
                  gen_architecture=gen_architecture, disc_architecture=disc_architecture, enc_architecture=enc_architecture,
                  folder="../../Results/Test/CBiGAN", image_shape=image_shape, append_y_at_every_layer=True)
    cbigan.log_architecture()
    print(cbigan.show_architecture())
    cbigan.compile(logged_images=x_train_log, logged_labels=y_train_log)
    cbigan.train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=100, gen_steps=1, disc_steps=1)
