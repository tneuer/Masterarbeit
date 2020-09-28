#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-08-28 22:45:25
    # Description :
####################################################################################
"""
import time
import sys
sys.path.insert(1, "./building_blocks")
sys.path.insert(1, "../Utilities")

import numpy as np
import tensorflow as tf

from building_blocks.layers import logged_dense, reshape_layer
from building_blocks.networks import Generator, Critic
from building_blocks.generativeModels import GenerativeModel

from functionsOnImages import padding_zeros

class WGAN(GenerativeModel):
    def __init__(self, x_dim, z_dim, gen_architecture, critic_architecture, last_layer_activation,
                 folder="./WGAN", image_shape=None
        ):
        super(WGAN, self).__init__(x_dim, z_dim, [gen_architecture, critic_architecture],
                                   last_layer_activation, folder, image_shape)

        self._gen_architecture = self._architectures[0]
        self._critic_architecture = self._architectures[1]

        ################# Define architecture
        if len(self._x_dim) == 1:
            self._gen_architecture.append([logged_dense, {"units": x_dim, "activation": self._last_layer_activation, "name": "Output"}])
        else:
            self._critic_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
        self._critic_architecture.append([logged_dense, {"units": 1, "activation": tf.identity, "name": "Output"}])

        self._generator = Generator(self._gen_architecture, name="Generator")
        self._critic = Critic(self._critic_architecture, name="Critic")

        self._nets = [self._generator, self._critic]

        ################# Connect inputs and networks
        self._output_gen = self._generator.generate_net(self._Z_input, tf_trainflag=self._is_training)
        self._output_critic_fake = self._critic.generate_net(self._output_gen, tf_trainflag=self._is_training)
        self._output_critic_real = self._critic.generate_net(self._X_input, tf_trainflag=self._is_training)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, learning_rate_gen=0.00005, learning_rate_critic=0.00005, optimizer=tf.train.RMSPropOptimizer, **kwargs):
        self._define_loss()
        with tf.name_scope("Optimizer"):
            gen_optimizer = optimizer(learning_rate=learning_rate_gen, **kwargs)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars(scope="Generator"), name="Generator")
            critic_optimizer = optimizer(learning_rate=learning_rate_critic, **kwargs)
            self._critic_optimizer = critic_optimizer.minimize(self._critic_loss, var_list=self._get_vars(scope="Critic"), name="Critic")
        self._clip_critic_param = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self._get_vars(scope="Critic")]
        self._summarise()


    def _define_loss(self):
        with tf.name_scope("Loss") as scope:
            self._gen_loss = -tf.reduce_mean(self._output_critic_fake)
            tf.summary.scalar("Generator_loss", self._gen_loss)
            self._critic_loss = -(tf.reduce_mean(self._output_critic_real) - tf.reduce_mean(self._output_critic_fake))
            tf.summary.scalar("Critic_loss", self._critic_loss)


    def train(self, x_train, x_test=None, epochs=100, batch_size=64, gen_steps=1, critic_steps=5, log_step=3):
        self._set_up_training(log_step=log_step)
        self._set_up_test_train_sample(x_train, x_test)
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

            print("Epoch {}: Critic: {}\n\t\t\tGenerator: {}.".format(epoch, critic_loss_epoch, gen_loss_epoch))

            if log_step is not None:
                self._log(epoch+1, epoch_train_time)


    def _optimize(self, dataset, batch_size, critic_steps, gen_steps):
        for i in range(critic_steps):
            current_batch_x = dataset.get_next_batch(batch_size)
            Z_noise = self.sample_noise(n=len(current_batch_x))
            _, critic_loss_batch, clipping_D = self._sess.run([
                                            self._critic_optimizer, self._critic_loss, self._clip_critic_param],
                                            feed_dict={
                                                    self._X_input: current_batch_x, self._Z_input: Z_noise,
                                                    self._is_training: True
                                            })

        for _ in range(gen_steps):
            Z_noise = self.sample_noise(n=len(current_batch_x))
            _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss], feed_dict={self._Z_input: Z_noise, self._is_training: True})
        return critic_loss_batch, gen_loss_batch


    def predict(self, inpt_x, is_encoded):
        if is_encoded:
            inpt_x = self._generator.generate_samples(noise=inpt_x, sess=self._sess)
        return self._critic.predict(inpt_x, self._sess)





if __name__ == '__main__':
    nr_examples = 60000
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train = x_train[:nr_examples], y_train[:nr_examples]
    x_test, y_test = x_train[:500], y_train[:500]
    x_train = x_train/255.

    ########### Flattened input
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # gen_architecture = [
    #                     [logged_dense, {"units": 256, "activation": tf.nn.relu}],
    #                     [logged_dense, {"units": 512, "activation": tf.nn.relu}],
    #                     ]
    # critic_architecture = [
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
    gen_architecture = [
                        [logged_dense, {"units": 4*4*512, "activation": tf.nn.relu}],
                        [reshape_layer, {"shape": [4, 4, 512]}],

                        [tf.layers.conv2d_transpose, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        # [tf.layers.batch_normalization, {}],
                        # [tf.layers.dropout, {}],

                        [tf.layers.conv2d_transpose, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

                        [tf.layers.conv2d_transpose, {"filters": 1, "kernel_size": 2, "strides": 2, "activation": tf.nn.sigmoid}]
                        ]
    critic_architecture = [
                        [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        # [tf.layers.batch_normalization, {}],
                        # [tf.layers.dropout, {}],

                        [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

                        [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        ]
    inpt_dim = x_train[0].shape
    image_shape=[32, 32, 1]

    z_dim = 128
    wgan = WGAN(x_dim=inpt_dim, z_dim=z_dim, last_layer_activation=tf.nn.sigmoid,
                  gen_architecture=gen_architecture, critic_architecture=critic_architecture,
                  folder="../../Results/Test/WGAN", image_shape=image_shape)
    print(wgan.show_architecture())
    wgan.log_architecture()
    wgan.compile()
    wgan.train(x_train, x_test, epochs=100, critic_steps=5, gen_steps=1)



