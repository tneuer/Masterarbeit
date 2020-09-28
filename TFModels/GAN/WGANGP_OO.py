#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-10-20 18:04:17
    # Description :
####################################################################################
"""
import time
import sys
sys.path.insert(1, "./building_blocks")
sys.path.insert(1, "../Utilities")

import numpy as np
import tensorflow as tf

from building_blocks.layers import logged_dense
from building_blocks.networks import Generator, Critic
from building_blocks.generativeModels import GenerativeModel

class WGANGP(GenerativeModel):
    def __init__(self, x_dim, z_dim, gen_architecture, critic_architecture, last_layer_activation,
                 folder="./WGANGP", image_shape=None
        ):
        super(WGANGP, self).__init__(x_dim, z_dim, [gen_architecture, critic_architecture],
                                   last_layer_activation, folder, image_shape)

        self._gen_architecture = self._architectures[0]
        self._critic_architecture = self._architectures[1]

        ################# Define architecture
        self._gen_architecture.append([tf.layers.dense, {"units": x_dim, "activation": self._last_layer_activation, "name": "Output"}])
        self._generator = Generator(self._gen_architecture, name="Generator")

        self._critic_architecture.append([tf.layers.dense, {"units": 1, "activation": tf.identity, "name": "Output"}])
        self._critic = Critic(self._critic_architecture, name="Critic")

        self._nets = [self._generator, self._critic]

        ################# Connect inputs and networks
        self._output_gen = self._generator.generate_net(self._Z_input)
        self._output_critic_fake = self._critic.generate_net(self._output_gen)
        self._output_critic_real = self._critic.generate_net(self._X_input)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, learning_rate_gen=0.00005, learning_rate_critic=0.00005, optimizer=tf.train.RMSPropOptimizer):
        self._define_loss()
        with tf.name_scope("Optimizer"):
            gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars("Generator"), name="Generator")
            critic_optimizer = optimizer(learning_rate=learning_rate_critic)
            self._critic_optimizer = critic_optimizer.minimize(self._critic_loss, var_list=self._get_vars("Critic"), name="Critic")
        self._summarise()


    def _define_loss(self):
        self._define_gradient_penalty()
        with tf.name_scope("Loss") as scope:
            self._gen_loss = -tf.reduce_mean(self._output_critic_fake)
            tf.summary.scalar("Generator_loss", self._gen_loss)
            self._critic_loss_original = -(tf.reduce_mean(self._output_critic_real) - tf.reduce_mean(self._output_critic_fake))
            tf.summary.scalar("Critic_loss_original", self._critic_loss_original)
            self._critic_loss = self._critic_loss_original + 10*self._define_gradient_penalty()
            tf.summary.scalar("Critic_loss_penalized", self._critic_loss)


    def _define_gradient_penalty(self):
        alpha = tf.random_uniform(shape=tf.shape(self._X_input), minval=0., maxval=1.)
        differences = self._output_gen - self._X_input
        interpolates = self._X_input + (alpha * differences)
        gradients = tf.gradients(self._critic.generate_net(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        self._gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        tf.summary.scalar("Gradient_penalty", self._gradient_penalty)
        return self._gradient_penalty


    def train(self, x_train, x_test=None, epochs=100, batch_size=64, gen_steps=1, critic_steps=5, log_step=3,
              steps=None, gpu_options=None):
        if steps is not None:
            gen_steps = 1
            critic_steps = steps
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
            _, critic_loss_batch = self._sess.run([self._critic_optimizer, self._critic_loss], feed_dict={self._X_input: current_batch_x, self._Z_input: Z_noise})

        for _ in range(gen_steps):
            Z_noise = self.sample_noise(n=len(current_batch_x))
            _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss], feed_dict={self._Z_input: Z_noise})
        return critic_loss_batch, gen_loss_batch



    def predict(self, inpt_x, is_encoded):
        if is_encoded:
            inpt_x = self._generator.generate_samples(noise=inpt_x, sess=self._sess)
        return self._critic.predict(inpt_x, self._sess)







if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train/255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # x_train, y_train = x_train[:2000], y_train[:2000]

    gen_architecture = [
                        [tf.layers.dense, {"units": 128, "activation": tf.nn.relu, "name": "Dense0"}],
                        [tf.layers.dense, {"units": 256, "activation": tf.nn.relu, "name": "Dense1"}],
                        ]
    critic_architecture = [
                        [tf.layers.dense, {"units": 256, "activation": tf.nn.relu, "name": "Dense0"}],
                        ]
    inpt_dim = 784
    z_dim = 64
    wgangp = WGANGP(x_dim=inpt_dim, z_dim=z_dim, last_layer_activation=tf.nn.sigmoid,
                  gen_architecture=gen_architecture, critic_architecture=critic_architecture,
                  folder="../../Results/Test/WGANGP", image_shape=[28, 28])
    wgangp.log_architecture()
    print(wgangp.show_architecture())
    wgangp.compile()
    wgangp.train(x_train, epochs=100, critic_steps=5, gen_steps=1)

