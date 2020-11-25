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

from building_blocks.layers import logged_dense
from building_blocks.networks import Generator, Discriminator
from building_blocks.generativeModels import GenerativeModel

class VanillaGAN(GenerativeModel):
    def __init__(self, x_dim, z_dim, gen_architecture, disc_architecture, folder="./VanillaGan"
        ):
        super(VanillaGAN, self).__init__(x_dim, z_dim, [gen_architecture, disc_architecture], folder)

        self._gen_architecture = self._architectures[0]
        self._gen_architecture[-1][1]["name"] = "Output"
        self._disc_architecture = self._architectures[1]

        ################# Define architecture
        self._generator = Generator(self._gen_architecture, name="Generator")

        self._disc_architecture.append([logged_dense, {"units": 1, "activation": tf.nn.sigmoid, "name": "Output"}])
        self._discriminator = Discriminator(self._disc_architecture, name="Discriminator")

        self._nets = [self._generator, self._discriminator]

        ################# Connect inputs and networks
        self._output_gen = self._generator.generate_net(self._Z_input)
        self._output_disc_fake = self._discriminator.generate_net(self._output_gen)
        self._output_disc_real = self._discriminator.generate_net(self._X_input)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, learning_rate_gen=0.0002, learning_rate_disc=0.0002, optimizer=tf.train.AdamOptimizer):
        self._define_loss()
        with tf.name_scope("Optimizer"):
            gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars(scope="Generator"), name="Generator")
            disc_optimizer = optimizer(learning_rate=learning_rate_disc)
            self._disc_optimizer = disc_optimizer.minimize(self._disc_loss, var_list=self._get_vars(scope="Discriminator"), name="Discriminator")
        self._summarise()


    def _define_loss(self):
        with tf.name_scope("Loss") as scope:
            self._gen_loss = -tf.reduce_mean(tf.log(self._output_disc_fake+0.0001))
            tf.summary.scalar("Generator_loss", self._gen_loss)
            self._disc_loss = -tf.reduce_mean(tf.log(self._output_disc_real+0.0001) + tf.log(1.0-self._output_disc_fake+0.0001))
            tf.summary.scalar("Discriminator_loss", self._disc_loss)


    def train(self, x_train, x_test=None, epochs=100, batch_size=64, gen_steps=1, disc_steps=1, log_step=3):
        self._set_up_training(log_step=log_step)
        self._set_up_test_train_sample(x_train, x_test)
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

            self._real_images_predicted_real = self.get_accuracy(self._x_test, np.ones(self._x_test.shape[0]), is_encoded=False)
            self._fake_images_predicted_real = self.get_accuracy(self._z_test, np.ones(self._x_test.shape[0]), is_encoded=True)
            print("Epoch {}: Discriminator: {} ({})\n\t\t\tGenerator: {} ({}).".format(epoch, disc_loss_epoch,
                                                                                       self._real_images_predicted_real, gen_loss_epoch,
                                                                                       self._fake_images_predicted_real))

            if log_step is not None:
                self._log(epoch, epoch_train_time)


    def _optimize(self, dataset, batch_size, disc_steps, gen_steps):
        for i in range(disc_steps):
            current_batch_x = dataset.get_next_batch(batch_size)
            Z_noise = self.sample_noise(n=len(current_batch_x))
            _, disc_loss_batch = self._sess.run([self._disc_optimizer, self._disc_loss], feed_dict={self._X_input: current_batch_x, self._Z_input: Z_noise})

        for _ in range(gen_steps):
            Z_noise = self.sample_noise(n=len(current_batch_x))
            _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss], feed_dict={self._Z_input: Z_noise})
        return disc_loss_batch, gen_loss_batch


    def get_accuracy(self, inpt_x, labels, is_encoded):
        if is_encoded:
            inpt_x = self.generate_samples(inpt=inpt_x)
        return self._discriminator.get_accuracy(inpt_x, labels, self._sess)


    def predict(self, inpt_x, is_encoded):
        if is_encoded:
            inpt_x = self.generate_samples(inpt=inpt_x)
        return self._discriminator.predict(inpt_x, self._sess)




if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train/255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # x_train, y_train = x_train[:2000], y_train[:2000]

    gen_architecture = [
                        [logged_dense, {"units": 128, "activation": tf.nn.relu, "name": "Dense0"}],
                        [logged_dense, {"units": 256, "activation": tf.nn.relu, "name": "Dense1"}]
                        ]
    disc_architecture = [
                        [logged_dense, {"units": 128, "activation": tf.nn.relu, "name": "Dense0"}],
                        [logged_dense, {"units": 256, "activation": tf.nn.relu, "name": "Dense1"}]
                        ]
    inpt_dim = 784
    z_dim = 64
    vanillgan = VanillaGAN(x_dim=inpt_dim, z_dim=z_dim, last_layer_activation=tf.nn.sigmoid,
                  gen_architecture=gen_architecture, disc_architecture=disc_architecture,
                  folder="../../Results/Test/VanillaGAN", image_shape=[28, 28])
    vanillgan.log_architecture()
    print(vanillgan.show_architecture())
    vanillgan.compile()
    vanillgan.train(x_train, epochs=100, disc_steps=1, gen_steps=1)


