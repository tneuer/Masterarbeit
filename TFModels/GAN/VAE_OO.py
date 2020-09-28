#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-08-20 16:50:44
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
from building_blocks.networks import Encoder, Decoder
from building_blocks.generativeModels import GenerativeModel

class VAE(GenerativeModel):
    def __init__(self, x_dim, z_dim, dec_architecture, enc_architecture, last_layer_activation,
                 folder="./VAE", image_shape=None
        ):
        super(VAE, self).__init__(x_dim, z_dim, [dec_architecture, enc_architecture],
                                   last_layer_activation, folder, image_shape)

        self._gen_architecture = self._architectures[0]
        self._enc_architecture = self._architectures[1]

        ################# Define architecture
        last_layer_mean = [logged_dense, {"units": z_dim, "activation": tf.identity, "name": "Mean"}]
        self._encoder_mean = Encoder(self._enc_architecture + [last_layer_mean], name="Encoder")

        last_layer_std = [logged_dense, {"units": z_dim, "activation": tf.identity, "name": "Std"}]
        self._encoder_std = Encoder(self._enc_architecture + [last_layer_std], name="Encoder")

        self._gen_architecture.append([logged_dense, {"units": x_dim, "activation": self._last_layer_activation, "name": "Output"}])
        self._decoder = Decoder(self._gen_architecture, name="Generator")

        self._nets = [self._decoder, self._encoder_mean]

        ################# Connect inputs and networks
        self._mean_layer = self._encoder_mean.generate_net(self._X_input)
        self._std_layer = self._encoder_std.generate_net(self._X_input)

        self._output_enc_with_noise = self._mean_layer + tf.exp(0.5*self._std_layer)*self._Z_input

        self._output_dec = self._decoder.generate_net(self._output_enc_with_noise)
        self._output_dec_from_encoding = self._decoder.generate_net(self._Z_input)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, learning_rate=0.0001, optimizer=tf.train.AdamOptimizer):
        self._define_loss()
        with tf.name_scope("Optimizer"):
            vae_optimizer = optimizer(learning_rate=learning_rate)
            self._vae_optimizer = vae_optimizer.minimize(self._vae_loss, name="VAE")
        self._summarise()


    def _define_loss(self):
        with tf.name_scope("Loss") as scope:
            self._data_fidelity_loss = self._X_input*tf.log(1e-10 + self._output_dec) + (1 - self._X_input)*tf.log(1e-10 + 1 - self._output_dec)
            self._data_fidelity_loss = -tf.reduce_sum(self._data_fidelity_loss, 1)

            self._KLdiv = 0.5*(tf.square(self._mean_layer) + tf.exp(self._std_layer) - self._std_layer - 1)
            self._KLdiv = tf.reduce_sum(self._KLdiv, 1)

            self._vae_loss = tf.reduce_mean(self._data_fidelity_loss + self._KLdiv)
            tf.summary.scalar("Loss", self._vae_loss)



    def train(self, x_train, x_test, epochs=100, batch_size=64, steps=5, log_step=3):
        self._set_up_training(log_step=log_step)
        self._set_up_test_train_sample(x_train, x_test)
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
                self._log(epoch, epoch_train_time)


    def _optimize(self, dataset, batch_size, steps):
        for i in range(steps):
            current_batch_x = dataset.get_next_batch(batch_size)
            Z_noise = self._decoder.sample_noise(n=len(current_batch_x))
            _, loss_batch = self._sess.run([self._vae_optimizer, self._vae_loss], feed_dict={self._X_input: current_batch_x, self._Z_input: Z_noise})
        return loss_batch


    def decode(self, inpt_x, is_encoded):
        if not is_encoded:
            inpt_x = self._encoder_mean.encode(noise=inpt_x, sess=self._sess)
        return self._decoder.decode(inpt_x, self._sess)


    def generate_image_from_noise(self, n):
        noise = self.sample_noise(n=n)
        return self._decoder.generate_samples(noise, self._sess)


    def generate_image_from_image(self, inpt_x):
        encoding = self._encoder.encode(inpt=inpt_x, sess=self._sess)
        return self._decoder.generate_samples(encoding, self._sess)





if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train/255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # x_train, y_train = x_train[:2000], y_train[:2000]

    enc_architecture = [
                        [logged_dense, {"units": 256, "activation": tf.nn.relu, "name": "Dense0"}],
                        ]
    dec_architecture = [
                        [logged_dense, {"units": 256, "activation": tf.nn.relu, "name": "Dense1"}],
                        ]
    inpt_dim = 784
    z_dim = 64
    vae = VAE(x_dim=inpt_dim, z_dim=z_dim, last_layer_activation=tf.nn.sigmoid,
                  dec_architecture=dec_architecture, enc_architecture=enc_architecture,
                  folder="../../Results/Test/VAE", image_shape=[28, 28])
    vae.log_architecture()
    print(vae.show_architecture())
    vae.compile()
    vae.train(x_train, x_test, epochs=100, steps=3)

