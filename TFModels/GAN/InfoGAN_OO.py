#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-10-11 12:46:32
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
from building_blocks.networks import ConditionalGenerator, Discriminator, Encoder
from building_blocks.generativeModels import ConditionalGenerativeModel
from Dataset import Dataset

class InfoGAN(ConditionalGenerativeModel):
    def __init__(self, x_dim, y_dim, z_dim, gen_architecture, disc_architecture, aux_architecture, last_layer_activation,
                 folder="./CWGAN", image_shape=None
        ):
        super(InfoGAN, self).__init__(x_dim, y_dim, z_dim, [gen_architecture, disc_architecture, aux_architecture],
                                   last_layer_activation, folder, image_shape, None)

        self._gen_architecture = self._architectures[0]
        self._disc_architecture = self._architectures[1]
        self._aux_architecture = self._architectures[2]

        ################# Define architecture
        self._gen_architecture.append([logged_dense, {"units": x_dim, "activation": tf.nn.sigmoid, "name": "Output"}])
        self._generator = ConditionalGenerator(self._gen_architecture, name="Generator")

        self._disc_architecture.append([logged_dense, {"units": 1, "activation": tf.nn.sigmoid, "name": "Output"}])
        self._disc = Discriminator(self._disc_architecture, name="Discriminator")

        self._aux_architecture.append([logged_dense, {"units": y_dim, "activation": tf.nn.softmax, "name": "Output"}])
        self._aux = Encoder(self._aux_architecture, name="Auxiliary")

        self._nets = [self._generator, self._disc, self._aux]

        ################# Connect inputs and networks
        self._output_gen = self._generator.generate_net(self._mod_Z_input)
        self._output_disc_fake = self._disc.generate_net(self._output_gen)
        self._output_disc_real = self._disc.generate_net(self._X_input)
        self._output_aux = self._aux.generate_net(self._output_gen)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, learning_rate=0.0003, learning_rate_gen=None, learning_rate_disc=None, learning_rate_aux=None, optimizer=tf.train.RMSPropOptimizer):
        self._define_loss()
        if learning_rate_gen is None:
            learning_rate_gen = learning_rate
        if learning_rate_disc is None:
            learning_rate_disc = learning_rate
        if learning_rate_aux is None:
            learning_rate_aux = learning_rate
        with tf.name_scope("Optimizer"):
            gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars(scope="Generator"), name="Generator")
            disc_optimizer = optimizer(learning_rate=learning_rate_disc)
            self._disc_optimizer = disc_optimizer.minimize(self._disc_loss, var_list=self._get_vars(scope="Discriminator"), name="Discriminator")
            aux_optimizer = optimizer(learning_rate=learning_rate_aux)
            self._aux_optimizer = aux_optimizer.minimize(self._aux_loss, var_list=self._get_vars("Auxiliary")+self._get_vars("Generator"), name="Auxiliary")
        self._summarise(logged_labels=np.identity(self._y_dim))


    def _define_loss(self):
        with tf.name_scope("Loss") as scope:
            self._gen_loss = -tf.reduce_mean(tf.log(self._output_disc_fake+0.00001))
            tf.summary.scalar("Generator_loss", self._gen_loss)
            self._disc_loss  = -tf.reduce_mean(tf.log(self._output_disc_real+0.00001) + tf.log(1.0-self._output_disc_fake+0.00001))
            tf.summary.scalar("Discriminator_loss", self._disc_loss)
            self._aux_loss = -tf.reduce_mean(-tf.reduce_sum(tf.log(self._output_aux+0.00001)*self._Y_input, 1))
            tf.summary.scalar("Auxiliary_loss", self._aux_loss)


    def train(self, x_train, epochs=100, batch_size=64, gen_steps=1, disc_steps=1, log_step=3):
        self._set_up_training(log_step=log_step)
        self._trainset = Dataset(x_train)
        nr_test_samples = 5000 if len(x_train) >=5000 else len(x_train)
        self._x_test = self._trainset.sample(nr_test_samples)
        self._y_test = self.sample_condition(n=len(self._x_test))
        self._z_test = self._generator.sample_noise(n=nr_test_samples)
        for epoch in range(epochs):
            batch_nr = 0
            disc_loss_epoch = 0
            gen_loss_epoch = 0
            aux_loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            while trained_examples < len(x_train):
                disc_loss_batch, gen_loss_batch, aux_loss_batch = self._optimize(self._trainset, batch_size, disc_steps, gen_steps)
                trained_examples += batch_size

                disc_loss_epoch += disc_loss_batch
                gen_loss_epoch += gen_loss_batch
                aux_loss_epoch += aux_loss_batch

            epoch_train_time = (time.clock() - start)/60
            disc_loss_epoch = np.round(disc_loss_epoch, 2)
            gen_loss_epoch = np.round(gen_loss_epoch, 2)
            aux_loss_epoch = np.round(aux_loss_epoch, 2)

            print("Epoch {}: Discriminator: {}\n\t\t\tGenerator: {}\n\t\t\tAuxiliary: {}.".format(epoch, disc_loss_epoch, gen_loss_epoch, aux_loss_epoch))

            if log_step is not None:
                self._log(epoch, epoch_train_time)


    def _optimize(self, dataset, batch_size, disc_steps, gen_steps):
        for i in range(disc_steps):
            current_batch_x = dataset.get_next_batch(batch_size)
            Z_noise = self.sample_noise(n=len(current_batch_x))
            C_noise = self.sample_condition(n=len(current_batch_x))
            latent_space = np.concatenate((Z_noise, C_noise), axis=1)
            _, disc_loss_batch = self._sess.run([self._disc_optimizer, self._disc_loss],
                                                feed_dict={self._X_input: current_batch_x, self._Y_input: C_noise, self._Z_input: Z_noise
            })

        for _ in range(gen_steps):
            Z_noise = self.sample_noise(n=len(current_batch_x))
            C_noise = self.sample_condition(n=len(current_batch_x))
            latent_space = np.concatenate((Z_noise, C_noise), axis=1)
            _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss], feed_dict={self._Y_input: C_noise, self._Z_input: Z_noise})
            _, aux_loss_batch = self._sess.run([self._aux_optimizer, self._aux_loss], feed_dict={self._Y_input: C_noise, self._Z_input: Z_noise})
        return disc_loss_batch, gen_loss_batch, aux_loss_batch


    def sample_condition(self, n):
        return np.random.multinomial(1, self._y_dim*[1/self._y_dim], size=n)


    def predict(self, inpt_x, is_encoded, inpt_c=None):
        if is_encoded:
            if inpt_c is None:
                raise ValueError("If input is encoded, the conditional input is also needed")
            else:
                input_x = np.concatenate((inpt_x, inpt_c), axis=1)
            inpt_x = self._generator.generate_samples(noise=inpt_x, sess=self._sess)
        return self._disc.predict(inpt_x, self._sess)




if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train/255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # x_train, y_train = x_train[:2000], y_train[:2000]

    gen_architecture = [
                        [logged_dense, {"units": 256, "activation": tf.nn.relu}],
                        ]
    disc_architecture = [
                        [logged_dense, {"units": 128, "activation": tf.nn.relu}]
                        ]
    aux_architecture = [
                        [logged_dense, {"units": 128, "activation": tf.nn.relu}]
                        ]
    inpt_dim = 784
    z_dim = 64
    y_dim = 10
    infogan = InfoGAN(x_dim=inpt_dim, y_dim=y_dim, z_dim=z_dim, last_layer_activation=tf.nn.sigmoid,
                  gen_architecture=gen_architecture, disc_architecture=disc_architecture, aux_architecture=aux_architecture,
                  folder="../../Results/Test/InfoGAN", image_shape=[28, 28])
    infogan.log_architecture()
    print(infogan.show_architecture())
    infogan.compile()
    infogan.train(x_train, epochs=100, gen_steps=1, disc_steps=1)
