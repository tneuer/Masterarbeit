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
from building_blocks.networks import Critic, ConditionalGenerator, Encoder
from building_blocks.generativeModels import ConditionalGenerativeModel


class CC_CWGAN2(ConditionalGenerativeModel):
    def __init__(self, x_dim, y_dim, z_dim, gen_architecture, critic_architecture, aux_architecture,
                 last_layer_activation, folder="./CC_CWGAN1", image_shape=None, append_y_at_every_layer=None
        ):
        super(CC_CWGAN2, self).__init__(x_dim, y_dim, z_dim, [gen_architecture, critic_architecture, aux_architecture],
                                   last_layer_activation, folder, image_shape, append_y_at_every_layer)

        self._gen_architecture = self._architectures[0]
        self._critic_architecture = self._architectures[1]
        self._aux_architecture = self._architectures[2]

        ################# Define architecture
        self._gen_architecture.append([logged_dense, {"units": x_dim, "activation": self._last_layer_activation, "name": "Output"}])
        self._generator = ConditionalGenerator(self._gen_architecture, name="Generator")

        self._critic_architecture.append([logged_dense, {"units": 1, "activation": tf.identity, "name": "Output"}])
        self._critic = Critic(self._critic_architecture, name="Critic")

        self._aux_architecture.append([logged_dense, {"units": y_dim, "activation": tf.identity, "name": "Output"}])
        self._aux = Encoder(self._aux_architecture, name="Auxiliary")

        self._nets = [self._generator, self._critic, self._aux]

        ################# Connect inputs and networks
        self._output_gen = self._generator.generate_net(self._mod_Z_input,
                                                        append_elements_at_every_layer=self._append_at_every_layer)

        with tf.name_scope("InputsCritic"):
            self._input_fake = tf.concat(axis=1, values=[self._output_gen, self._Y_input], name="fake")
            self._input_real = tf.concat(axis=1, values=[self._X_input, self._Y_input], name="real")

        self._output_critic_real = self._critic.generate_net(self._input_real)
        self._output_critic_fake = self._critic.generate_net(self._input_fake)

        self._output_aux = self._aux.generate_net(self._output_gen)
        self._output_critic_real = self._critic.generate_net(self._input_real)
        self._output_critic_fake = self._critic.generate_net(self._input_fake)

        ################# Finalize
        self._init_folders()
        self._verify_init()


    def compile(self, logged_images=None, logged_labels=None, learning_rate_gen=0.0001, learning_rate_critic=0.0001,
                learning_rate_aux=0.0001, optimizer=tf.train.RMSPropOptimizer):
        self._define_loss()
        with tf.name_scope("Optimizer"):
            gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars(scope="Generator"), name="Generator")
            critic_optimizer = optimizer(learning_rate=learning_rate_critic)
            self._critic_optimizer = critic_optimizer.minimize(self._critic_loss, var_list=self._get_vars(scope="Critic"), name="Critic")
            aux_optimizer = optimizer(learning_rate=learning_rate_aux)
            self._aux_optimizer = aux_optimizer.minimize(self._aux_loss, var_list=self._get_vars(scope="Generator")+self._get_vars(scope="Auxiliary"), name="Auxiliary")
        self._clip_critic_param = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self._get_vars("Critic")]
        self._summarise(logged_images=logged_images, logged_labels=logged_labels)


    def _define_loss(self):
        with tf.name_scope("Loss") as scope:
            self._gen_loss = -tf.reduce_mean(self._output_critic_fake)
            tf.summary.scalar("Generator_loss", self._gen_loss)
            self._critic_loss = -(tf.reduce_mean(self._output_critic_real) - tf.reduce_mean(self._output_critic_fake))
            tf.summary.scalar("Critic_loss", self._critic_loss)
            self._aux_loss = tf.reduce_mean(tf.square(self._Y_input - self._output_aux))
            tf.summary.scalar("Auxiliary_loss", self._aux_loss)


    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=100, batch_size=64, gen_steps=1, critic_steps=5, log_step=3):
        self._set_up_training(log_step=log_step)
        self._set_up_test_train_sample(x_train, y_train, x_test, y_test)
        for epoch in range(epochs):
            batch_nr = 0
            critic_loss_epoch = 0
            gen_loss_epoch = 0
            aux_loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            while trained_examples < len(x_train):
                critic_loss_batch, gen_loss_batch, aux_loss_batch = self._optimize(self._trainset, batch_size, critic_steps, gen_steps)
                trained_examples += batch_size

                critic_loss_epoch += critic_loss_batch
                gen_loss_epoch += gen_loss_batch
                aux_loss_epoch += aux_loss_batch

            epoch_train_time = (time.clock() - start)/60
            critic_loss_epoch = np.round(critic_loss_epoch, 2)
            gen_loss_epoch = np.round(gen_loss_epoch, 2)

            print("Epoch {}: Critic: {} \n\t\t\tGenerator: {}\n\t\t\tAuxiliary: {}.".format(epoch, critic_loss_epoch, gen_loss_epoch, aux_loss_epoch))

            if log_step is not None:
                self._log(epoch, epoch_train_time)


    def _optimize(self, dataset, batch_size, critic_steps, gen_steps):
        for i in range(critic_steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            Z_noise = self.sample_noise(n=len(current_batch_x))
            _, critic_loss_batch, clipping_D = self._sess.run([
                                            self._critic_optimizer, self._critic_loss, self._clip_critic_param
                                            ],
                                            feed_dict={self._X_input: current_batch_x, self._Y_input: current_batch_y, self._Z_input: Z_noise
            })

        for _ in range(gen_steps):
            Z_noise = self._generator.sample_noise(n=len(current_batch_x))
            _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss],
                                               feed_dict={self._Z_input: Z_noise, self._Y_input: current_batch_y})
            _, aux_loss_batch = self._sess.run([self._aux_optimizer, self._aux_loss],
                                               feed_dict={self._Z_input: Z_noise, self._Y_input: current_batch_y})
        return critic_loss_batch, gen_loss_batch, aux_loss_batch


    def predict(self, inpt_x, inpt_y, is_encoded):
        if is_encoded:
            inpt_x = self._sess.run(self._mod_Z_input, feed_dict={self._Z_input: inpt_x, self._Y_input_oneHot: inpt_y})
            inpt_x = self._generator.generate_samples(noise=inpt_x, sess=self._sess)
        inpt = self._sess.run(self._input_real, feed_dict={self._X_input: inpt_x, self._Y_input: inpt_y})
        return self._critic.predict(inpt, self._sess)






if __name__ == '__main__':
    from sklearn.preprocessing import OneHotEncoder
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train/255.
    x_train_log = [x_train[y_train.tolist().index(i)] for i in range(10)]
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # x_train, y_train = x_train[:2000], y_train[:2000]

    nr_classes = 10
    y_train_log = np.identity(nr_classes)
    enc = OneHotEncoder(sparse=False)
    y_train = enc.fit_transform(y_train.reshape(-1, 1))
    y_test = enc.transform(y_test.reshape(-1, 1))

    gen_architecture = [
                        [logged_dense, {"units": 128, "activation": tf.nn.relu, "name": "Dense0"}],
                        [logged_dense, {"units": 256, "activation": tf.nn.relu, "name": "Dense1"}],
                        ]
    critic_architecture = [
                        [logged_dense, {"units": 256, "activation": tf.nn.relu, "name": "Dense0"}],
                        ]
    aux_architecture = [
                    [logged_dense, {"units": 256, "activation": tf.nn.relu, "name": "Dense0"}],
                    [logged_dense, {"units": 128, "activation": tf.nn.relu, "name": "Dense1"}],
                    ]
    inpt_dim = 784
    z_dim = 64
    label_dim = nr_classes
    cc_cwgan = CC_CWGAN2(x_dim=inpt_dim, y_dim=label_dim, z_dim=z_dim, last_layer_activation=tf.nn.sigmoid,
                  gen_architecture=gen_architecture, critic_architecture=critic_architecture, aux_architecture=aux_architecture,
                  folder="../../Results/Test/CC_CWGAN2", image_shape=[28, 28], append_y_at_every_layer=True)
    print(cc_cwgan.show_architecture())
    cc_cwgan.log_architecture()
    cc_cwgan.compile(logged_images=x_train_log, logged_labels=y_train_log)
    cc_cwgan.train(x_train, y_train, x_test, y_test, epochs=100, critic_steps=5, gen_steps=1, log_step=3)
