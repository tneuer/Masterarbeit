#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-08-28 22:45:25
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

from building_blocks.layers import logged_dense, reshape_layer, image_condition_concat
from building_blocks.networks import Critic, ConditionalGenerator
from building_blocks.generativeModels import ConditionalGenerativeModel
from functionsOnImages import padding_zeros


class CWGANGP(ConditionalGenerativeModel):
    def __init__(self, x_dim, y_dim, z_dim, gen_architecture, critic_architecture, last_layer_activation,
                 folder="./CWGANGP", image_shape=None, append_y_at_every_layer=None, PatchGAN=False
        ):
        super(CWGANGP, self).__init__(x_dim, y_dim, z_dim, [gen_architecture, critic_architecture],
                                   last_layer_activation, folder, image_shape, append_y_at_every_layer)

        self._gen_architecture = self._architectures[0]
        self._critic_architecture = self._architectures[1]
        self._is_patchgan = PatchGAN

        ################# Define architecture
        if self._is_patchgan:
            f_xy = self._critic_architecture[-1][-1]["filters"]
            assert f_xy == 1, "If is PatchGAN, last layer of Discriminator_XY needs 1 filter. Given: {}.".format(f_xy)

            a_xy = self._critic_architecture[-1][-1]["activation"]
            assert a_xy == tf.identity, "If is PatchGAN, last layer of Discriminator_XY needs tf.nn.sigmoid. Given: {}.".format(a_xy)
        else:
            self._critic_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            self._critic_architecture.append([logged_dense, {"units": 1, "activation": tf.identity, "name": "Output"}])

        if len(self._x_dim) == 1:
            self._gen_architecture.append([tf.layers.dense, {"units": x_dim, "activation": self._last_layer_activation, "name": "Output"}])
        else:
            self._gen_architecture[-1][1]["name"] = "Output"

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

        # with tf.name_scope("InputsCritic"):
        #     self._input_fake = self._output_gen
        #     self._input_real = self._X_input

        self._output_critic_real = self._critic.generate_net(self._input_real, tf_trainflag=self._is_training)
        self._output_critic_fake = self._critic.generate_net(self._input_fake, tf_trainflag=self._is_training)

        ################# Finalize
        self._init_folders()
        self._verify_init()

        if self._is_patchgan:
            print("PATCHGAN chosen with output: {}.".format(self._output_critic_real.shape))


    def compile(self, logged_images=None, logged_labels=None, learning_rate=0.0005,
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

            self._gen_grads_and_vars = gen_optimizer.compute_gradients(self._gen_loss)
            self._critic_grads_and_vars = critic_optimizer.compute_gradients(self._critic_loss)
        self._summarise(logged_images=logged_images, logged_labels=logged_labels)


    def _define_loss(self):
        self._gen_loss = -tf.reduce_mean(self._output_critic_fake)
        self._critic_loss_original = -(tf.reduce_mean(self._output_critic_real) - tf.reduce_mean(self._output_critic_fake))
        self._critic_loss = self._critic_loss_original + 10*self._define_gradient_penalty()
        with tf.name_scope("Loss") as scope:
            tf.summary.scalar("Critic_loss_original", self._critic_loss_original)
            tf.summary.scalar("Generator_loss", self._gen_loss)
            tf.summary.scalar("Critic_loss_penalized", self._critic_loss)


    def _define_gradient_penalty(self):
        alpha = tf.random_uniform(shape=tf.shape(self._input_real), minval=0., maxval=1.)
        differences = self._input_fake - self._input_real
        interpolates = self._input_real + (alpha * differences)
        gradients = tf.gradients(self._critic.generate_net(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        with tf.name_scope("Loss") as scope:
            self._gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            tf.summary.scalar("Gradient_penalty", self._gradient_penalty)
        return self._gradient_penalty


    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=100, batch_size=64,
              gen_steps=1, critic_steps=5, log_step=3, batch_log_step=None, steps=None, gpu_options=None):
        if steps is not None:
            gen_steps = 1
            critic_steps = steps
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        self._set_up_test_train_sample(x_train, y_train, x_test, y_test)
        self._log_results(epoch=0, epoch_time=0)
        nr_batches = np.floor(len(x_train) / batch_size)

        self._dominating_disc = 0
        self._gen_out_zero = 0
        for epoch in range(epochs):
            batch_nr = 0
            critic_loss_epoch = 0
            gen_loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            ii = 0

            while trained_examples < len(x_train):
                critic_loss_batch, gen_loss_batch = self._optimize(self._trainset, batch_size, critic_steps, gen_steps)
                trained_examples += batch_size

                if np.isnan(critic_loss_batch) or np.isnan(gen_loss_batch):
                    print("DiscLoss / GenLoss: ",  critic_loss_batch, gen_loss_batch)
                    self._check_tf_variables(ii, nr_batches)
                    raise

                if (batch_log_step is not None) and (ii % batch_log_step == 0):
                    batch_train_time = (time.clock() - start)/60
                    self._log(int(epoch*nr_batches+ii), batch_train_time)

                # if ii % 100 == 0:
                #     print("DiscLoss / GenLoss: ",  critic_loss_batch, gen_loss_batch)
                #     self._check_tf_variables(ii, nr_batches)

                critic_loss_epoch += critic_loss_batch
                gen_loss_epoch += gen_loss_batch
                ii += 1


            epoch_train_time = (time.clock() - start)/60
            critic_loss_epoch = np.round(critic_loss_epoch, 2)
            gen_loss_epoch = np.round(gen_loss_epoch, 2)

            print("Epoch {}: Critic: {} \n\t\t\tGenerator: {}.".format(epoch+1, critic_loss_epoch, gen_loss_epoch))

            if self._log_step is not None:
                self._log(epoch+1, epoch_train_time)

            self._check_tf_variables(epoch, epochs)


    def _optimize(self, dataset, batch_size, critic_steps, gen_steps):
        for i in range(critic_steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            self._Z_noise = self.sample_noise(n=len(current_batch_x))
            _, critic_loss_batch = self._sess.run([
                                            self._critic_optimizer, self._critic_loss
                                            ],
                                            feed_dict={self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                            self._Z_input: self._Z_noise, self._is_training: True
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


    def _check_tf_variables(self, batch_nr, nr_batches):
        Z_noise = self._generator.sample_noise(n=len(self._x_test))
        gen_grads = [self._sess.run(gen_gv[0], feed_dict={self._X_input: self._x_test,
                                    self._Y_input: self._y_test, self._Z_input: Z_noise, self._is_training: False})

                                for gen_gv in self._gen_grads_and_vars]
        disc_grads = [self._sess.run(disc_gv[0], feed_dict={self._X_input: self._x_test,
                                    self._Y_input: self._y_test, self._Z_input: Z_noise, self._is_training: False})

                                for disc_gv in self._critic_grads_and_vars]
        gen_grads_maxis = [np.max(gv) for gv in gen_grads]
        gen_grads_means = [np.mean(gv) for gv in gen_grads]
        gen_grads_minis = [np.min(gv) for gv in gen_grads]
        disc_grads_maxis = [np.max(dv) for dv in disc_grads]
        disc_grads_means = [np.mean(dv) for dv in disc_grads]
        disc_grads_minis = [np.min(dv) for dv in disc_grads]

        real_logits, fake_logits, gen_out = self._sess.run(
                [self._output_critic_real, self._output_critic_fake, self._output_gen],
                feed_dict={self._X_input: self._x_test, self._Y_input: self._y_test,
                            self._Z_input: Z_noise, self._is_training: False})
        real_logits = np.mean(real_logits)
        fake_logits = np.mean(fake_logits)

        gen_varsis = np.array([x.eval(session=self._sess) for x in self._generator.get_network_params()])
        disc_varsis = np.array([x.eval(session=self._sess) for x in self._critic.get_network_params()])
        gen_maxis = np.array([np.max(x) for x in gen_varsis])
        disc_maxis = np.array([np.max(x) for x in disc_varsis])
        gen_means = np.array([np.mean(x) for x in gen_varsis])
        disc_means = np.array([np.mean(x) for x in disc_varsis])
        gen_minis = np.array([np.min(x) for x in gen_varsis])
        disc_minis = np.array([np.min(x) for x in disc_varsis])

        print(batch_nr, "/", nr_batches, ":")
        print("DiscReal / DiscFake: ",  real_logits, fake_logits)
        print("GenWeight Max / Mean / Min: ",  np.max(gen_maxis), np.mean(gen_means), np.min(gen_minis))
        print("GenGrads Max / Mean / Min: ",  np.max(gen_grads_maxis), np.mean(gen_grads_means), np.min(gen_grads_minis))
        print("DiscWeight Max / Mean / Min: ",  np.max(disc_maxis), np.mean(disc_means), np.min(disc_minis))
        print("DiscGrads Max / Mean / Min: ",  np.max(disc_grads_maxis), np.mean(disc_grads_means), np.min(disc_grads_minis))
        print("GenOut Max / Mean / Min: ",  np.max(gen_out), np.mean(gen_out), np.min(gen_out))
        print("\n")

        if real_logits > 0.99 and fake_logits < 0.01:
            self._dominating_disc += 1
            if self._dominating_disc == 5:
                raise SystemError("Dominating discriminator!")
        else:
            self._dominating_disc = 0

        print(np.max(gen_out))
        print(np.max(gen_out) < 0.05)
        if np.max(gen_out) < 0.05:
            self._gen_out_zero += 1
            print(self._gen_out_zero)
            if self._gen_out_zero == 5:
                raise SystemError("Generator outputs zeros")
        else:
            self._gen_out_zero = 0
        print(self._gen_out_zero)


if __name__ == '__main__':
    if "lhcb_data2" in os.getcwd():
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        gpu_frac = 0.3
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
        print("1 GPU limited to {}% memory.".format(np.round(gpu_frac*100)))
    else:
        gpu_options = None

    from sklearn.preprocessing import OneHotEncoder
    nr_examples = 60000
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train = x_train[:nr_examples], y_train[:nr_examples]
    x_test, y_test = x_train[:500], y_train[:500]
    x_train, x_test = x_train/255., x_test/255.

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
    #                     [tf.layers.dense, {"units": 256, "activation": tf.nn.relu}],
    #                     [tf.layers.dense, {"units": 512, "activation": tf.nn.relu}],
    #                     ]
    # critic_architecture = [
    #                     [tf.layers.dense, {"units": 512, "activation": tf.nn.relu}],
    #                     [tf.layers.dense, {"units": 256, "activation": tf.nn.relu}],
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
                        [tf.layers.dense, {"units": 4*4*512, "activation": tf.nn.relu}],
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
                        [tf.layers.conv2d, {"filters": 1, "kernel_size": 5, "strides": 1, "padding": "same", "activation": tf.nn.relu}],
                        ]
    inpt_dim = x_train[0].shape
    image_shape=[32, 32, 1]

    z_dim = 128
    label_dim = 10
    gpu_options = None
    is_patchGAN = True
    max_iterations = 10

    if is_patchGAN:
        critic_architecture[-1][1]["activation"] = tf.identity

    fails = ""
    run_nr = 1
    continue_training = True
    while continue_training and run_nr <= max_iterations:
        try:
            cwgangp = CWGANGP(x_dim=inpt_dim, y_dim=label_dim, z_dim=z_dim, last_layer_activation=tf.nn.sigmoid,
                          gen_architecture=gen_architecture, critic_architecture=critic_architecture,
                          folder="../../Results/Test/CWGANGP", image_shape=image_shape, append_y_at_every_layer=False,
                          PatchGAN=is_patchGAN)
            print(cwgangp.show_architecture())
            cwgangp.log_architecture()
            cwgangp.compile(logged_images=x_train_log, logged_labels=y_train_log)
            cwgangp.train(x_train, y_train, x_test, y_test, epochs=100, critic_steps=5, gen_steps=1, log_step=3,
                          batch_log_step=2000, gpu_options=gpu_options)
            continue_training = False
            with open(save_folder+"/FAILS.txt", "w") as f:
                fails += "{})\t SUCCESS!\n".format(str(run_nr))
                f.write(fails)
        except SystemError as e:
            if not run_until_success:
                break
            print("!!!!!!!!!!!!RESTARTNG ALGORITHM DUE TO FAILURE DURING TRAINING!!!!!!!!!!!!")
            tf.reset_default_graph()
            import shutil
            shutil.rmtree(save_folder+"/GeneratedSamples")
            shutil.rmtree(save_folder+"/TFGraphs")
            os.mkdir(save_folder+"/GeneratedSamples")
            os.mkdir(save_folder+"/TFGraphs")
            with open(save_folder+"/FAILS.txt", "w") as f:
                fails += "{})\t {}.\n".format(str(run_nr), str(e))
                f.write(fails)
            run_nr += 1

