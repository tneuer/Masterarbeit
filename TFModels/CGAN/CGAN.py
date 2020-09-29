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
sys.path.insert(1, "../building_blocks")
sys.path.insert(1, "../../Utilities")
import time

import numpy as np
import tensorflow as tf

from layers import logged_dense, reshape_layer, image_condition_concat
from networks import Critic, ConditionalGenerator, Encoder
from generativeModels import ConditionalGenerativeModel
from functionsOnImages import padding_zeros


class CGAN(ConditionalGenerativeModel):
    def __init__(self, x_dim, y_dim, z_dim, gen_architecture, adversarial_architecture, folder="./CGAN",
                 append_y_at_every_layer=None, is_patchgan=False, is_wasserstein=False, aux_architecture=None,
        ):
        architectures = [gen_architecture, adversarial_architecture]
        self._is_cycle_consistent = False
        if aux_architecture is not None:
            architectures.append(aux_architecture)
            self._is_cycle_consistent = True
        super(CGAN, self).__init__(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, architectures=architectures,
                                   folder=folder, append_y_at_every_layer=append_y_at_every_layer)

        self._gen_architecture = self._architectures[0]
        self._adversarial_architecture = self._architectures[1]
        self._is_patchgan = is_patchgan
        self._is_wasserstein = is_wasserstein
        self._is_feature_matching = False

        ################# Define architecture
        if self._is_patchgan:
            f_xy = self._adversarial_architecture[-1][-1]["filters"]
            assert f_xy == 1, "If is PatchGAN, last layer of adversarial_XY needs 1 filter. Given: {}.".format(f_xy)

            a_xy = self._adversarial_architecture[-1][-1]["activation"]
            if self._is_wasserstein:
                assert a_xy == tf.identity, "If is PatchGAN, last layer of adversarial needs tf.identity. Given: {}.".format(a_xy)
            else:
                assert a_xy == tf.nn.sigmoid, "If is PatchGAN, last layer of adversarial needs tf.nn.sigmoid. Given: {}.".format(a_xy)
        else:
            self._adversarial_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            if self._is_wasserstein:
                self._adversarial_architecture.append([logged_dense, {"units": 1, "activation": tf.identity, "name": "Output"}])
            else:
                self._adversarial_architecture.append([logged_dense, {"units": 1, "activation": tf.nn.sigmoid, "name": "Output"}])
        self._gen_architecture[-1][1]["name"] = "Output"

        self._generator = ConditionalGenerator(self._gen_architecture, name="Generator")
        self._adversarial = Critic(self._adversarial_architecture, name="Adversarial")

        self._nets = [self._generator, self._adversarial]

        ################# Connect inputs and networks
        self._output_gen = self._generator.generate_net(self._mod_Z_input,
                                                        append_elements_at_every_layer=self._append_at_every_layer,
                                                        tf_trainflag=self._is_training)

        with tf.name_scope("InputsAdversarial"):
            if len(self._x_dim) == 1:
                self._input_real = tf.concat(axis=1, values=[self._X_input, self._Y_input], name="real")
                self._input_fake = tf.concat(axis=1, values=[self._output_gen, self._Y_input], name="fake")
            else:
                self._input_real = image_condition_concat(inputs=self._X_input, condition=self._Y_input, name="real")
                self._input_fake = image_condition_concat(inputs=self._output_gen, condition=self._Y_input, name="fake")

        self._output_adversarial_real = self._adversarial.generate_net(self._input_real, tf_trainflag=self._is_training)
        self._output_adversarial_fake = self._adversarial.generate_net(self._input_fake, tf_trainflag=self._is_training)

        assert self._output_gen.get_shape()[1:] == x_dim, (
            "Output of generator is {}, but x_dim is {}.".format(self._output_gen.get_shape(), x_dim)
        )

        ################# Auxiliary network for cycle consistency
        if self._is_cycle_consistent:
            self._auxiliary = Encoder(self._architectures[2], name="Auxiliary")
            self._output_auxiliary = self._auxiliary.generate_net(self._output_gen, tf_trainflag=self._is_training)
            assert self._output_auxiliary.get_shape().as_list() == self._mod_Z_input.get_shape().as_list(), (
                "Wrong shape for auxiliary vs. mod Z: {} vs {}.".format(self._output_auxiliary.get_shape(), self._mod_Z_input.get_shape())
            )
            self._nets.append(self._auxiliary)


        ################# Finalize
        self._init_folders()
        self._verify_init()

        if self._is_patchgan:
            print("PATCHGAN chosen with output: {}.".format(self._output_adversarial_real.shape))


    def compile(self, loss, logged_images=None, logged_labels=None, learning_rate=0.0005,
                learning_rate_gen=None, learning_rate_adversarial=None, optimizer=tf.train.RMSPropOptimizer,
                feature_matching=False, label_smoothing=1):
        if self._is_wasserstein and loss != "wasserstein":
            raise ValueError("If is_wasserstein is true in Constructor, loss needs to be wasserstein.")
        if not self._is_wasserstein and loss == "wasserstein":
            raise ValueError("If loss is wasserstein, is_wasserstein needs to be true in constructor.")

        if learning_rate_gen is None:
            learning_rate_gen = learning_rate
        if learning_rate_adversarial is None:
            learning_rate_adversarial = learning_rate
        self._define_loss(loss, feature_matching, label_smoothing)
        with tf.name_scope("Optimizer"):
            gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars("Generator"), name="Generator")
            adversarial_optimizer = optimizer(learning_rate=learning_rate_adversarial)
            self._adversarial_optimizer = adversarial_optimizer.minimize(self._adversarial_loss,
                                                                         var_list=self._get_vars("Adversarial"), name="Adversarial")

            if self._is_cycle_consistent:
                aux_optimizer = optimizer(learning_rate=learning_rate_gen)
                self._aux_optimizer = aux_optimizer.minimize(self._aux_loss,
                    var_list=self._get_vars(scope="Generator")+self._get_vars(scope="Auxiliary"), name="Auxiliary"
                )

            self._gen_grads_and_vars = gen_optimizer.compute_gradients(self._gen_loss)
            self._adversarial_grads_and_vars = adversarial_optimizer.compute_gradients(self._adversarial_loss)
        self._summarise(logged_images=logged_images, logged_labels=logged_labels)


    def _define_loss(self, loss, feature_matching, label_smoothing):
        possible_losses = ["cross-entropy", "L1", "L2", "wasserstein", "KL"]
        def get_labels_one(tensor):
            return tf.ones_like(tensor)*label_smoothing
        eps = 1e-7
        if loss == "cross-entropy":
            self._logits_real = tf.math.log( self._output_adversarial_real / (1+eps - self._output_adversarial_real) + eps)
            self._logits_fake = tf.math.log( self._output_adversarial_fake / (1+eps - self._output_adversarial_fake) + eps)

            self._gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_one(self._logits_fake), logits=self._logits_fake
            ))
            self._adversarial_loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_one(self._logits_real), logits=self._logits_real
                                    ) +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.zeros_like(self._logits_fake), logits=self._logits_fake
                                    )
            )

        elif loss == "L1":
            self._gen_loss = tf.reduce_mean(tf.abs(self._output_adversarial_fake - get_labels_one(self._output_adversarial_fake)))
            self._adversarial_loss = (
                        tf.reduce_mean(
                                        tf.abs(self._output_adversarial_real - get_labels_one(self._output_adversarial_real)) +
                                        tf.abs(self._output_adversarial_fake)
                        )
            ) / 2.0

        elif loss == "L2":
            self._gen_loss = tf.reduce_mean(tf.square(self._output_adversarial_fake - get_labels_one(self._output_adversarial_fake)))
            self._adversarial_loss = (
                        tf.reduce_mean(
                                        tf.square(self._output_adversarial_real - get_labels_one(self._output_adversarial_real)) +
                                        tf.square(self._output_adversarial_fake)
                        )
            ) / 2.0
        elif loss == "wasserstein":
            self._gen_loss = -tf.reduce_mean(self._output_adversarial_fake)
            self._adversarial_loss = (
                -(tf.reduce_mean(self._output_adversarial_real) - tf.reduce_mean(self._output_adversarial_fake)) +
                10*self._define_gradient_penalty()
            )
        elif loss == "KL":
            self._logits_real = tf.math.log( self._output_adversarial_real / (1+eps - self._output_adversarial_real) + eps)
            self._logits_fake = tf.math.log( self._output_adversarial_fake / (1+eps - self._output_adversarial_fake) + eps)

            self._gen_loss = -tf.reduce_mean(self._logits_fake)
            self._adversarial_loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_one(self._logits_real), logits=self._logits_real
                                    ) +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.zeros_like(self._logits_fake), logits=self._logits_fake
                                    )
            )
        else:
            raise ValueError("Loss not implemented. Choose from {}. Given: {}.".format(possible_losses, loss))

        if feature_matching:
            self._is_feature_matching = True
            otp_adv_real = self._adversarial.generate_net(self._input_real, tf_trainflag=self._is_training, return_idx=-2)
            otp_adv_fake = self._adversarial.generate_net(self._input_fake, tf_trainflag=self._is_training, return_idx=-2)
            self._gen_loss = tf.reduce_mean(tf.square(otp_adv_real - otp_adv_fake))

        if self._is_cycle_consistent:
            self._aux_loss = tf.reduce_mean(tf.abs(self._mod_Z_input - self._output_auxiliary))
            self._gen_loss += self._aux_loss

        with tf.name_scope("Loss") as scope:
            tf.summary.scalar("Generator_Loss", self._gen_loss)
            tf.summary.scalar("Adversarial_Loss", self._adversarial_loss)
            if self._is_cycle_consistent:
                tf.summary.scalar("Auxiliary_Loss", self._aux_loss)


    def _define_gradient_penalty(self):
        alpha = tf.random_uniform(shape=tf.shape(self._input_real), minval=0., maxval=1.)
        differences = self._input_fake - self._input_real
        interpolates = self._input_real + (alpha * differences)
        gradients = tf.gradients(self._adversarial.generate_net(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        with tf.name_scope("Loss") as scope:
            self._gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            tf.summary.scalar("Gradient_penalty", self._gradient_penalty)
        return self._gradient_penalty


    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=100, batch_size=64,
              gen_steps=1, adversarial_steps=5, log_step=3, batch_log_step=None, steps=None, gpu_options=None):
        if steps is not None:
            gen_steps = 1
            adversarial_steps = steps
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        self._set_up_test_train_sample(x_train, y_train, x_test, y_test)
        self._log_results(epoch=0, epoch_time=0)
        nr_batches = np.floor(len(x_train) / batch_size)

        self._dominating_adversarial = 0
        self._gen_out_zero = 0
        for epoch in range(epochs):
            batch_nr = 0
            adversarial_loss_epoch = 0
            gen_loss_epoch = 0
            aux_loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            ii = 0

            while trained_examples < len(x_train):
                adversarial_loss_batch, gen_loss_batch, aux_loss_batch = self._optimize(self._trainset, batch_size, adversarial_steps, gen_steps)
                trained_examples += batch_size

                if np.isnan(adversarial_loss_batch) or np.isnan(gen_loss_batch):
                    print("adversarialLoss / GenLoss: ",  adversarial_loss_batch, gen_loss_batch)
                    oar, oaf = self._sess.run([self._output_adversarial_real, self._output_adversarial_fake],
                                    feed_dict={
                                        self._X_input: self.current_batch_x, self._Y_input: self.current_batch_y,
                                        self._Z_input: self._Z_noise, self._is_training: True
                                })
                    print(oar)
                    print(oaf)
                    print(np.max(oar))
                    print(np.max(oaf))

                    # self._check_tf_variables(ii, nr_batches)
                    raise GeneratorExit("Nan found.")

                if (batch_log_step is not None) and (ii % batch_log_step == 0):
                    batch_train_time = (time.clock() - start)/60
                    self._log(int(epoch*nr_batches+ii), batch_train_time)

                adversarial_loss_epoch += adversarial_loss_batch
                gen_loss_epoch += gen_loss_batch
                aux_loss_epoch += aux_loss_batch
                ii += 1


            epoch_train_time = (time.clock() - start)/60
            adversarial_loss_epoch = np.round(adversarial_loss_epoch, 2)
            gen_loss_epoch = np.round(gen_loss_epoch, 2)

            print("Epoch {}: Adversarial: {}.".format(epoch+1, adversarial_loss_epoch))
            print("\t\t\tGenerator: {}.".format(gen_loss_epoch))
            print("\t\t\tEncoder: {}.".format(aux_loss_epoch))

            if self._log_step is not None:
                self._log(epoch+1, epoch_train_time)

            # self._check_tf_variables(epoch, epochs)


    def _optimize(self, dataset, batch_size, adversarial_steps, gen_steps):
        for i in range(adversarial_steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            # self.current_batch_x, self.current_batch_y = current_batch_x, current_batch_y
            self._Z_noise = self.sample_noise(n=len(current_batch_x))
            _, adversarial_loss_batch = self._sess.run([
                                            self._adversarial_optimizer, self._adversarial_loss
                                            ],
                                            feed_dict={
                                                self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                                self._Z_input: self._Z_noise, self._is_training: True
            })

        aux_loss_batch = 0
        for _ in range(gen_steps):
            Z_noise = self._generator.sample_noise(n=len(current_batch_x))
            if not self._is_feature_matching:
                _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss],
                                                   feed_dict={self._Z_input: Z_noise, self._Y_input: current_batch_y,
                                                   self._is_training: True})
            else:
                _, gen_loss_batch = self._sess.run([
                                                   self._gen_optimizer, self._gen_loss
                                                ],
                                                feed_dict={
                                                self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                                self._Z_input: self._Z_noise, self._is_training: True
            })
            if self._is_cycle_consistent:
                _, aux_loss_batch = self._sess.run([self._aux_optimizer, self._aux_loss],
                                                   feed_dict={self._Z_input: Z_noise, self._Y_input: current_batch_y,
                                                   self._is_training: True})

        return adversarial_loss_batch, gen_loss_batch, aux_loss_batch


    def predict(self, inpt_x, inpt_y):
        inpt = self._sess.run(self._input_real, feed_dict={self._X_input: inpt_x, self._Y_input: inpt_y, self._is_training: True})
        return self._adversarial.predict(inpt, self._sess)


    def _check_tf_variables(self, batch_nr, nr_batches):
        Z_noise = self._generator.sample_noise(n=len(self._x_test))
        gen_grads = [self._sess.run(gen_gv[0], feed_dict={self._X_input: self._x_test,
                                    self._Y_input: self._y_test, self._Z_input: Z_noise, self._is_training: False})

                                for gen_gv in self._gen_grads_and_vars]
        adversarial_grads = [self._sess.run(adversarial_gv[0], feed_dict={self._X_input: self._x_test,
                                    self._Y_input: self._y_test, self._Z_input: Z_noise, self._is_training: False})

                                for adversarial_gv in self._adversarial_grads_and_vars]
        gen_grads_maxis = [np.max(gv) for gv in gen_grads]
        gen_grads_means = [np.mean(gv) for gv in gen_grads]
        gen_grads_minis = [np.min(gv) for gv in gen_grads]
        adversarial_grads_maxis = [np.max(dv) for dv in adversarial_grads]
        adversarial_grads_means = [np.mean(dv) for dv in adversarial_grads]
        adversarial_grads_minis = [np.min(dv) for dv in adversarial_grads]

        real_logits, fake_logits, gen_out = self._sess.run(
                [self._output_adversarial_real, self._output_adversarial_fake, self._output_gen],
                feed_dict={self._X_input: self._x_test, self._Y_input: self._y_test,
                            self._Z_input: Z_noise, self._is_training: False})
        real_logits = np.mean(real_logits)
        fake_logits = np.mean(fake_logits)

        gen_varsis = np.array([x.eval(session=self._sess) for x in self._generator.get_network_params()])
        adversarial_varsis = np.array([x.eval(session=self._sess) for x in self._adversarial.get_network_params()])
        gen_maxis = np.array([np.max(x) for x in gen_varsis])
        adversarial_maxis = np.array([np.max(x) for x in adversarial_varsis])
        gen_means = np.array([np.mean(x) for x in gen_varsis])
        adversarial_means = np.array([np.mean(x) for x in adversarial_varsis])
        gen_minis = np.array([np.min(x) for x in gen_varsis])
        adversarial_minis = np.array([np.min(x) for x in adversarial_varsis])

        print(batch_nr, "/", nr_batches, ":")
        print("adversarialReal / adversarialFake: ",  real_logits, fake_logits)
        print("GenWeight Max / Mean / Min: ",  np.max(gen_maxis), np.mean(gen_means), np.min(gen_minis))
        print("GenGrads Max / Mean / Min: ",  np.max(gen_grads_maxis), np.mean(gen_grads_means), np.min(gen_grads_minis))
        print("adversarialWeight Max / Mean / Min: ",  np.max(adversarial_maxis), np.mean(adversarial_means), np.min(adversarial_minis))
        print("adversarialGrads Max / Mean / Min: ",  np.max(adversarial_grads_maxis), np.mean(adversarial_grads_means), np.min(adversarial_grads_minis))
        print("GenOut Max / Mean / Min: ",  np.max(gen_out), np.mean(gen_out), np.min(gen_out))
        print("\n")

        if real_logits > 0.99 and fake_logits < 0.01:
            self._dominating_adversarial += 1
            if self._dominating_adversarial == 5:
                raise GeneratorExit("Dominating adversarialriminator!")
        else:
            self._dominating_adversarial = 0

        print(np.max(gen_out))
        print(np.max(gen_out) < 0.05)
        if np.max(gen_out) < 0.05:
            self._gen_out_zero += 1
            print(self._gen_out_zero)
            if self._gen_out_zero == 50:
                raise GeneratorExit("Generator outputs zeros")
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
    adversarial_architecture = [
                        [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        # [tf.layers.batch_normalization, {}],
                        # [tf.layers.dropout, {}],

                        [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        [tf.layers.conv2d, {"filters": 1, "kernel_size": 5, "strides": 1, "padding": "same", "activation": tf.nn.sigmoid}],
                        ]
    auxiliary_architecture = [
                        [tf.layers.flatten, {}],
                        [tf.layers.dense, {"units": 138, "activation": tf.identity}]
                        ]
    inpt_dim = x_train[0].shape
    image_shape=[32, 32, 1]

    z_dim = 128
    label_dim = 10
    gpu_options = None
    is_patchGAN = True
    max_iterations = 10
    loss = "wasserstein"
    save_folder = "../../../Results/Test/CGAN"
    feature_matching = True

    is_wasserstein = loss == "wasserstein"

    if is_patchGAN and is_wasserstein:
        adversarial_architecture[-1][1]["activation"] = tf.identity

    fails = ""
    run_nr = 1
    continue_training = True
    while continue_training and run_nr <= max_iterations:
        try:
            cgan = CGAN(x_dim=inpt_dim, y_dim=label_dim, z_dim=z_dim,
                        gen_architecture=gen_architecture, adversarial_architecture=adversarial_architecture,
                        folder=save_folder, append_y_at_every_layer=False, is_patchgan=is_patchGAN,
                        is_wasserstein=is_wasserstein, aux_architecture=auxiliary_architecture)
            print(cgan.show_architecture())
            print(cgan._generator.get_number_params())
            print(cgan._adversarial.get_number_params())
            cgan.log_architecture()
            cgan.compile(loss=loss, logged_images=x_train_log, logged_labels=y_train_log, feature_matching=feature_matching)
            cgan.train(x_train, y_train, x_test, y_test, epochs=100, adversarial_steps=5, gen_steps=1, log_step=3,
                          batch_log_step=2000, gpu_options=gpu_options)
            continue_training = False
            with open(save_folder+"/FAILS.txt", "w") as f:
                fails += "{})\t SUCCESS!\n".format(str(run_nr))
                f.write(fails)
        except GeneratorExit as e:
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

