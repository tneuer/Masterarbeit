#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-08-10 11:09:04
    # Description :
####################################################################################
"""
import os
import sys
import time
import json
sys.path.insert(1, "../..")
sys.path.insert(1, "../building_blocks")
sys.path.insert(1, "../../Utilities")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from layers import logged_dense, conv2d_logged, conv2d_transpose_logged, reshape_layer
from layers import concatenate_with, residual_block, unet, unet_original, inception_block
from networks import Generator, Discriminator
from generativeModels import Image2ImageGenerativeModel
from functionsOnImages import padding_zeros
import Preprocessing.initialization as init
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, crop_images


class Pix2PixGAN(Image2ImageGenerativeModel):
    def __init__(self, x_dim, y_dim, gen_architecture, disc_architecture,
                    folder="./Pix2Pix", PatchGAN=False, is_wasserstein=False
        ):
        super(Pix2PixGAN, self).__init__(x_dim, y_dim, [gen_architecture, disc_architecture], folder)

        self._gen_architecture = self._architectures[0]
        self._disc_architecture = self._architectures[1]
        self._is_patchgan = PatchGAN
        self._is_wasserstein = is_wasserstein

        ################# Define architecture

        if self._is_patchgan:
            f_xy = self._disc_architecture[-1][-1]["filters"]
            assert f_xy == 1, "If is PatchGAN, last layer of Discriminator_XY needs 1 filter. Given: {}.".format(f_xy)

            a_xy = self._disc_architecture[-1][-1]["activation"]
            if self._is_wasserstein:
                assert a_xy == tf.identity, "If is PatchGAN, last layer of Discriminator_XY needs tf.nn.identity. Given: {}.".format(a_xy)
            else:
                assert a_xy == tf.nn.sigmoid, "If is PatchGAN, last layer of Discriminator_XY needs tf.nn.sigmoid. Given: {}.".format(a_xy)
        else:
            self._disc_architecture.append([tf.layers.flatten, {"name": "Flatten"}])
            if self._is_wasserstein:
                self._disc_architecture.append([logged_dense, {"units": 1, "activation": tf.identity, "name": "Output"}])
            else:
                self._disc_architecture.append([logged_dense, {"units": 1, "activation": tf.nn.sigmoid, "name": "Output"}])

        self._gen_architecture[-1][1]["name"] = "Output"

        self._generator = Generator(self._gen_architecture, name="Generator")
        self._discriminator = Discriminator(self._disc_architecture, name="Discriminator")

        self._nets = [self._generator, self._discriminator]

        ################# Connect inputs and networks
        self._output_gen = self._generator.generate_net(self._X_input, tf_trainflag=self._is_training)

        with tf.name_scope("InputsMod"):
            gen_out_shape = self._output_gen.get_shape()
            x_shape = self._X_input.get_shape()
            y_shape = self._Y_input.get_shape()
            assert gen_out_shape[1] == x_shape[1] and gen_out_shape[2] == x_shape[2], (
                "Wrong shapes: Generator output has {}, while X input has {}.".format(gen_out_shape, x_shape)
            )
            assert y_shape[1] == x_shape[1] and y_shape[2] == x_shape[2], (
                "Wrong shapes: Y input has {}, while X input has {}.".format(y_shape, x_shape)
            )
            self._output_gen_mod = tf.concat(values=[self._output_gen, self._X_input], axis=3, name="mod_gen")
            self._Y_input_mod = tf.concat(values=[self._Y_input, self._X_input], axis=3, name="mod_y")

        self._output_disc_real = self._discriminator.generate_net(self._Y_input_mod, tf_trainflag=self._is_training)
        self._output_disc_fake = self._discriminator.generate_net(self._output_gen_mod, tf_trainflag=self._is_training)

        if self._is_patchgan:
            print("PATCHGAN chosen with output: {}.".format(self._output_disc_real.shape))

        ################# Finalize

        self._init_folders()
        self._verify_init()


    def compile(self, learning_rate=0.0005, learning_rate_gen=None, learning_rate_disc=None,
                    optimizer=tf.train.AdamOptimizer, lmbda=10, loss="cross-entropy", **kwargs):
        if self._is_wasserstein and loss != "wasserstein":
            raise ValueError("If is_wasserstein is true in Constructor, loss needs to be wasserstein.")
        if not self._is_wasserstein and loss == "wasserstein":
            raise ValueError("If loss is wasserstein, is_wasserstein needs to be true in constructor.")

        if learning_rate_gen is None:
            learning_rate_gen = learning_rate
        if learning_rate_disc is None:
            learning_rate_disc = learning_rate
        self._define_loss(lmbda, loss)
        with tf.name_scope("Optimizer"):
            gen_optimizer = optimizer(learning_rate=learning_rate_gen, **kwargs)
            self._gen_optimizer = gen_optimizer.minimize(self._gen_loss,
                                                         var_list=self._get_vars(scope="Generator"),
                                                         name="Generator")
            disc_optimizer = optimizer(learning_rate=learning_rate_disc, **kwargs)
            self._disc_optimizer = disc_optimizer.minimize(self._disc_loss,
                                                            var_list=self._get_vars(scope="Discriminator"),
                                                            name="Discriminator")

            self._gen_grads_and_vars = gen_optimizer.compute_gradients(self._gen_loss, var_list=self._get_vars(scope="Generator"))
            self._disc_grads_and_vars = disc_optimizer.compute_gradients(self._disc_loss, var_list=self._get_vars(scope="Discriminator"))
        self._summarise()


    def _define_loss(self, lmbda, loss):
        possible_losses = ["cross-entropy", "L1", "L2", "wasserstein"]
        if loss == "wasserstein":
            self._gen_loss_vanilla = -tf.reduce_mean(self._output_disc_fake)
            self._disc_loss = (-(tf.reduce_mean(self._output_disc_real) -
                                    tf.reduce_mean(self._output_disc_fake)) +
                                    10*self._define_gradient_penalty()
            )
        elif loss == "cross-entropy":
            eps = 1e-7
            self._logits_real = tf.math.log(self._output_disc_real / (1 - self._output_disc_real + eps))
            self._logits_fake = tf.math.log(self._output_disc_fake / (1 - self._output_disc_fake + eps))

            self._gen_loss_vanilla = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=(1-eps)*tf.ones_like(self._logits_fake), logits=self._logits_fake + eps
            ))
            self._disc_loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=(1-eps)*tf.ones_like(self._logits_real), logits=self._logits_real + eps
                                    ) +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.zeros_like(self._logits_fake) + eps, logits=self._logits_fake + eps
                                    )
            )

        elif loss == "L1":
            self._gen_loss_vanilla = tf.reduce_mean(tf.abs(self._output_disc_fake - tf.ones_like(self._output_disc_fake)))
            self._disc_loss = (
                        tf.reduce_mean(
                                        tf.abs(self._output_disc_real - tf.ones_like(self._output_disc_real)) +
                                        tf.abs(self._output_disc_fake)
                        )
            ) / 2.0

        elif loss == "L2":
            self._gen_loss_vanilla = tf.reduce_mean(tf.square(self._output_disc_fake - tf.ones_like(self._output_disc_fake)))
            self._disc_loss = (
                        tf.reduce_mean(
                                        tf.square(self._output_disc_real - tf.ones_like(self._output_disc_real)) +
                                        tf.square(self._output_disc_fake)
                        )
            ) / 2.0
        else:
            raise ValueError("Loss not implemented. Choose from {}. Given: {}.".format(possible_losses, loss))

        with tf.name_scope("Loss") as scope:
            tf.summary.scalar("Generator_vanilla_loss", self._gen_loss_vanilla)

            self._recon_loss = lmbda*tf.reduce_mean(tf.abs(self._Y_input - self._output_gen))
            tf.summary.scalar("Generator_recon_loss", self._recon_loss)

            self._gen_loss = self._gen_loss_vanilla + self._recon_loss
            tf.summary.scalar("Generator_total_loss", self._gen_loss)

            tf.summary.scalar("Discriminator_loss", self._disc_loss)


    def _define_gradient_penalty(self):
        alpha = tf.random_uniform(shape=tf.shape(self._Y_input), minval=0., maxval=1.)
        differences = self._output_gen - self._Y_input
        interpolates = self._Y_input + (alpha * differences)
        interpolates = tf.concat(values=[interpolates, self._X_input], axis=3)
        gradients = tf.gradients(self._discriminator.generate_net(interpolates, tf_trainflag=self._is_training), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        with tf.name_scope("Loss") as scope:
            self._gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            tf.summary.scalar("Gradient_penalty", self._gradient_penalty)
        return self._gradient_penalty


    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=100, batch_size=64,
              gen_steps=1, disc_steps=5, steps=None, log_step=3, batch_log_step=None, gpu_options=None):
        if steps is not None:
            gen_steps = 1
            disc_steps = steps
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        self._set_up_test_train_sample(x_train, y_train, x_test, y_test)
        self._log_results(epoch=0, epoch_time=0)
        nr_batches = np.floor(len(x_train) / batch_size)

        self._dominating_disc = 0
        self._gen_out_zero = 0
        self._mode_collapse = 0

        for epoch in range(epochs):
            batch_nr = 0
            disc_loss_epoch = 0
            gen_loss_epoch = 0
            start = time.clock()
            trained_examples = 0
            ii = 0

            while trained_examples < len(x_train):
                disc_loss_batch, gen_loss_batch = self._optimize(self._trainset, batch_size, disc_steps, gen_steps)
                trained_examples += batch_size

                if (batch_log_step is not None) and (ii % batch_log_step == 0):
                    batch_train_time = (time.clock() - start)/60
                    self._log(int(epoch*nr_batches+ii), batch_train_time)

                disc_loss_epoch += disc_loss_batch
                gen_loss_epoch += gen_loss_batch
                ii += 1

            disc_loss_epoch /= nr_batches
            gen_loss_epoch /= nr_batches
            print("D: ", self._sess.run([self._disc_loss],
                                                     feed_dict={self._X_input: x_test, self._Y_input: y_test, self._is_training: False}))
            print("G: ", self._sess.run([self._gen_loss_vanilla],
                                                     feed_dict={self._X_input: x_test, self._Y_input: y_test, self._is_training: False}))
            print("R: ", self._sess.run([self._recon_loss],
                                                     feed_dict={self._X_input: x_test, self._Y_input: y_test, self._is_training: False}))
            print("G Total: ", self._sess.run([self._gen_loss],
                                                     feed_dict={self._X_input: x_test, self._Y_input: y_test, self._is_training: False}))

            epoch_train_time = (time.clock() - start)/60
            disc_loss_epoch = np.round(disc_loss_epoch, 7)
            gen_loss_epoch = np.round(gen_loss_epoch, 7)

            print("Epoch {}: Discrimiantor: {} \n\t\t\tGenerator: {}.".format(epoch+1, disc_loss_epoch, gen_loss_epoch))

            if self._log_step is not None:
                self._log(epoch+1, epoch_train_time)

            self._check_tf_variables(epoch, epochs)


    def _optimize(self, dataset, batch_size, disc_steps, gen_steps):
        d_loss = 0
        for i in range(disc_steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            _, disc_loss_batch = self._sess.run([
                                            self._disc_optimizer, self._disc_loss
                                            ],
                                            feed_dict={self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                            self._is_training: True})
            d_loss += disc_loss_batch
        d_loss /= disc_steps

        g_loss = 0
        for _ in range(gen_steps):
            current_batch_x, current_batch_y = dataset.get_next_batch(batch_size)
            _, gen_loss_batch = self._sess.run([self._gen_optimizer, self._gen_loss],
                                               feed_dict={self._X_input: current_batch_x, self._Y_input: current_batch_y,
                                               self._is_training: True})
            g_loss += gen_loss_batch
        g_loss /= gen_steps
        return d_loss, g_loss


    def _check_tf_variables(self, batch_nr, nr_batches):
        # gen_grads = [self._sess.run(gen_gv[0], feed_dict={self._X_input: self._x_test,
        #                             self._Y_input: self._y_test, self._is_training: False})

        #                         for gen_gv in self._gen_grads_and_vars]
        # disc_grads = [self._sess.run(disc_gv[0], feed_dict={self._X_input: self._x_test,
        #                             self._Y_input: self._y_test, self._is_training: False})

        #                         for disc_gv in self._disc_grads_and_vars]
        # gen_grads_maxis = [np.max(gv) for gv in gen_grads]
        # gen_grads_means = [np.mean(gv) for gv in gen_grads]
        # gen_grads_minis = [np.min(gv) for gv in gen_grads]
        # disc_grads_maxis = [np.max(dv) for dv in disc_grads]
        # disc_grads_means = [np.mean(dv) for dv in disc_grads]
        # disc_grads_minis = [np.min(dv) for dv in disc_grads]

        real_logits, fake_logits, gen_out, tf_recon_loss, tf_vanilla_loss = self._sess.run(
                [self._output_disc_real, self._output_disc_fake, self._output_gen, self._recon_loss, self._gen_loss_vanilla],
                feed_dict={self._X_input: self._x_test, self._Y_input: self._y_test, self._is_training: False})
        real_logits = np.mean(real_logits)
        fake_logits = np.mean(fake_logits)

        # gen_varsis = np.array([x.eval(session=self._sess) for x in self._generator.get_network_params()])
        # disc_varsis = np.array([x.eval(session=self._sess) for x in self._discriminator.get_network_params()])
        # gen_maxis = np.array([np.max(x) for x in gen_varsis])
        # disc_maxis = np.array([np.max(x) for x in disc_varsis])
        # gen_means = np.array([np.mean(x) for x in gen_varsis])
        # disc_means = np.array([np.mean(x) for x in disc_varsis])
        # gen_minis = np.array([np.min(x) for x in gen_varsis])
        # disc_minis = np.array([np.min(x) for x in disc_varsis])

        # my_recon_loss = np.mean(np.abs(gen_out - self._y_test))
        # my_vanilla_loss = -np.mean(np.log(fake_logits + 1e-7))

        print(batch_nr, "/", nr_batches, ":")
        # print("MyReconLoss / TFReconLoss: ",  my_recon_loss, tf_recon_loss)
        # print("MyVanillaLoss / TFVanillaLoss: ",  my_vanilla_loss, tf_vanilla_loss)
        # print("DiscReal / DiscFake: ",  real_logits, fake_logits)
        # print("GenWeight Max / Mean / Min: ",  np.max(gen_maxis), np.mean(gen_means), np.min(gen_minis))
        # print("GenGrads Max / Mean / Min: ",  np.max(gen_grads_maxis), np.mean(gen_grads_means), np.min(gen_grads_minis))
        # print("DiscWeight Max / Mean / Min: ",  np.max(disc_maxis), np.mean(disc_means), np.min(disc_minis))
        # print("DiscGrads Max / Mean / Min: ",  np.max(disc_grads_maxis), np.mean(disc_grads_means), np.min(disc_grads_minis))
        # print("GenOut Max / Mean / Min: ",  np.max(gen_out), np.mean(gen_out), np.min(gen_out))
        # print("\n")

        if real_logits > 0.99 and fake_logits < 0.01:
            self._dominating_disc += 1
            if self._dominating_disc == 7:
                raise GeneratorExit("Dominating discriminator!")
        else:
            self._dominating_disc = 0


        if np.max(gen_out) < 0.05:
            self._gen_out_zero += 1
            if self._gen_out_zero == 7:
                raise GeneratorExit("Generator outputs zeros.")
        else:
            self._gen_out_zero = 0


        cells = get_number_of_activated_cells(gen_out.reshape([-1, self._image_shape[0], self._image_shape[1]]))
        cells_collapsed = np.allclose(cells, np.mean(cells), atol=1e-4)
        if cells_collapsed:
            self._mode_collapse += 1
            if self._mode_collapse == 7:
                raise GeneratorExit("Generator collapsed to single mode.")
        else:
            self._mode_collapse = 0


    def evaluate(self, true, condition, epoch):
        if not os.path.exists(self._folder+"/Evaluation"):
            os.mkdir(self._folder+"/Evaluation")
        if not os.path.exists(self._folder+"/Evaluation/Cells"):
            os.mkdir(self._folder+"/Evaluation/Cells")
            os.mkdir(self._folder+"/Evaluation/CenterOfMassX")
            os.mkdir(self._folder+"/Evaluation/CenterOfMassY")
            os.mkdir(self._folder+"/Evaluation/Energy")
            os.mkdir(self._folder+"/Evaluation/MaxEnergy")
            os.mkdir(self._folder+"/Evaluation/StdEnergy")
        fake = self.generate_samples(inpt=condition)
        true = true.reshape([-1, self._image_shape[0], self._image_shape[1]])
        fake = fake.reshape([-1, self._image_shape[0], self._image_shape[1]])
        maxEnergy = 6120
        build_histogram(true=true, fake=fake, function=get_energies, name="Energy", epoch=epoch,
                        folder=self._folder, energy_scaler=maxEnergy)
        build_histogram(true=true, fake=fake, function=get_number_of_activated_cells, name="Cells", epoch=epoch,
                        folder=self._folder, threshold=5/maxEnergy)
        build_histogram(true=true, fake=fake, function=get_max_energy, name="MaxEnergy", epoch=epoch,
                        folder=self._folder, energy_scaler=maxEnergy)
        build_histogram(true=true, fake=fake, function=get_center_of_mass_x, name="CenterOfMassX", epoch=epoch,
                        folder=self._folder, image_shape=self._image_shape)
        build_histogram(true=true, fake=fake, function=get_center_of_mass_y, name="CenterOfMassY", epoch=epoch,
                        folder=self._folder, image_shape=self._image_shape)
        build_histogram(true=true, fake=fake, function=get_std_energy, name="StdEnergy", epoch=epoch,
                        folder=self._folder, energy_scaler=maxEnergy)
        plt.close("all")



if __name__ == '__main__':
    ########### Image input
    if "lhcb_data2" in os.getcwd():
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        gpu_frac = 0.3
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
        print("1 GPU limited to {}% memory.".format(np.round(gpu_frac*100)))
    else:
        gpu_options = None

    import pickle
    if "lhcb_data2" in os.getcwd():
        data_path = "../../Data/fashion_mnist"
    else:
        data_path = "/home/tneuer/Backup/Algorithmen/0TestData/image_to_image/fashion_mnist"
    with open("{}/train_images.pickle".format(data_path), "rb") as f:
        x_train = pickle.load(f)[0].reshape(-1, 28, 28, 1) / 255
    with open("{}/train_images_rotated.pickle".format(data_path), "rb") as f:
        y_train = pickle.load(f).reshape(-1, 28, 28, 1) / 255


    ########### Architecture

    x_train = padding_zeros(x_train, top=2, bottom=2, left=2, right=2)
    y_train = padding_zeros(y_train, top=2, bottom=2, left=2, right=2)
    gen_architecture = [
                        [unet_original, {"depth": 2, "filters": 64, "activation": tf.nn.leaky_relu, "logged": True}],
                        [conv2d_logged, {"filters": 128, "kernel_size": 5, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 256, "kernel_size": 5, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 1, "kernel_size": 4, "activation": tf.nn.sigmoid, "padding": "same"}],
                        ]
    disc_architecture = [
                        [conv2d_logged, {"filters": 64, "kernel_size": 4, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [conv2d_logged, {"filters": 128, "kernel_size": 4, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [tf.layers.batch_normalization, {}],
                        [conv2d_logged, {"filters": 256, "kernel_size": 4, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        [tf.layers.batch_normalization, {}],
                        # [conv2d_logged, {"filters": 512, "kernel_size": 4, "strides": 2, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        # [tf.layers.batch_normalization, {}],
                        # [conv2d_logged, {"filters": 512, "kernel_size": 4, "activation": tf.nn.leaky_relu, "padding": "same"}],
                        # [tf.layers.batch_normalization, {}],
                        [conv2d_logged, {"filters": 1, "kernel_size": 4, "activation": tf.nn.sigmoid, "padding": "same"}],
                        ]

    x_test = x_train[-100:]
    x_train = x_train[:-100]
    y_test = y_train[-100:]
    y_train = y_train[:-100]

    lmbda = 100
    learning_rate = 0.0002
    beta1=0.5
    batch_size = 1
    loss = "cross-entropy"
    is_patchgan = False
    epochs = 2
    activation = tf.nn.relu
    gen_steps = 1
    disc_steps = 1
    run_until_success = True
    max_iterations = 10

    if loss == "wasserstein":
        is_wasserstein = True
        disc_architecture[-1][1]["activation"] = tf.identity
    else:
        is_wasserstein = False

    inpt_dim = x_train[0].shape
    opt_dim = y_train[0].shape
    print(inpt_dim)
    print(opt_dim)
    print(np.max(x_train), np.max(y_train))

    config_data = init.create_config_file(globals())
    save_folder = "../../../Results/Test"
    save_folder = init.initialize_folder(algorithm="Pix2Pix_"+loss, base_folder=save_folder)

    fails = ""
    run_nr = 1
    continue_training = True
    while continue_training and run_nr <= max_iterations:
        try:
            pix2pix = Pix2PixGAN(x_dim=inpt_dim, y_dim=opt_dim, gen_architecture=gen_architecture,
                             disc_architecture=disc_architecture,
                            folder=save_folder, PatchGAN=is_patchgan, is_wasserstein=is_wasserstein)
            print(pix2pix._generator.get_number_params())
            print(pix2pix._discriminator.get_number_params())
            nr_params = pix2pix.get_number_params()
            print(pix2pix.show_architecture())
            with open(save_folder+"/config.json", "w") as f:
                json.dump(config_data, f, indent=4)
            pix2pix.log_architecture()
            raise
            pix2pix.compile(learning_rate=learning_rate, lmbda=lmbda, loss=loss, optimizer=tf.train.AdamOptimizer, beta1=beta1)
            pix2pix.train(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size,
                           disc_steps=disc_steps, gen_steps=gen_steps, log_step=1, batch_log_step=2000, gpu_options=gpu_options)
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
