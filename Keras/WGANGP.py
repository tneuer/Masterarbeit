#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-08-21 20:21:42
    # Description : Code more or less directly from:
        https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
####################################################################################
"""
import os
if "lhcb_data2" in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.insert(1, "../Preprocessing")
sys.path.insert(1, "../Utilities")
import json
import pickle
import grid_search

import numpy as np
import tensorflow as tf
import json_to_keras as jtk
import matplotlib.pyplot as plt
import Preprocessing.initialization as init

from functools import partial
from sklearn.preprocessing import OneHotEncoder
from functionsOnImages import padding_zeros, build_images
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, crop_images

if "lhcb_data2" in os.getcwd():
    from keras.backend.tensorflow_backend import set_session
    gpu_frac = 0.3
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
    set_session(tf.Session(config=config))
    print("1 GPU limited to {}% memory.".format(np.round(gpu_frac*100)))
else:
    gpu_options = None

#####################
import keras
from keras.models import Model
from keras.layers.merge import _Merge
from keras.optimizers import RMSprop, Adam
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense
from keras.utils.vis_utils import plot_model
import keras.backend as K


class CWGANGP():
    def __init__(self, x_dim, y_dim, z_dim, path_to_json, folder="./CWGANGP", image_shape=None, optimizer=Adam, learning_rate=0.001):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.channels = image_shape[-1]
        self.image_shape = image_shape
        self.latent_dim = z_dim
        self.folder = folder
        self.path_to_json = path_to_json

        # Following parameter and optimizer set as recommended in paper
        optimizer = optimizer(lr=learning_rate)

        # Build the generator and critic
        self.generator = self.build_generator_from_json()
        self.critic = self.build_discriminator_from_json()


        class RandomWeightedAverage(_Merge):
            """Provides a (random) weighted average between real and generated image samples"""
            def _merge_function(self, inputs):
                alpha = K.random_uniform((batch_size, 1, 1, 1))
                return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.image_shape)

        # Noise input
        z_input = Input(shape=(self.latent_dim+self.y_dim, ))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_input)

        y_input = Input(shape=(*self.image_shape[:-1], self.y_dim))
        fake_img_concat = Concatenate(axis=3)([fake_img, y_input])
        real_img_concat = Concatenate(axis=3)([real_img, y_input])

        # Discriminator determines validity of the real and fake images
        output_critic_fake = self.critic(fake_img_concat)
        output_critic_real = self.critic(real_img_concat)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        interpolated_img_concat = Concatenate(axis=3)([interpolated_img, y_input])

        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img_concat)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_input, y_input],
                            outputs=[output_critic_real, output_critic_fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss],
                                    optimizer=optimizer,
                                    loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_input_gen = Input(shape=(self.latent_dim+self.y_dim,))
        y_input_gen = Input(shape=(*self.image_shape[:-1], self.y_dim))
        # Generate images based of noise
        fake_img_gen = self.generator(z_input_gen)
        img_concat_gen = Concatenate(axis=3)([fake_img_gen, y_input_gen])

        # Discriminator determines validity
        output_critic_gen = self.critic(img_concat_gen)
        # Defines generator model
        self.generator_model = Model(inputs=[z_input_gen, y_input_gen], outputs=output_critic_gen)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


        #-------------------------------
        # Save model architecture
        #         for Generator
        #-------------------------------
        sys_console = sys.stdout
        sys.stdout = open('{}/architecture_details.txt'.format(path_saving), 'w')
        self.generator.summary()
        print("\n\n\n")
        self.critic.summary()
        print("\n\n\n")
        sys.stdout = sys_console

        # plot the model
        plot_model(self.generator, to_file='{}/g_model_plot.png'.format(path_saving), show_shapes=True, show_layer_names=True)
        plot_model(self.critic, to_file='{}/d_model_plot.png'.format(path_saving), show_shapes=True, show_layer_names=True)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def build_generator_from_json(self):
        # image input
        in_image = Input(shape=[self.latent_dim+self.y_dim])
        out_image = jtk.json_to_keras(inpt=in_image, path_to_json=self.path_to_json, network="Generator")
        # define model
        model = Model(in_image, out_image)
        return model

    def build_discriminator_from_json(self):
        in_image = Input(shape=[self.image_shape[0], self.image_shape[1], self.image_shape[2]+self.y_dim])
        out_image = jtk.json_to_keras(inpt=in_image, path_to_json=path_to_json, network="Critic")
        out_image = Conv2D(filters=64, kernel_size=4, strides=2, padding="same")(out_image)
        out_image = Conv2D(filters=8, kernel_size=4, strides=2, padding="same", activation="relu")(out_image)
        out_image = Flatten()(out_image)
        out_image = Dense(units=1, activation="tanh")(out_image)
        model = Model(in_image, out_image)
        return model


    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=100, batch_size=32, evaluate_interval=50,
              label_smoothing=1, adversarial_steps=1):

        out_patch = self.critic.layers[-1].output_shape[1:]
        real_labels = -np.ones((batch_size, *out_patch))*label_smoothing
        fake_labels =  np.ones((batch_size, *out_patch))*label_smoothing
        dummy_labels = np.zeros((batch_size, *out_patch))*label_smoothing # Dummy gt for gradient penalty

        self._gen_out_zero = 0
        for epoch in range(epochs):

            for i in range(adversarial_steps):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                train_images = x_train[idx]
                train_labels = y_train[idx]
                train_labels = np.array([np.tile(lbl, (*image_shape[:-1], 1)) for lbl in train_labels])

                # Sample generator input
                train_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                train_input = np.concatenate([train_noise, y_train[idx]], axis=1)

                # Train the critic
                d_loss = self.critic_model.train_on_batch([train_images, train_input, train_labels],
                                                                [real_labels, fake_labels, dummy_labels])


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch([train_input, train_labels], real_labels)

            if np.isinf(g_loss) or np.isnan(g_loss) or np.isnan(d_loss[0]) or np.isinf(d_loss[0]):
                raise GeneratorExit("Nan or inf in losses: d_loss: {} |-| g_loss: {}.".format(d_loss, g_loss))

            # Print the progress
            if epoch % evaluate_interval == 0:
                print(">%d / %d, d_tot[%.3f], d_real[%.3f], d_fake[%.3f], d_grad[%.3f], g[%.3f]" %
                        (epoch+1, epochs, d_loss[0], d_loss[1], d_loss[2], d_loss[3] , g_loss))
                nr_eval = 100
                im_shape = self.image_shape
                test_noise = np.random.normal(0, 1, (nr_eval, self.latent_dim))
                test_input = np.concatenate([test_noise, y_test[:nr_eval]], axis=1)
                samples_gen = self.generator.predict(test_input).reshape([-1, *im_shape[:-1]])
                samples_x = x_test[:nr_eval].reshape([-1, *im_shape[:-1]])

                array_of_images = np.array([
                    [
                        tx.reshape(im_shape[0:2]),
                        gx.reshape(im_shape[0:2])
                    ] for tx, gx in zip(x_test[:10], samples_gen[:10])
                ])
                fig, ax = build_images(array_of_images, column_titles=["X", "Y", "G(X)"], fs_x=45, fs_y=25)
                plt.savefig("{}/GeneratedSamples/Samples_{}.png".format(path_saving, epoch))
                energy_scaler = scaler["Calo"]

                build_histogram(true=samples_x, fake=samples_gen, function=get_energies, name="Energy", epoch=epoch,
                                folder=path_saving, energy_scaler=energy_scaler)
                build_histogram(true=samples_x, fake=samples_gen, function=get_number_of_activated_cells, name="Cells", epoch=epoch,
                                folder=path_saving, threshold=5/energy_scaler)
                build_histogram(true=samples_x, fake=samples_gen, function=get_max_energy, name="MaxEnergy", epoch=epoch,
                                folder=path_saving, energy_scaler=energy_scaler)
                build_histogram(true=samples_x, fake=samples_gen, function=get_center_of_mass_x, name="CenterOfMassX", epoch=epoch,
                                folder=path_saving, image_shape=im_shape)
                build_histogram(true=samples_x, fake=samples_gen, function=get_center_of_mass_y, name="CenterOfMassY", epoch=epoch,
                                folder=path_saving, image_shape=im_shape)
                build_histogram(true=samples_x, fake=samples_gen, function=get_std_energy, name="StdEnergy", epoch=epoch,
                                folder=path_saving, energy_scaler=energy_scaler)

                save_files = os.listdir(path_saving+"/ModelSave")
                for f in save_files:
                    os.remove(path_saving+"/ModelSave/"+f)
                with open(path_saving+"/ModelSave/Generator_{}.pickle".format(epoch), "wb") as f:
                    pickle.dump(self.generator, f)
                with open(path_saving+"/ModelSave/Discriminator_{}.pickle".format(epoch), "wb") as f:
                    pickle.dump(self.critic, f)
                self.sample_images(epoch, samples_gen)

                samples_gen = 0
                if np.max(samples_gen) < 0.05:
                    self._gen_out_zero += 1
                    if self._gen_out_zero == 20:
                        raise GeneratorExit("Generator outputs zeros")
                else:
                    self._gen_out_zero = 0


    def sample_images(self, epoch, gen_imgs):
        r, c = 10, 5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("{}/GeneratedSamples/Samples_{}.png".format(self.folder, epoch))
        plt.close()



if __name__ == "__main__":
    ############################################################################################################
    # Parameter definiton
    ############################################################################################################
    keep_cols = ["x_projections", "y_projections", "real_ET"]
    padding = {"top":2, "bottom":2, "left":0, "right":0}
    x_dim = image_shape = (52+padding["top"]+padding["bottom"], 64+padding["left"]+padding["right"], 1)
    nr_test = 100

    if "lhcb_data2" in os.getcwd():
        path_loading = "../../Data/PiplusLowerP/LargeSample"
        path_results = "../../Results/PiplusLowerP"
    else:
        path_loading = "../../Data/PiplusLowerP/Debug"
        path_results = "../../Results/Test/PiplusLowerP"
    algorithm = "CWGANGP-Keras"


    ############################################################################################################
    # Structure preparation
    ############################################################################################################
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    ############################################################################################################
    # Data loading
    ############################################################################################################
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    data, scaler = init.load_processed_data(path_loading, return_scaler=True)
    train_calo = data["train"]["Calo"]
    train_tracker = data["train"]["Tracker"]
    test_calo = data["test"]["Calo"]
    test_tracker = data["test"]["Tracker"]

    train_calo = padding_zeros(train_calo, **padding).reshape([-1, *image_shape])
    test_calo = padding_zeros(test_calo, **padding).reshape([-1, *image_shape])
    test_calo = test_calo[:nr_test]
    logging_calo = test_calo[:15]

    ##### Rescale and check that identical
    def invert_standardize_data(data, scaler, exclude=None):
        import pandas as pd
        standardized_data = data.drop(exclude, axis=1, inplace=False)
        colnames = standardized_data.columns.values
        standardized_data = pd.DataFrame(data=scaler.inverse_transform(standardized_data), columns=colnames, index=data.index)
        data = pd.concat([standardized_data, data[exclude]], axis=1, sort=False)
        return data

    train_tracker["real_ET"] = invert_standardize_data(data=train_tracker, scaler=scaler["Tracker"], exclude=["theta", "phi", "region"])["real_ET"]
    train_tracker["real_ET"] /= scaler["Calo"]

    test_tracker["real_ET"] = invert_standardize_data(data=test_tracker, scaler=scaler["Tracker"], exclude=["theta", "phi", "region"])["real_ET"]
    test_tracker["real_ET"] /= scaler["Calo"]

    assert np.max(train_calo) == 1, "Train calo maximum not one. Given: {}.".format(np.max(train_calo))
    assert np.allclose(np.mean(train_tracker[keep_cols[:-1]], axis=0), 0, atol=1e-5), "Train not centralized: {}.".format(
        np.mean(train_tracker[keep_cols], axis=0)
    )
    # assert np.allclose(np.mean(test_tracker, axis=0), 0, atol=1e-1), "Test not centralized: {}.".format(np.mean(test_tracker, axis=0))
    assert np.allclose(np.std(train_tracker[keep_cols[:-1]], axis=0), 1, atol=1e-10), "Train not standardized: {}.".format(
        np.std(train_tracker[keep_cols], axis=0)
    )
    assert image_shape == train_calo.shape[1:], "Wrong image shape vs train shape: {} vs {}.".format(image_shape, train_calo.shape[1:])
    train_tracker = train_tracker[keep_cols].values
    test_tracker = test_tracker[keep_cols].values
    test_tracker = test_tracker[:nr_test]
    logging_tracker = test_tracker[:15]

    nr_train = train_tracker.shape[0]

    ############################################################################################################
    # Model defintion
    ############################################################################################################
    param_dict = {
            "z_dim": [32, 64],
            "optimizer": [Adam, RMSprop],
            "dataset": ["PiplusLowerP"],
            "gen_steps": [1],
            "adv_steps": [5, 1],
            "architecture": ["unbalanced2", "unbalanced", "unbalanced5", "unbalanced6"],
            "batch_size": [8, 16, 32],
            "lr": [0.001, 0.001, 0.005],
            "label_smoothing": [1]
    }
    sampled_params = grid_search.get_parameter_grid(param_dict=param_dict, n=30, allow_repetition=True)
    for params in sampled_params:

        path_saving = init.initialize_folder(algorithm=algorithm, base_folder=path_results)
        os.mkdir(path_saving+"/Evaluation/Cells")
        os.mkdir(path_saving+"/Evaluation/CenterOfMassX")
        os.mkdir(path_saving+"/Evaluation/CenterOfMassY")
        os.mkdir(path_saving+"/Evaluation/Energy")
        os.mkdir(path_saving+"/Evaluation/MaxEnergy")
        os.mkdir(path_saving+"/Evaluation/StdEnergy")
        os.mkdir(path_saving+"/ModelSave")

        adv_steps = int(params["adv_steps"])
        z_dim = 32
        y_dim = train_tracker.shape[1]
        architecture = str(params["architecture"])
        path_to_json = "../../Architectures/CGAN/{}.json".format(architecture)
        batch_size = int(params["batch_size"])
        epochs = int((nr_train / batch_size) * 120)
        evaluate_interval = int(epochs / 200)
        learning_rate = float(params["lr"])
        optimizer = params["optimizer"]
        label_smoothing = float(params["label_smoothing"])

        config_data = init.create_config_file(globals())
        with open(path_saving+"/config.json", "w") as f:
            json.dump(config_data, f, indent=4)

        wgan = CWGANGP(x_dim =image_shape, y_dim=y_dim, z_dim=z_dim, path_to_json=path_to_json,
                       image_shape=image_shape,  folder=path_saving, optimizer=optimizer, learning_rate=learning_rate)
        try:
            wgan.train(train_calo, train_tracker, x_test=test_calo, y_test=test_tracker, epochs=epochs, evaluate_interval=evaluate_interval,
                       batch_size=batch_size, label_smoothing=label_smoothing, adversarial_steps=adv_steps)
            with open(path_saving+"/EXIT_FLAG0.txt", "w") as f:
                f.write("EXIT STATUS: 0. No errors or warnings.")
        except GeneratorExit as e:
            with open(path_saving+"/EXIT_FLAG1.txt", "w") as f:
                f.write("EXIT STATUS: 1. {}.".format(e))