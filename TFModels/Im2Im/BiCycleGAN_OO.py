#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-08-20 16:50:44
    # Description :
####################################################################################
"""
import os
import sys
import time
import json
import copy
sys.path.insert(1, "../..")
sys.path.insert(1, "../building_blocks")
sys.path.insert(1, "../../Utilities")
import grid_search

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import norm
from layers import logged_dense, conv2d_logged, conv2d_transpose_logged, reshape_layer, image_condition_concat
from layers import concatenate_with, residual_block, unet, unet_original, inception_block
from networks import Generator, Discriminator, Encoder
from CVAEGAN_OO import CVAEGAN
from functionsOnImages import padding_zeros
import Preprocessing.initialization as init
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, crop_images
from generativeModels import GenerativeModel


class BiCycleGAN(CVAEGAN):
    def __init__(self, x_dim, y_dim, z_dim, enc_architecture, gen_architecture, adversarial_architecture, folder="./CVAEGAN",
                 is_patchgan=False, is_wasserstein=False):
        super(BiCycleGAN, self).__init__(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, enc_architecture=enc_architecture,
                                         gen_architecture=gen_architecture, adversarial_architecture=adversarial_architecture,
                                         folder=folder, is_patchgan=is_patchgan, is_wasserstein=is_wasserstein)
        self._mean_layer_from_encoding = self._encoder_mean.generate_net(self._output_gen_from_encoding)


    def compile(self, loss, optimizer, learning_rate=None, learning_rate_enc=None, learning_rate_gen=None, learning_rate_adv=None,
                label_smoothing=1, lmbda_kl=0.1, lmbda_y=1, lmbda_z=1, feature_matching=False, random_labeling=0):

        if self._is_wasserstein and loss != "wasserstein":
            raise ValueError("If is_wasserstein is true in Constructor, loss needs to be wasserstein.")
        if not self._is_wasserstein and loss == "wasserstein":
            raise ValueError("If loss is wasserstein, is_wasserstein needs to be true in constructor.")

        if np.all([lr is None for lr in [learning_rate, learning_rate_enc, learning_rate_gen, learning_rate_adv]]):
            raise ValueError("Need learning_rate.")
        if learning_rate is not None and learning_rate_enc is None:
            learning_rate_enc = learning_rate
        if learning_rate is not None and learning_rate_gen is None:
            learning_rate_gen = learning_rate
        if learning_rate is not None and learning_rate_adv is None:
            learning_rate_adv = learning_rate

        self._define_loss(loss=loss, label_smoothing=label_smoothing, lmbda_kl=lmbda_kl, lmbda_y=lmbda_y, lmbda_z=lmbda_z,
                          feature_matching=feature_matching, random_labeling=random_labeling)
        with tf.name_scope("Optimizer"):
            self._enc_optimizer = optimizer(learning_rate=learning_rate_enc)
            self._enc_optimizer_op = self._enc_optimizer.minimize(self._enc_loss, var_list=self._get_vars("Encoder"), name="Encoder")
            self._gen_optimizer = optimizer(learning_rate=learning_rate_gen)
            self._gen_oprimizer_op = self._gen_optimizer.minimize(self._gen_loss, var_list=self._get_vars("Generator"), name="Generator")
            self._adv_optimizer = optimizer(learning_rate=learning_rate_adv)
            self._adv_optimizer_op = self._adv_optimizer.minimize(self._adv_loss, var_list=self._get_vars("Adversarial"), name="Adversarial")
        self._summarise()


    def _define_loss(self, loss, label_smoothing, lmbda_kl, lmbda_y, lmbda_z, feature_matching, random_labeling):
        possible_losses = ["cross-entropy", "L2", "wasserstein", "KL"]
        def get_labels_one():
            return tf.math.multiply(self._output_label_real, label_smoothing)
        def get_labels_zero():
            return self._output_label_fake
        eps = 1e-7
        self._label_smoothing = label_smoothing
        self._random_labeling = random_labeling
        ## Kullback-Leibler divergence
        self._KLdiv = 0.5*(tf.square(self._mean_layer) + tf.exp(self._std_layer) - self._std_layer - 1)
        self._KLdiv = tf.reduce_mean(self._KLdiv)*lmbda_kl

        ## L1 loss image space
        self._recon_loss_y = lmbda_y*tf.reduce_mean(tf.abs(self._Y_input - self._output_gen))

        ## L1 loss latent space
        self._recon_loss_z = lmbda_z*tf.reduce_mean(tf.abs(self._Z_input - self._mean_layer_from_encoding))

        ## Adversarial loss
        if loss == "cross-entropy":
            self._logits_real = tf.math.log( self._output_adv_real / (1+eps - self._output_adv_real) + eps)
            self._logits_fake_from_real = tf.math.log( self._output_adv_fake_from_real / (1+eps - self._output_adv_fake_from_real) + eps)
            self._logits_fake_from_latent = tf.math.log( self._output_adv_fake_from_latent / (1+eps - self._output_adv_fake_from_latent) + eps)
            self._generator_loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.ones_like(self._logits_fake_from_real), logits=self._logits_fake_from_real
                                    )  +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=tf.ones_like(self._logits_fake_from_latent), logits=self._logits_fake_from_latent
                                    )
            )
            self._adversarial_loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_one(), logits=self._logits_real
                                    ) +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_zero(), logits=self._logits_fake_from_real
                                    )  +
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_zero(), logits=self._logits_fake_from_latent
                                    )
            )
        elif loss == "L2":
            self._generator_loss = tf.reduce_mean(
                tf.square(self._output_adv_fake_from_real - tf.ones_like(self._output_adv_fake_from_real)) +
                tf.square(self._output_adv_fake_from_latent - tf.ones_like(self._output_adv_fake_from_latent))
            ) / 2
            self._adversarial_loss = (
                tf.reduce_mean(
                    tf.square(self._output_adv_real - get_labels_one()) +
                    tf.square(self._output_adv_fake_from_real - get_labels_zero()) +
                    tf.square(self._output_adv_fake_from_latent - get_labels_zero())
                )
            ) / 3.0
        elif loss == "wasserstein":
            self._generator_loss = -tf.reduce_mean(self._output_adv_fake_from_real) - tf.reduce_mean(self._output_adv_fake_from_latent)
            self._adversarial_loss = (
                -(tf.reduce_mean(self._output_adv_real) -
                    tf.reduce_mean(self._output_adv_fake_from_real) -
                    tf.reduce_mean(self._output_adv_fake_from_latent)) +
                    10*self._define_gradient_penalty()
            )
        elif loss == "KL":
            self._logits_real = tf.math.log( self._output_adv_real / (1+eps - self._output_adv_real) + eps)
            self._logits_fake_from_real = tf.math.log( self._output_adv_fake_from_real / (1+eps - self._output_adv_fake_from_real) + eps)
            self._logits_fake_from_latent = tf.math.log( self._output_adv_fake_from_latent / (1+eps - self._output_adv_fake_from_latent) + eps)
            self._generator_loss = (-tf.reduce_mean(self._logits_fake_from_real) - tf.reduce_mean(self._logits_fake_from_latent))/2
            self._adversarial_loss = tf.reduce_mean(
                                    0.5*tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_one(), logits=self._logits_real
                                    ) +
                                    0.25*tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_zero(), logits=self._logits_fake_from_real
                                    )  +
                                    0.25*tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=get_labels_zero(), logits=self._logits_fake_from_latent
                                    )
            )
        else:
            raise ValueError("Loss not implemented. Choose from {}. Given: {}.".format(possible_losses, loss))


        if feature_matching:
            self._is_feature_matching = True
            otp_adv_real = self._adversarial.generate_net(self._input_real, tf_trainflag=self._is_training, return_idx=-2)
            otp_adv_fake = self._adversarial.generate_net(self._input_fake_from_real, tf_trainflag=self._is_training, return_idx=-2)
            self._generator_loss = tf.reduce_mean(tf.square(otp_adv_real - otp_adv_fake))

        with tf.name_scope("Loss") as scope:

            self._enc_loss = self._KLdiv + self._recon_loss_y + self._recon_loss_z + self._generator_loss
            self._gen_loss = self._recon_loss_y + self._recon_loss_z + self._generator_loss
            self._adv_loss = self._adversarial_loss

            tf.summary.scalar("Kullback-Leibler", self._KLdiv)
            tf.summary.scalar("Reconstruction_y", self._recon_loss_y)
            tf.summary.scalar("Reconstruction_z", self._recon_loss_z)
            tf.summary.scalar("Vanilla_Generator", self._generator_loss)

            tf.summary.scalar("Encoder", self._enc_loss)
            tf.summary.scalar("Generator", self._gen_loss)
            tf.summary.scalar("Adversarial", self._adv_loss)


    def _prepare_monitoring(self):
        self._total_train_time = 0
        self._total_log_time = 0
        self._count_batches = 0
        self._batches = []

        self._max_allowed_failed_checks = 20
        self._enc_grads_and_vars = self._enc_optimizer.compute_gradients(self._enc_loss, var_list=self._get_vars("Encoder"))
        self._gen_grads_and_vars = self._gen_optimizer.compute_gradients(self._gen_loss, var_list=self._get_vars("Generator"))
        self._adv_grads_and_vars = self._adv_optimizer.compute_gradients(self._adv_loss, var_list=self._get_vars("Adversarial"))

        self._monitor_dict = {
            "Gradients": [
                [self._enc_grads_and_vars, self._gen_grads_and_vars, self._adv_grads_and_vars],
                ["Encoder", "Generator", "Adversarial"],
                [[] for i in range(9)]
            ],
            "Losses": [
                    [self._enc_loss, self._gen_loss, self._adversarial_loss, self._generator_loss,
                    self._recon_loss_y, self._KLdiv, self._recon_loss_z],
                    ["Encoder (V+R_y+R_z+K)", "Generator (V+R_y+R_z)", "Adversarial", "Vanilla_Generator",
                    "Reconstruction_y", "Kullback-Leibler", "Reconstruction_z"],
                    [[] for i in range(7)]
            ],
            "Output Adversarial": [
                    [self._output_adv_fake_from_real, self._output_adv_fake_from_latent, self._output_adv_real],
                    ["Fake_from_real", "Fake_from_latent", "Real"],
                    [[] for i in range(3)],
                    [np.mean]
            ]
        }

        self._check_dict = {
            "Dominating Discriminator": {
                "Tensors": [self._output_adv_real, self._output_adv_fake_from_real],
                "OPonTensors": [np.mean, np.mean],
                "Relation": [">", "<"], "Threshold": [self._label_smoothing*0.95, (1-self._label_smoothing)*1.05],
                "TensorRelation": np.logical_and
            },
            "Generator outputs zeros": {
                "Tensors": [self._output_gen_from_encoding],
                "OPonTensors": [np.max],
                "Relation": ["<"], "Threshold": [0.05]
            }
        }
        self._check_count = [0 for key in self._check_dict]



if __name__ == '__main__':
    ########### Image input
    if "lhcb_data" in os.getcwd():
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        gpu_frac = 0.2
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
        print("1 GPU limited to {}% memory.".format(np.round(gpu_frac*100)))
    else:
        gpu_options = None

    import pickle
    if "lhcb_data" in os.getcwd():
        data_path = "../../../Data/fashion_mnist"
    else:
        data_path = "/home/tneuer/Backup/Algorithmen/0TestData/image_to_image/fashion_mnist"
    with open("{}/train_images.pickle".format(data_path), "rb") as f:
        x_train_orig = (pickle.load(f)[0].reshape(-1, 28, 28, 1) / 255)[:20000]
        x_test_orig = x_train_orig[-100:]
        x_train_orig = x_train_orig[:-100]
    with open("{}/train_images_rotated.pickle".format(data_path), "rb") as f:
        y_train_orig = (pickle.load(f).reshape(-1, 28, 28, 1) / 255)[:20000]
        y_test_orig = y_train_orig[-100:]
        y_train_orig = y_train_orig[:-100]


    ########### Architecture

    x_train_orig = padding_zeros(x_train_orig, top=2, bottom=2, left=2, right=2)
    x_test_orig = padding_zeros(x_test_orig, top=2, bottom=2, left=2, right=2)
    y_train_orig = padding_zeros(y_train_orig, top=2, bottom=2, left=2, right=2)
    y_test_orig = padding_zeros(y_test_orig, top=2, bottom=2, left=2, right=2)
    inpt_dim = x_train_orig[0].shape
    opt_dim = y_train_orig[0].shape

    param_dict = {
            "adv_steps": [1],
            "architecture": ["keraslike"],
            "batch_size": [8],
            "feature_matching": [True, False],
            "gen_steps": [1],
            "is_patchgan": [True],
            "invert_images": [True, False],
            "lmbda_kl": [0.1, 1, 5],
            "lmbda_y": [0.1, 1, 2],
            "lmbda_z": [0.1, 1, 5],
            "loss": ["KL", "L2"],
            "lr": [0.0001, 0.00005],
            "lr_adv": [0.000005],
            "optimizer": [tf.train.AdamOptimizer],
            "random_labeling": [0.01, 0.05, 0.1],
            "z_dim": [32, 64],
    }
    sampled_params = grid_search.get_parameter_grid(param_dict=param_dict, n=50, allow_repetition=True)

    for params in sampled_params:

        adv_steps = int(params["adv_steps"])
        architecture_path = "../../../Architectures/Im2Im/CVAEGAN/{}.json".format(params["architecture"])
        batch_size = int(params["batch_size"])
        learning_rate = float(params["lr"])
        learning_rate_adv = float(params["lr_adv"])
        z_dim = int(params["z_dim"])
        lmbda_kl = int(params["lmbda_kl"])
        lmbda_y = int(params["lmbda_y"])
        lmbda_z = int(params["lmbda_z"])
        optimizer = params["optimizer"]
        feature_matching = bool(params["feature_matching"])
        loss = str(params["loss"])
        is_patchGAN = bool(params["is_patchgan"])
        invert_images = bool(params["invert_images"])
        random_labeling = float(params["random_labeling"])

        x_train = x_train_orig
        x_test = x_test_orig
        y_train = y_train_orig
        y_test = y_test_orig
        if invert_images:
            x_train = 1-x_train_orig
            x_test = 1-x_test_orig
            y_train = 1-y_train_orig
            y_test = 1-y_test_orig

        epochs = 15
        batch_log_step = int(len(x_train) / batch_size / 20)
        gen_steps = 1
        is_wasserstein = loss == "wasserstein"
        label_smoothing = 0.95

        architectures = GenerativeModel.load_from_json(architecture_path)
        enc_architecture = architectures["Encoder"]
        gen_architecture = architectures["Generator"]
        adversarial_architecture = architectures["Adversarial"]

        if is_patchGAN and is_wasserstein:
            adversarial_architecture[-1][1]["activation"] = tf.identity
            adversarial_architecture[-1][1]["filters"] = 1
        elif is_patchGAN and not is_wasserstein:
            adversarial_architecture[-1][1]["activation"] = tf.nn.sigmoid
            adversarial_architecture[-1][1]["filters"] = 1
        elif not is_patchGAN:
            adversarial_architecture[-1][1]["activation"] = tf.nn.leaky_relu


        path_saving = "../../../Results/Test"
        path_saving = init.initialize_folder(algorithm="BiCycleGAN_", base_folder=path_saving)
        print(path_saving)

        init_params = {
            "x_dim": inpt_dim, "y_dim": opt_dim, "z_dim": z_dim, "enc_architecture": enc_architecture,
            "gen_architecture": gen_architecture, "adversarial_architecture": adversarial_architecture, "folder": path_saving,
            "is_patchgan": is_patchGAN, "is_wasserstein": is_wasserstein
        }
        compile_params = {
            "loss": loss, "learning_rate": learning_rate, "learning_rate_adv": learning_rate_adv,
            "optimizer": optimizer, "lmbda_kl": lmbda_kl, "lmbda_y": lmbda_y, "lmbda_z": lmbda_z,
            "feature_matching": feature_matching, "label_smoothing": label_smoothing, "random_labeling": random_labeling

        }
        train_params = {
            "x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test":y_test, "epochs": epochs, "batch_size": batch_size,
            "adv_steps": adv_steps, "gen_steps": gen_steps, "log_step": 1, "batch_log_step": batch_log_step,
            "gpu_options": gpu_options
        }

        config_data = copy.deepcopy(init_params); config_data.update(compile_params); config_data.update(train_params)
        config_data["nr_train"] = len(x_train)
        config_data["nr_test"] = len(x_test)
        config_data["invert_images"] = invert_images
        config_data["architecture"] = params["architecture"]
        config_data["optimizer"] = config_data["optimizer"].__name__
        for key in ["x_train", "y_train", "x_test", "y_test", "gpu_options"]:
            config_data.pop(key)
        config_data = init.function_to_string(config_data)

        grid_search.run_agorithm(BiCycleGAN, init_params=init_params, compile_params=compile_params, train_params=train_params,
                                 path_saving=path_saving, config_data=config_data)
        tf.reset_default_graph()


