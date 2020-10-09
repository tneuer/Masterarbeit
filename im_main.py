#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-06-24 16:28:06
    # Description :
####################################################################################
"""
import os
import copy
if "lhcb_data" in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.insert(1, "Preprocessing")
sys.path.insert(1, "TFModels")
sys.path.insert(1, "TFModels/Im2Im")
sys.path.insert(1, "TFModels/building_blocks")
sys.path.insert(1, "Utilities")
import json
import pickle
import grid_search

import numpy as np
import tensorflow as tf
if "lhcb_data" in os.getcwd():
    gpu_fraction = 0.3
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    print("1 GPU limited to {}% memory.".format(round(gpu_fraction*100)))
else:
    gpu_options = None

from Im2Im.CVAEGAN_OO import CVAEGAN
from Im2Im.BiCycleGAN_OO import BiCycleGAN

import Preprocessing.initialization as init
from TFModels.building_blocks.layers import logged_dense, conv2d_logged, conv2d_transpose_logged, reshape_layer
from TFModels.building_blocks.layers import residual_block, unet, unet_original, inception_block
from functionsOnImages import padding_zeros
from generativeModels import GenerativeModel


############################################################################################################
# Parameter definiton
############################################################################################################

param_dict = {
        "adv_steps": [1],
        "algorithm": [BiCycleGAN],
        "architecture": ["keraslike_residual_VGG"],
        "batch_size": [4],
        "feature_matching": [False],
        "gen_steps": [1],
        "is_patchgan": [True],
        "invert_images": [False],
        "label_smoothing": [0.9],
        "lmbda_kl": [0.1, 1],
        "lmbda_y": [0.5],
        "lmbda_z": [1],
        "loss": ["cross-entropy"],
        "learning_rate": [0.00005],
        "learning_rate_adv": [0.000005],
        "optimizer": [tf.train.AdamOptimizer],
        "random_labeling": [0],
        "z_dim": [64],
}
sampled_params = grid_search.get_parameter_grid(param_dict=param_dict, n=50, allow_repetition=True)

for i, params in enumerate(sampled_params):

    ########### Architecturea
    architecture = str(params["architecture"])
    adv_steps = int(params["adv_steps"])
    algorithm = params["algorithm"]
    algorithm_name = algorithm.__name__
    architecture_path = "../Architectures/Im2Im/CVAEGAN/{}.json".format(architecture)
    batch_size = int(params["batch_size"])
    learning_rate = float(params["learning_rate"])
    learning_rate_adv = float(params["learning_rate_adv"])
    label_smoothing = float(params["label_smoothing"])
    lmbda_kl = float(params["lmbda_kl"])
    lmbda_y = float(params["lmbda_y"])
    lmbda_z = float(params["lmbda_z"])
    optimizer = params["optimizer"]
    feature_matching = bool(params["feature_matching"])
    gen_steps = int(params["gen_steps"])
    loss = str(params["loss"])
    is_patchgan = bool(params["is_patchgan"])
    invert_images = bool(params["invert_images"])
    random_labeling = float(params["random_labeling"])
    z_dim = int(params["z_dim"])

    epochs = 3
    log_per_epoch = 40
    padding1 = {"top": 4, "bottom": 4, "left":0, "right":0}
    padding2 = {"top": 6, "bottom": 6, "left":0, "right":0}
    is_wasserstein = True if loss == "wasserstein" else False

    architectures = GenerativeModel.load_from_json(architecture_path)
    enc_architecture = architectures["Encoder"]
    gen_architecture = architectures["Generator"]
    adv_architecture = architectures["Adversarial"]
    if is_patchgan and is_wasserstein:
        adv_architecture[-1][1]["activation"] = tf.identity
        adv_architecture[-1][1]["filters"] = 1
    elif is_patchgan and not is_wasserstein:
        adv_architecture[-1][1]["activation"] = tf.nn.sigmoid
        adv_architecture[-1][1]["filters"] = 1
    elif not is_patchgan:
        adv_architecture[-1][1]["activation"] = tf.nn.leaky_relu

    if "lhcb_data" in os.getcwd():
        path_loading = "../Data/B2Dmunu/LargeSample"
        path_results = "../Results/B2Dmunu"
    else:
        path_loading = "../Data/B2Dmunu/Debug"
        path_results = "../Results/Test/B2Dmunu"

    ############################################################################################################
    # Data loading
    ############################################################################################################
    with open("{}/Trained/PiplusLowerP_CWGANGP8_out_1.pickle".format(path_loading), "rb") as f:
        train_x = pickle.load(f)

    with open("{}/calo_images.pickle".format(path_loading), "rb") as f:
        train_y = pickle.load(f)
    image_shape = [64, 64, 1]
    train_x = padding_zeros(train_x, **padding1)[:50000]
    train_y = padding_zeros(train_y, **padding2).reshape([-1, *image_shape])[:50000]

    energy_scaler = np.max(train_y)
    train_x = np.clip(train_x, a_min=0, a_max=energy_scaler)
    train_x /= energy_scaler
    train_y /= energy_scaler

    nr_test = int(min(0.1*len(train_x), 100))

    test_x = train_x[-nr_test:]
    train_x = train_x[:-nr_test]
    test_y = train_y[-nr_test:]
    train_y = train_y[:-nr_test]

    nr_train = len(train_x)
    nr_test = len(test_x)
    inpt_dim = train_x[0].shape
    opt_dim = train_y[0].shape
    nr_batches = (nr_train / batch_size)
    batch_log_step = max(1, int(1/log_per_epoch*nr_batches))

    if invert_images:
        train_x = 1 - train_x
        train_y = 1 - train_y
        test_x = 1 - test_x
        test_y = 1 - test_y

    assert np.max(train_x) == 1, "train_x maximum is not 1, but {}.".format(np.max(train_x))
    assert np.max(train_y) == 1, "train_y maximum is not 1, but {}.".format(np.max(train_y))
    assert np.max(test_x) <= 1, "test_x maximum is greater 1: {}.".format(np.max(test_x))
    assert np.max(test_y) <= 1, "test_y maximum is greater 1: {}.".format(np.max(test_y))

    print(train_x.shape, train_y.shape, test_x.shape, test_x.shape)
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(nrows=2, ncols=2)
    # from functionsOnImages import get_energies, get_number_of_activated_cells
    # axs[0, 0].hist([get_energies(train_x), get_energies(train_y)], label=["GAN", "MC"])
    # axs[0, 0].legend()
    # axs[0, 1].hist([get_energies(test_x), get_energies(test_y)], label=["GAN", "MC"])
    # axs[0, 1].legend()
    # axs[1, 0].hist([get_number_of_activated_cells(train_x.reshape(-1, 64, 64), threshold=6/6120),
    #                get_number_of_activated_cells(train_y.reshape(-1, 64, 64), threshold=6/6120)], label=["GAN", "MC"])
    # axs[1, 0].legend()
    # axs[1, 1].hist([get_number_of_activated_cells(test_x.reshape(-1, 64, 64), threshold=6/6120),
    #                get_number_of_activated_cells(test_y.reshape(-1, 64, 64), threshold=6/6120)], label=["GAN", "MC"])
    # axs[1, 1].legend()

    # n = 10
    # fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(20, 12))
    # for i in range(n):
    #     axs[i, 0].imshow(train_x[i].reshape(64, 64))
    #     axs[i, 1].imshow(train_y[i].reshape(64, 64))
    # plt.show()


    ############################################################################################################
    # Preparation
    ############################################################################################################
    if not os.path.exists(path_results):
        os.mkdir(path_results)
    path_saving = init.initialize_folder(algorithm=algorithm_name, base_folder=path_results)

    ############################################################################################################
    # Model Training
    ############################################################################################################
    if algorithm_name == "CycleGAN":
        raise
    elif algorithm_name == "Pix2PixGAN":
        raise
    elif algorithm_name == "CVAEGAN":
        init_params = {
            "x_dim": inpt_dim, "y_dim": opt_dim, "z_dim": z_dim, "enc_architecture": enc_architecture,
            "gen_architecture": gen_architecture, "adversarial_architecture": adv_architecture, "folder": path_saving,
            "is_patchgan": is_patchgan, "is_wasserstein": is_wasserstein
        }
        compile_params = {
            "loss": loss, "learning_rate": learning_rate, "learning_rate_adv": learning_rate_adv,
            "optimizer": optimizer, "lmbda_kl": lmbda_kl, "lmbda_y": lmbda_y, "feature_matching": feature_matching,
            "label_smoothing": label_smoothing, "random_labeling": random_labeling

        }
        train_params = {
            "x_train": train_x, "y_train": train_y, "x_test": test_x, "y_test": test_y, "epochs": epochs, "batch_size": batch_size,
            "adv_steps": adv_steps, "gen_steps": gen_steps, "log_step": 1, "batch_log_step": batch_log_step,
            "gpu_options": gpu_options
        }
    elif algorithm_name == "BiCycleGAN":
        init_params = {
            "x_dim": inpt_dim, "y_dim": opt_dim, "z_dim": z_dim, "enc_architecture": enc_architecture,
            "gen_architecture": gen_architecture, "adversarial_architecture": adv_architecture, "folder": path_saving,
            "is_patchgan": is_patchgan, "is_wasserstein": is_wasserstein
        }
        compile_params = {
            "loss": loss, "learning_rate": learning_rate, "learning_rate_adv": learning_rate_adv,
            "optimizer": optimizer, "lmbda_kl": lmbda_kl, "lmbda_y": lmbda_y, "lmbda_z": lmbda_z,
            "feature_matching": feature_matching, "label_smoothing": label_smoothing, "random_labeling": random_labeling

        }
        train_params = {
            "x_train": train_x, "y_train": train_y, "x_test": test_x, "y_test": test_y, "epochs": epochs, "batch_size": batch_size,
            "adv_steps": adv_steps, "gen_steps": gen_steps, "log_step": 1, "batch_log_step": batch_log_step,
            "gpu_options": gpu_options
        }
    else:
        raise ValueError("Algorithm not implemented.")

    config_data = copy.deepcopy(init_params); config_data.update(compile_params); config_data.update(train_params)
    config_data["nr_train"] = len(train_x)
    config_data["nr_test"] = len(test_x)
    config_data["invert_images"] = invert_images
    config_data["architecture"] = params["architecture"]
    config_data["optimizer"] = config_data["optimizer"].__name__
    config_data["algorithm"] = algorithm_name
    config_data["padding1"] = padding1; config_data["padding2"] = padding2
    for key in ["x_train", "y_train", "x_test", "y_test", "gpu_options"]:
        config_data.pop(key)
    config_data = init.function_to_string(config_data)

    grid_search.run_agorithm(algorithm, init_params=init_params, compile_params=compile_params, train_params=train_params,
                             path_saving=path_saving, config_data=config_data)
    tf.reset_default_graph()
