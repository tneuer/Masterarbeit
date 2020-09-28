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
if "lhcb_data2" in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.insert(1, "Preprocessing")
sys.path.insert(1, "TFModels")
sys.path.insert(1, "TFModels/building_blocks")
sys.path.insert(1, "Utilities")
import json
import pickle
import grid_search

import numpy as np
import tensorflow as tf
if "lhcb_data2" in os.getcwd():
    gpu_fraction = 0.3
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    print("1 GPU limited to {}% memory.".format(round(gpu_fraction*100)))
else:
    gpu_options = None

from TFModels.PGAN import create_algorithm
from TFModels.CycleGAN_OO import CycleGAN
from TFModels.Pix2Pix_OO import Pix2PixGAN
import Preprocessing.initialization as init
from TFModels.building_blocks.layers import logged_dense, conv2d_logged, conv2d_transpose_logged, reshape_layer
from TFModels.building_blocks.layers import residual_block, unet, unet_original, inception_block
from functionsOnImages import padding_zeros
from generativeModels import GenerativeModel


############################################################################################################
# Parameter definiton
############################################################################################################
param_dict = {
            "optimizer": [tf.train.AdamOptimizer],
            "batch_size": [1],
            "lmbda": [5, 10, 50, 100],
            "learning_rate": [0.0002],
            "loss": ["wasserstein", "L1", "cross-entropy"],
            "is_patchgan": [True],
            "gen_steps": [1, 5],
            "shuffle": [False],
            "algorithm": [Pix2PixGAN],
            "architecture": ["dense"]
}
sampled_params = grid_search.get_parameter_grid(param_dict=param_dict, n=30, allow_repetition=True)

for i, params in enumerate(sampled_params):

    ########### Architecturea
    activation = tf.nn.leaky_relu
    optimizer = params["optimizer"]
    learning_rate = float(params["learning_rate"])
    epochs = 29
    batch_size = int(params["batch_size"])
    padding1 = {"top": 4, "bottom": 4, "left":0, "right":0}
    padding2 = {"top": 6, "bottom": 6, "left":0, "right":0}
    samplesize = "Debug"
    loss = str(params["loss"])
    is_patchgan = bool(params["is_patchgan"])
    lmbda = float(params["lmbda"])
    disc_steps = 1
    gen_steps = int(params["gen_steps"])
    shuffle = bool(params["shuffle"])
    algorithm = params["algorithm"].__name__
    log_per_epoch = 10
    arch = params["architecture"]
    architecture_path = "../Architectures/Pix2Pix/{}.json".format(arch)
    print(arch, loss)
    # architecture_path = "../Architectures/Pix2Pix/unet{}.json".format(depth)

    is_wasserstein = True if loss == "wasserstein" else False

    architectures = GenerativeModel.load_from_json(architecture_path)
    gen_xy_architecture = architectures["Generator"]
    disc_xy_architecture = architectures["Discriminator"]
    if is_patchgan:
        if is_wasserstein:
            disc_xy_architecture[-1][1]["activation"] = tf.identity
        else:
            disc_xy_architecture[-1][1]["activation"] = tf.nn.sigmoid
    else:
        disc_xy_architecture[-1][1]["activation"] = tf.nn.leaky_relu
        disc_xy_architecture.append([tf.layers.flatten, {}])
        disc_xy_architecture.append([tf.layers.dense, {"units": 128, "activation": activation}])

    if "lhcb_data2" in os.getcwd():
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
    train_x = padding_zeros(train_x, **padding1)
    train_y = padding_zeros(train_y, **padding2).reshape([-1, *image_shape])

    # if "lhcb_data2" in os.getcwd():
    #     data_path = "../Data/fashion_mnist"
    # else:
    #     data_path = "/home/tneuer/Backup/Algorithmen/0TestData/image_to_image/mnist"
    # with open("{}/train_images.pickle".format(data_path), "rb") as f:
    #     train_x = pickle.load(f)[0].reshape(-1, 28, 28, 1) / 255
    # with open("{}/train_images_rotated.pickle".format(data_path), "rb") as f:
    #     train_y = pickle.load(f).reshape(-1, 28, 28, 1) / 255
    # image_shape = [32, 32, 1]
    # padding_val = 2
    # train_x = padding_zeros(train_x, top=padding_val, bottom=padding_val, left=padding_val, right=padding_val).reshape([-1, *image_shape])
    # train_y = padding_zeros(train_y, top=padding_val, bottom=padding_val, left=padding_val, right=padding_val).reshape([-1, *image_shape])



    energy_scaler = np.max(train_y)
    train_x = np.clip(train_x, a_min=0, a_max=energy_scaler)
    train_x /= energy_scaler
    train_y /= energy_scaler

    nr_test = int(min(0.1*len(train_x), 100))

    test_x = train_x[-nr_test:]
    train_x = train_x[:-nr_test]
    test_y = train_y[-nr_test:]
    train_y = train_y[:-nr_test]

    nr_train = train_x.shape[0]
    nr_test = test_x.shape[0]

    logging_calo = test_y[:100]

    print(train_x.shape)
    print(train_y.shape)
    print(np.max(train_x))
    print(np.max(train_y))

    nr_train = len(train_x)
    nr_test = len(test_x)


    ############################################################################################################
    # Preparation
    ############################################################################################################
    if not os.path.exists(path_results):
        os.mkdir(path_results)
    path_saving = init.initialize_folder(algorithm=algorithm, base_folder=path_results)

    def prepare_algorithm(network, optimizer, learning_rate):
        network.compile(optimizer=optimizer, learning_rate=learning_rate, lmbda=lmbda, loss=loss)
        # network.set_attributes(keep_cols)
        post_message = """\nCalo shape:
                        \nAppend attributes at every layer: {}""".format(train_y.shape, False)
        network.log_architecture(post_message=post_message)

        nr_params = network.get_number_params()
        sampler = network.get_sampling_distribution()
        config_data.update({"nr_params": nr_params, "sampler": sampler, "optimizer": optimizer.__name__})
        if algorithm == "CycleGAN":
            config_data.update({"generator_out": network._generator_xy._output_layer.name,
                               "generator_out_yx": network._generator_yx._output_layer.name
                               })
        elif algorithm == "Pix2PixGAN":
            config_data.update({"generator_out": network._generator._output_layer.name})
        else:
            raise ValueError("Algorithm not implemented.")

        with open(path_saving+"/config.json", "w") as f:
            json.dump(config_data, f, indent=4)

    nr_batches = (nr_train / batch_size)
    log_batch_step = int(1/log_per_epoch*nr_batches)

    config_data = init.create_config_file(globals())


    ############################################################################################################
    # Model Training
    ############################################################################################################
    inpt_dim = train_x[0].shape
    opt_dim = train_y[0].shape

    try:
        if algorithm == "CycleGAN":
            network = CycleGAN(x_dim=inpt_dim, y_dim=opt_dim, gen_xy_architecture=gen_xy_architecture,
                            disc_xy_architecture=disc_xy_architecture, gen_yx_architecture=gen_yx_architecture,
                            disc_yx_architecture=disc_yx_architecture, PatchGAN=is_patchgan,
                            folder=path_saving, is_wasserstein=is_wasserstein)
            prepare_algorithm(network, optimizer, learning_rate)
            network.train(x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y,
                        epochs=epochs, batch_size=batch_size, disc_steps=disc_steps, gen_steps=gen_steps, log_step=1, gpu_options=gpu_options,
                        shuffle=shuffle)
        elif algorithm == "Pix2PixGAN":
            network = Pix2PixGAN(x_dim=inpt_dim, y_dim=opt_dim, gen_architecture=gen_xy_architecture,
                            disc_architecture=disc_xy_architecture, PatchGAN=is_patchgan,
                            folder=path_saving, is_wasserstein=is_wasserstein)
            prepare_algorithm(network, optimizer, learning_rate)
            network.train(x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y,
                        epochs=epochs, batch_size=batch_size, disc_steps=disc_steps, gen_steps=gen_steps, log_step=1, gpu_options=gpu_options,
                        batch_log_step=log_batch_step)
        else:
            raise ValueError("Algorithm not implemented.")
        with open(path_saving+"/EXIT_FLAG0.txt", "w") as f:
            f.write("EXIT STATUS: 0. No errors or warnings.")
        tf.reset_default_graph()

    except GeneratorExit as e:
        with open(path_saving+"/EXIT_FLAG1.txt", "w") as f:
            f.write("EXIT STATUS: 1. {}.".format(e))
        tf.reset_default_graph()