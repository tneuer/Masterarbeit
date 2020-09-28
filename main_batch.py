#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-18 14:45:06
    # Description :
####################################################################################
"""
import os
if "lhcb_data2" in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    print("1 GPU limited to 25% memory.")
else:
    gpu_options = None
import Preprocessing.initialization as init

from shutil import copyfile
from functionsOnImages import padding_zeros
from Utilities.create_index import create_index
from TFModels.PGAN import create_algorithm
from TFModels.building_blocks.layers import logged_dense, reshape_layer, sample_vector_layer, replicate_vector_layer


############################################################################################################
# Parameter definiton
############################################################################################################
param_dict = {
            "z_dim": [64],
            "rotate": [True],
}
sampled_params = grid_search.get_parameter_grid(param_dict=param_dict, n=1, allow_repetition=False)

for params in sampled_params:
param_dict = {
            "z_dim": [32, 64, 128],
            "optimizer": [tf.train.AdamOptimizer, tf.train.RMSPropOptimizer],
}
sampled_params = grid_search.get_parameter_grid(param_dict=param_dict, n=5, allow_repetition=True)

for params in sampled_params:

    activation = tf.nn.leaky_relu
    add_last_conv = False
    algorithm = "CVAE"
    append_y = False

    batchnorm = False
    batch_size = 64
    dropout = False
    epochs = 1
    filters_gen = [int(512/2**i) for i in range(3)]

    image_flatten = False
    image_scaling = True
    if image_scaling is True:
      activation_last_layer = tf.nn.sigmoid
    else:
      activation_last_layer = tf.nn.relu

    keep_cols = ["x_projections", "y_projections", "momentum_p"]

    layer = tf.layers.dense
    logging_seed = 42
    logging_size = 10

    particle = "piplus"
    path_loading = "../Data/Piplus/Debug"
    path_results = "../Results/Piplus"
    penalize_cells = False

    reshape_z = "none"
    rotate = False

    steps_adv = 6
    steps_gen = 1
    steps_log = 3

    test_seed = 42
    test_size = 0.1

    y_dim = len(keep_cols)
    z_dim = int(params["z_dim"])
    rm_adv_layers = 1
    optimizer = params["optimizer"]

    ############################################################################################################
    # Network initialization
    ############################################################################################################
    if image_flatten:
        padding = "none"
        image_shape = [52, 64, 1]
        x_dim = int(np.prod(image_shape))
        architecture_gen = [
                            [layer, {"units": 128, "activation": activation}],
                            [layer, {"units": 256, "activation": activation}],
                            [layer, {"units": 512, "activation": activation}],
                            [layer, {"units": 1024, "activation": activation_last_layer}],
                            ]
        architecture_adv = [
                            [layer, {"units": 1024, "activation": activation}],
                            [layer, {"units": 512, "activation": activation}],
                            [layer, {"units": 256, "activation": activation}],
                            [layer, {"units": 128, "activation": activation}],
                            ]
    else:
        if rotate:
            padding = {"top": 6, "bottom": 6, "left":0, "right":0}
            initial_size = [8, 8]
        else:
            padding = {"top": 2, "bottom": 2, "left":0, "right":0}
            initial_size = [7, 8]
        image_shape = [52+padding["top"]+padding["bottom"], 64+padding["left"]+padding["right"], 1]
        x_dim = image_shape

        architecture_adv = [
                      [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": activation}],
                      [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": activation}],
                      [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": activation}],
                      ]

        if reshape_z == "none":
            architecture_gen = [
                          [logged_dense, {"units": int(np.prod(initial_size))*filters_gen[0], "activation": activation}],
                          [reshape_layer, {"shape": [*initial_size, filters_gen[0]]}],
                          [tf.layers.conv2d_transpose, {"filters": filters_gen[1], "kernel_size": 2, "strides": 2, "activation": activation}],
                          [tf.layers.conv2d_transpose, {"filters": filters_gen[2], "kernel_size": 2, "strides": 2, "activation": activation}],
                          [tf.layers.conv2d_transpose, {"filters": 1, "kernel_size": 2, "strides": 2, "activation": activation_last_layer}]
                          ]
        elif reshape_z == "replicate":
            architecture_gen = [
                          [replicate_vector_layer, {"size": initial_size}],
                          [tf.layers.conv2d_transpose, {"filters": filters_gen[1], "kernel_size": 2, "strides": 2, "activation": activation}],
                          [tf.layers.conv2d_transpose, {"filters": filters_gen[2], "kernel_size": 2, "strides": 2, "activation": activation}],
                          [tf.layers.conv2d_transpose, {"filters": 1, "kernel_size": 2, "strides": 2, "activation": activation_last_layer}]
                          ]
        elif reshape_z == "sample":
            architecture_gen = [
                          [sample_vector_layer, {"size": initial_size, "y_dim": len(keep_cols), "rfunc": sampling_distribution[0], "rparams": sampling_distribution[1]}],
                          [tf.layers.conv2d_transpose, {"filters": filters_gen[1], "kernel_size": 2, "strides": 2, "activation": activation}],
                          [tf.layers.conv2d_transpose, {"filters": filters_gen[2], "kernel_size": 2, "strides": 2, "activation": activation}],
                          [tf.layers.conv2d_transpose, {"filters": 1, "kernel_size": 2, "strides": 2, "activation": activation_last_layer}]
                          ]
        else:
            raise NotImplementedError("Wrong reshape_z method.")

    if add_last_conv:
        architecture_gen = architecture_gen[:-1]
        architecture_gen.append([tf.layers.conv2d_transpose, {"filters": 32, "kernel_size": 2, "strides": 2, "activation": activation}])
        architecture_gen.append([tf.layers.conv2d, {"filters": 1, "kernel_size": 3, "strides": 1, "activation": activation_last_layer, "padding": "SAME"}])

    if rm_adv_layers:
        architecture_adv = architecture_adv[:-rm_adv_layers]

    if dropout and batchnorm:
        len_adv = len(architecture_adv)
        [(architecture_adv.insert(1+3*i, [tf.layers.batch_normalization, {}]), architecture_adv.insert(2+3*i, [tf.layers.dropout, {}]))  for i in range(len_adv-1)]
    elif batchnorm:
        len_adv = len(architecture_adv)
        [architecture_adv.insert(1+2*i, [tf.layers.batch_normalization, {}]) for i in range(len_adv-1)]
    elif dropout:
        len_adv = len(architecture_adv)
        [architecture_adv.insert(1+2*i, [tf.layers.dropout, {}]) for i in range(len_adv-1)]


    architecture_aux = [
                      [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": activation}],
                      [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": activation}],
                      [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": activation}],
                      [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": activation}],
                      ]

    ############################################################################################################
    # Data loading
    ############################################################################################################
    path_saving = init.initialize_folder(algorithm=algorithm, base_folder=path_results)
    path_loading += "/Batches"
    copyfile(path_loading+"/Scalers.pickle", path_saving+"/Scalers.pickle")
    with open(path_loading+"/BatchX_Logging.pickle", "rb") as f:
        logging_calo = pickle.load(f)
    with open(path_loading+"/BatchY_Logging.pickle", "rb") as f:
        logging_tracker = pickle.load(f)
    with open(path_loading+"/BatchX_Test.pickle", "rb") as f:
        test_calo = pickle.load(f)
    with open(path_loading+"/BatchY_Test.pickle", "rb") as f:
        test_tracker = pickle.load(f)

    path_x_batches = path_loading + "/BatchesX"
    path_y_batches = path_loading + "/BatchesY"

    nr_test = test_calo.shape[0]

    print(np.max(test_calo), np.max(logging_calo))
    print(test_calo.shape, logging_calo.shape)


    ############################################################################################################
    # Preparation
    ############################################################################################################
    def prepare_algorithm(network):
        global optimizer
        network.compile(logged_labels=logging_tracker, logged_images=logging_calo, optimizer=optimizer)
        network.set_attributes(keep_cols)
        post_message = """\nCalo shape: {}\nTracker shape: {}
                        \nUsed attributes: {}
                        \nAppend attributes at every layer: {}""".format(test_calo.shape, test_tracker.shape, keep_cols, append_y)
        network.log_architecture(post_message=post_message)

        nr_params = network.get_number_params()
        sampler = network.get_sampling_distribution()
        config_data.update({"nr_params": nr_params, "sampler": sampler, "generator_out": network._generator._output_layer.name, "optimizer": optimizer.__name__})

        with open(path_saving+"/config.json", "w") as f:
            json.dump(config_data, f, indent=4)

    config_data = init.create_config_file(globals())

    ############################################################################################################
    # Model Training
    ############################################################################################################


    network = create_algorithm(algorithm, penalize_cells=penalize_cells, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim,
                                 gen_architecture=architecture_gen, adv_architecture=architecture_adv,
                                 folder=path_saving, image_shape=image_shape, append_y_at_every_layer=append_y,
                                 last_layer_activation = activation_last_layer)

    prepare_algorithm(network)
    network.train(batch_x_path=path_x_batches, batch_y_path=path_y_batches, x_test=test_calo, y_test=test_tracker,
                epochs=epochs, batch_size=batch_size, steps=steps_gen, log_step=steps_log, gpu_options=gpu_options,
                preprocess_func=preprocess_func, return_array=True)

    tf.reset_default_graph()
    with open(path_saving+"/EXIT_FLAG.txt", "w") as f:
        f.write("EXIT STATUS: 0. No errors or warnings.")
