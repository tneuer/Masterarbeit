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
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.insert(1, "Preprocessing")
sys.path.insert(1, "TFModels")
sys.path.insert(1, "TFModels/building_blocks")
sys.path.insert(1, "TFModels/GAN")
sys.path.insert(1, "TFModels/CGAN")
sys.path.insert(1, "TFModels/CGAN/OLD")
sys.path.insert(1, "Utilities")
import json
import grid_search

import numpy as np
import tensorflow as tf
if "lhcb_data2" in os.getcwd():
    gpu_frac = 0.3
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    print("1 GPU limited to {}% memory.".format(np.round(gpu_frac*100)))
else:
    gpu_options = None

from TFModels.PGAN import create_algorithm
import Preprocessing.initialization as init
from building_blocks.layers import logged_dense, conv2d_logged, conv2d_transpose_logged
from building_blocks.layers import reshape_layer, sample_vector_layer, replicate_vector_layer
from building_blocks.layers import logged_dense, conv2d_logged, conv2d_transpose_logged, residual_block, unet, unet_original, inception_block
from functionsOnImages import padding_zeros
from generativeModels import GenerativeModel


############################################################################################################
# Parameter definiton
############################################################################################################
param_dict = {
            "z_dim": [32, 64],
            "optimizer": [tf.train.AdamOptimizer, tf.train.RMSPropOptimizer],
            "algorithm": ["CGAN"],
            "dataset": ["PiplusLowerP"],
            "gen_steps": [1],
            "adv_steps": [5, 1],
            # "architecture": ["more_unbalanced"],
            "architecture": ["unbalanced2", "unbalanced", "unbalanced5", "unbalanced6"],
            "is_patchgan": [True, False],
            "batch_size": [8],
            "loss": ["cross-entropy", "KL"],
            "cc": [False],
            "lr": [0.001],
            "feature_matching": [False],
            "label_smoothing": [0.8, 0.9, 1]
}
sampled_params = grid_search.get_parameter_grid(param_dict=param_dict, n=30, allow_repetition=True)

for params in sampled_params:

    activation = tf.nn.leaky_relu
    algorithm = str(params["algorithm"])
    append_y = False
    architecture = str(params["architecture"])
    architecture_path = "../Architectures/CGAN/{}.json".format(architecture)
    is_patchgan = bool(params["is_patchgan"])
    loss = str(params["loss"])
    is_wasserstein = loss == "wasserstein"
    is_cycle_consistent = bool(params["cc"])
    label_smoothing = float(params["label_smoothing"])

    batch_size = int(params["batch_size"])
    dataset = str(params["dataset"])
    epochs = 120
    feature_matching = bool(params["feature_matching"])

    keep_cols = ["x_projections", "y_projections", "real_ET"]
    nr_test = 100

    optimizer = params["optimizer"]
    learning_rate = float(params["lr"])

    if "lhcb_data2" in os.getcwd():
        path_loading = "../Data/{}/LargeSample".format(dataset)
        path_results = "../Results/{}".format(dataset)
    else:
        path_loading = "../Data/{}/Debug".format(dataset)
        path_results = "../Results/Test/{}".format(dataset)

    reshape_z = "none"
    steps_adv = int(params["adv_steps"])
    steps_gen = int(params["gen_steps"])
    steps_log = 3

    padding = {"top":2, "bottom":2, "left":0, "right":0}
    x_dim = image_shape = (52+padding["top"]+padding["bottom"], 64+padding["left"]+padding["right"], 1)
    y_dim = len(keep_cols)
    z_dim = int(params["z_dim"])


    ############################################################################################################
    # Network initialization
    ############################################################################################################


    if reshape_z == "none":
        architectures = GenerativeModel.load_from_json(architecture_path)
        architecture_gen = architectures["Generator"]
        architecture_adv = architectures["Critic"]
        if is_patchgan:
            architecture_adv.append([conv2d_logged, {"filters": 64, "kernel_size": 4, "strides": 2, "activation": tf.nn.leaky_relu}])
            if is_wasserstein:
                architecture_adv.append([conv2d_logged, {"filters": 1, "kernel_size": 4, "strides": 1, "activation": tf.identity}])
            else:
                architecture_adv.append([conv2d_logged, {"filters": 1, "kernel_size": 4, "strides": 1, "activation": tf.nn.sigmoid}])
        else:
            architecture_adv[-1][1]["activation"] = tf.nn.leaky_relu

    elif reshape_z == "replicate":
        initial_size = [7, 8]
        architecture_gen = [
                      [replicate_vector_layer, {"size": initial_size}],
                      [conv2d_transpose_logged, {"filters": 512, "kernel_size": 2, "strides": 2, "activation": activation}],
                      [conv2d_transpose_logged, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": activation}],
                      [conv2d_transpose_logged, {"filters": 1, "kernel_size": 2, "strides": 2, "activation": activation_last_layer}]
                      ]
    elif reshape_z == "sample":
        initial_size = [7, 8]
        architecture_gen = [
                      [sample_vector_layer, {"size": initial_size, "y_dim": len(keep_cols),
                                            "rfunc": sampling_distribution[0], "rparams": sampling_distribution[1]}],
                      [conv2d_transpose_logged, {"filters": 512, "kernel_size": 2, "strides": 2, "activation": activation}],
                      [conv2d_transpose_logged, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": activation}],
                      [conv2d_transpose_logged, {"filters": 1, "kernel_size": 2, "strides": 2, "activation": activation_last_layer}]
                      ]
    else:
        raise NotImplementedError("Wrong reshape_z method.")

    if is_cycle_consistent:
        architecture_aux = [
                          [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": activation}],
                          [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": activation}],
                          [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": activation}],
                          [tf.layers.flatten, {}],
                          [tf.layers.dense, {"units": z_dim+y_dim, "activation": tf.identity}],
                          ]
    else:
        architecture_aux = None

    ############################################################################################################
    # Data loading
    ############################################################################################################
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    path_saving = init.initialize_folder(algorithm=algorithm, base_folder=path_results)

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

    nr_train = train_calo.shape[0]


    ############################################################################################################
    # Preparation
    ############################################################################################################
    def prepare_algorithm(network, optimizer, learning_rate):
        network.compile(logged_labels=logging_tracker, logged_images=logging_calo, optimizer=optimizer, learning_rate=learning_rate,
                        loss=loss, feature_matching=feature_matching, label_smoothing=label_smoothing)
        network.set_attributes(keep_cols)
        post_message = """\nCalo shape: {}\nTracker shape: {}
                        \nUsed attributes: {}
                        \nAppend attributes at every layer: {}""".format(train_calo.shape, train_tracker.shape, keep_cols, append_y)
        network.log_architecture(post_message=post_message)

        nr_params = network.get_number_params()
        nr_gen_params = network._nets[0].get_number_params()
        nr_disc_params = network._nets[1].get_number_params()
        sampler = network.get_sampling_distribution()
        config_data.update({"nr_params": nr_params, "sampler": sampler, "generator_out": network._generator._output_layer.name, "optimizer": optimizer.__name__, "nr_gen_params": nr_gen_params, "nr_disc_params": nr_disc_params})

        config_data.pop("architectures")
        with open(path_saving+"/config.json", "w") as f:
            json.dump(config_data, f, indent=4)

    config_data = init.create_config_file(globals())

    ############################################################################################################
    # Model Training
    ############################################################################################################

    try:
        network = create_algorithm(algorithm, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim,
                                     gen_architecture=architecture_gen, adv_architecture=architecture_adv,
                                     aux_architecture=architecture_aux,
                                     folder=path_saving, append_y_at_every_layer=append_y,
                                     is_patchgan=is_patchgan, is_wasserstein=is_wasserstein)

        prepare_algorithm(network, optimizer, learning_rate)

        network.show_architecture()
        network.train(x_train=train_calo, y_train=train_tracker, x_test=test_calo, y_test=test_tracker,
                    epochs=epochs, batch_size=batch_size, steps=steps_gen, log_step=steps_log, gpu_options=gpu_options,
                    batch_log_step=None)
        with open(path_saving+"/EXIT_FLAG0.txt", "w") as f:
            f.write("EXIT STATUS: 0. No errors or warnings.")
        tf.reset_default_graph()
    except GeneratorExit as e:
        with open(path_saving+"/EXIT_FLAG1.txt", "w") as f:
            f.write("EXIT STATUS: 1. {}.".format(e))
        tf.reset_default_graph()
