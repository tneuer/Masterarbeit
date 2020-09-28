#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-19 16:57:42
    # Description :
####################################################################################
"""
import os
import sys
sys.path.insert(1, "../Utilities")
sys.path.insert(1, "../TFModels/building_blocks")
import json
import copy
import pickle
import layers

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict

from functionsOnImages import scale_images, flatten_images, padding_zeros
from networks import NeuralNetwork

#############################################################################################################
############ Load data
#############################################################################################################
def load_data(data_path, mode="train"):
    valid_modes = ["all", "train", "events", "images", "tracker", "calo"]

    data = {}
    if mode not in valid_modes:
        raise ValueError("'mode' has to be in {}".format(valid_modes))
    if mode != "calo":
        data.update(load_tracker_data(data_path, mode))
    if mode != "tracker":
        data.update(load_calo_data(data_path, mode))

    return data

def load_tracker_data(data_path, mode):
    tracker_data = {}
    if mode in ["all", "train", "events", "tracker"]:
        if os.path.exists(data_path+"/tracker_events.csv"):
            tracker_data["tracker_events"] = pd.read_csv(data_path+"/tracker_events.csv")
        else:
            with open(data_path+"/tracker_events.pickle", "rb") as f:
                tracker_data["tracker_events"] = pickle.load(f)

    if mode in ["all", "images", "tracker"]:
        with open(data_path+"/tracker_images.pickle", "rb") as f:
            tracker_data["tracker_images"] = pickle.load(f)

    return tracker_data


def load_calo_data(data_path, mode):
    calo_data = {}
    if mode in ["all", "events", "calo"]:
        with open(data_path+"/calo_events.pickle", "rb") as f:
            calo_data["calo_events"] = pickle.load(f)

    if mode in ["all", "train", "images", "calo"]:
        with open(data_path+"/calo_images.pickle", "rb") as f:
            calo_data["calo_images"] = pickle.load(f)

    return calo_data


def load_logging_data(data_path):
    data = {}
    data["tracker_events"] = pd.read_csv(data_path+"/tracker_events.csv")
    with open(data_path+"/calo_images.pickle", "rb") as f:
            data["calo_images"] = pickle.load(f)
    return data


def load_processed_data(data_path, mode="all", return_scaler=True):
    possible_modes = ["all", "train", "validation", "test"]
    if mode not in possible_modes:
        raise ValueError("mode must be in {}.".format(possible_modes))

    data = {}
    if mode in ["all", "train"]:
        with open(data_path+"/ProcessedTrain.pickle", "rb") as f:
            data["train"] = pickle.load(f)
    if mode in ["all", "test"]:
        with open(data_path+"/ProcessedTest.pickle", "rb") as f:
            data["test"] = pickle.load(f)
    if mode in ["all", "validation"]:
        with open(data_path+"/ProcessedValidation.pickle", "rb") as f:
            data["validation"] = pickle.load(f)

    if return_scaler:
        with open(data_path+"/ProcessedScaler.pickle", "rb") as f:
            scaler = pickle.load(f)

        if mode != "all":
            return data[mode], scaler
        return data, scaler
    else:
        if mode != "all":
            return data[mode]
        return data


#############################################################################################################
############ Rotate data
#############################################################################################################
def rotate_system(images, tracker, return_array=False):
    if type(tracker) != pd.DataFrame:
        tracker = pd.DataFrame(tracker, columns=["x_projections", "y_projections", "momentum_p", "momentum_px", "momentum_py", "momentum_pz"])
    images = rotate_images(images)
    tracker = transform_rows(tracker)
    if return_array:
        return images, tracker.values
    return images, tracker


def rotate_images(images):
    assert (images[0].shape == np.rot90(images[0]).shape,
            "Rotated images has different shape than original one. Shape: {} -> {}".format(images[0].shape, np.rot90(images[0]).shape))
    images = np.concatenate([images,
                            np.stack([np.rot90(image, k=1) for image in images], axis=0),
                            np.stack([np.rot90(image, k=2) for image in images], axis=0),
                            np.stack([np.rot90(image, k=3) for image in images], axis=0)
                            ],
                            axis=0)
    return images


def transform_rows(df):
    nr_obs = df.shape[0]
    df = pd.concat([df.copy() for _ in range(4)], ignore_index=True)

    df.loc[nr_obs:2*nr_obs-1, ["y_projections", "momentum_py"]] *= -1
    df.loc[2*nr_obs:3*nr_obs-1, ["x_projections", "y_projections", "momentum_px", "momentum_py"]] *= -1
    df.loc[3*nr_obs:4*nr_obs, ["x_projections", "momentum_px"]] *= -1
    return df



#############################################################################################################
############ Other initilization
#############################################################################################################


def initialize_folder(algorithm, base_folder):
    for i in range(1, 2000):
        folder_path = "{}/{}{}".format(base_folder, algorithm, i)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            os.mkdir(folder_path+"/GeneratedSamples")
            os.mkdir(folder_path+"/TFGraphs")
            os.mkdir(folder_path+"/Evaluation")
            break
    else:
        raise MemoryError("Two many folders exist! Delete some!")
    return folder_path


def create_config_file(global_vars):
  config_data = get_config_dict(global_vars)
  config_data = function_to_string(config_data)
  return config_data


def get_config_dict(global_vars):
    config_data = {}
    for key, value in global_vars.items():
        if not (
            ("__" in key) or
            (not isinstance(value, (int, float, list, dict, str, list, tuple, bool, np.bool)) and key not in ["activation", "activation_last_layer", "layer"]) or
            (key in ["data", "log_data", "test_data", "log_message", "tracker_events", "_", "scaler",
             "config_data", "sampled_params", "params", "param_dict", "sampling_distribution"])
        ):
            config_data[key] = value
    config_data = copy.deepcopy(config_data)
    config_data = OrderedDict(sorted(config_data.items(), key=lambda x: x[0]))
    return config_data


def function_to_string(config_data):
    custom_layers = [l for l in dir(layers) if "__" not in l]
    try:
        config_data["activation"] = "tf.nn." + config_data["activation"].__name__
        config_data["activation_last_layer"] = "tf.nn." + config_data["activation_last_layer"].__name__
        config_data["layer"] = config_data["layer"].__name__
    except KeyError:
        pass
    for key, value in config_data.items():
        if "architecture" in key and isinstance(value, list):
            for layer in config_data[key]:
                if layer[0].__name__ in custom_layers:
                    layer[0] = layer[0].__name__
                else:
                    layer[0] = "tf.layers." + layer[0].__name__

                try:
                    layer[1]["activation"] = "tf.nn." + layer[1]["activation"].__name__
                except KeyError:
                    pass
    return config_data


def load_config(path):
    with open(path) as json_file:
        config_data = json.load(json_file)
        config_data = string_to_function(config_data)
    return config_data


def string_to_function(config_data):
    import tensorflow as tf
    from layers import logged_dense, reshape_layer, resize_layer, replicate_vector_layer, sample_vector_layer
    config_data["activation"] = eval(config_data["activation"])
    config_data["activation_last_layer"] = eval(config_data["activation_last_layer"])
    try:
        config_data["layer"] = eval(config_data["layer"])
    except NameError:
        pass
    for key, value in config_data.items():
        if "architecture" in key:
          for layer in config_data[key]:
            layer[0] = eval(layer[0])
            if layer[0] not in [reshape_layer]:
                layer[1]["activation"] = eval(layer[1]["activation"])
    return config_data


def load_architecture(path):
    import tensorflow as tf
    with open(path) as json_file:
        architectures = json.load(json_file)
        for network_type, network_architecture in architectures.items():
            for layer in network_architecture:
                if layer[0] in ["tf.layers.logged_dense", "tf.layers.reshape_layer"]:
                    layer[0] = layer[0].split(".")[-1]
                layer[0] = eval(layer[0])

                try:
                    layer[1]["activation"] = eval(layer[1]["activation"])
                except KeyError:
                    pass
    return architectures


