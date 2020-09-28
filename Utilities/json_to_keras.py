#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-09-08 16:28:06
    # Description :
####################################################################################
"""

import os
import json

from keras.layers import Conv2D, Conv2DTranspose, Concatenate, Flatten, Dense, Reshape
from keras.layers import Activation, LeakyReLU, ReLU
from keras.layers import Dropout, BatchNormalization

def json_to_keras(inpt, path_to_json, network):
    with open(path_to_json) as f:
        architecture = json.load(f)
    otpt = inpt
    layers = []
    for i, (layer, params) in enumerate(architecture[network]):
        if layer == "conv2d_logged" or layer == "tf.layers.conv2d":
            act = params.pop("activation")
            otpt = Conv2D(**params)(otpt)
            otpt = return_activated(inpt=otpt, tf_activation=act)
        elif layer == "conv2d_transpose_logged" or layer == "tf.layers.conv2d_transpose":
            act = params.pop("activation")
            otpt = Conv2DTranspose(**params)(otpt)
            otpt = return_activated(inpt=otpt, tf_activation=act)
        elif layer == "tf.layers.batch_normalization":
            otpt = BatchNormalization()(otpt)
        elif layer == "tf.layers.dropout":
            otpt = Dropout(rate=0.5)(otpt)
        elif layer == "concatenate_with":
            otpt = Concatenate()([otpt, layers[params["layer"]]])
        elif layer == "tf.layers.flatten":
            otpt = Flatten()(otpt)
        elif layer == "tf.layers.dense" or layer == "logged_dense":
            act = params.pop("activation")
            otpt = Dense(**params)(otpt)
            otpt = return_activated(inpt=otpt, tf_activation=act)
        elif layer == "reshape_layer":
            otpt = Reshape(target_shape=params["shape"])(otpt)
        else:
            raise NotImplementedError("{} not yet implemented (layer).".format(layer))

        layers.append(otpt)

    return otpt


def return_activated(inpt, tf_activation):
    if tf_activation == "tf.nn.relu":
        return(ReLU()(inpt))
    elif tf_activation == "tf.nn.leaky_relu":
        return(LeakyReLU(alpha=0.2)(inpt))
    elif tf_activation == "tf.nn.sigmoid":
        return(Activation('sigmoid')(inpt))
    else:
        raise NotImplementedError("{} not yet implemented (activation).".format(tf_activation))



if __name__ == "__main__":
    from keras.models import Input
    path_to_json = "/home/tneuer/Backup/Uni/Masterarbeit/Architectures/Pix2Pix/keraslike.json"
    inpt = Input(shape=[64, 64, 1])

    json_to_keras(inpt, path_to_json, network="Generator")
