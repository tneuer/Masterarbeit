#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-25 15:41:09
    # Description :
####################################################################################
"""
import re
import copy

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def logged_dense(inputs, **params):
    scope = tf.get_default_graph().get_name_scope()
    regexp = re.compile(r"(_[0-9]+)$")
    copy_params = copy.deepcopy(params)
    copy_params["activation"] = None; copy_params["name"] = params["name"] + "_dense"
    layer = tf.layers.dense(inputs, **copy_params)
    if not regexp.search(scope):
        layer_summary(scope+"/"+copy_params["name"], key=params["name"])
        layer = tf.identity(layer, name=params["name"]+"_z")
        tf.summary.histogram(params["name"]+"_unactivated", layer)
        if params["activation"] is not None:
            layer = params["activation"](layer, name=params["name"])
        tf.summary.histogram(params["name"]+"_activated", layer)
    else:
        layer = tf.identity(layer, name=params["name"]+"_z")
        if params["activation"] is not None:
            layer = params["activation"](layer, name=params["name"])
    return layer


def layer_summary(layername, key):
    variable_summary(layername+"/kernel:0", "{}/Weights".format(key))
    variable_summary(layername+"/bias:0", "{}/Bias".format(key))


def variable_summary(tensorname, name):
  var = get_tensor(tensorname)
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def get_tensor(name):
    return tf.get_default_graph().get_tensor_by_name(name)


def conv2d_logged(inputs, **params):
    scope = tf.get_default_graph().get_name_scope()
    regexp = re.compile(r"(_[0-9]+)$")
    copy_params = copy.deepcopy(params)
    copy_params["activation"] = None; copy_params["name"] = params["name"] + "_conv2d"
    layer = tf.layers.conv2d(inputs, **copy_params)
    if not regexp.search(scope):
        layer_summary(scope+"/"+copy_params["name"], key=params["name"])
        layer = tf.identity(layer, name=params["name"]+"_z")
        tf.summary.histogram(params["name"]+"_unactivated", layer)
        if params["activation"] is not None:
            layer = params["activation"](layer, name=params["name"])
        tf.summary.histogram(params["name"]+"_activated", layer)
    else:
        layer = tf.identity(layer, name=params["name"]+"_z")
        if params["activation"] is not None:
            layer = params["activation"](layer, name=params["name"])
    return layer


def conv2d_transpose_logged(inputs, **params):
    scope = tf.get_default_graph().get_name_scope()
    regexp = re.compile(r"(_[0-9]+)$")
    copy_params = copy.deepcopy(params)
    copy_params["activation"] = None; copy_params["name"] = params["name"] + "_conv2d"
    layer = tf.layers.conv2d_transpose(inputs, **copy_params)
    if not regexp.search(scope):
        layer_summary(scope+"/"+copy_params["name"], key=params["name"])
        layer = tf.identity(layer, name=params["name"]+"_z")
        tf.summary.histogram(params["name"]+"_unactivated", layer)
        if params["activation"] is not None:
            layer = params["activation"](layer, name=params["name"])
        tf.summary.histogram(params["name"]+"_activated", layer)
    else:
        layer = tf.identity(layer, name=params["name"]+"_z")
        if params["activation"] is not None:
            layer = params["activation"](layer, name=params["name"])
    return layer


def reshape_layer(inputs, shape, name=None):
    assert inputs.shape[1] == np.prod(shape), ("Reshaping not possible in " +
                                               "reshape_layer (name={}). Input: {}. Shape: {}.".format(name, inputs.shape, shape))
    return tf.reshape(tensor=inputs, shape=[-1, *shape], name=name)


def image_condition_concat(inputs, condition, name=None):
    """ Stack same noise to every pixel of the input image.
    """
    x_shapes = inputs.get_shape()
    condition_image = replicate_vector_layer(inputs=condition, size=(x_shapes[1], x_shapes[2]))
    outputs = tf.concat(values=[ inputs, condition_image ], axis=3, name=name)
    return outputs


def replicate_vector_layer(inputs, size, name=None):
    if isinstance(size, int) or len(size) != 2:
        raise TypeError("Size must be list with two entries.")
    inputs = tf.cast(tf.reshape(inputs, [-1, 1, 1, inputs.get_shape()[1]]), tf.float32)
    outputs = tf.ones([tf.shape(inputs)[0], size[0], size[1], 1]) * inputs
    return outputs


def resize_layer(inputs, size, name=None):
    return tf.image.resize_nearest_neighbor(inputs, size=(int(size), int(size)), name=name)


def sample_vector_layer(inputs, y_dim, size, rfunc, rparams, name=None):
    """ Stack different noise vector to every pixel, but sampled from the same latent distribution.
    """
    if isinstance(size, int) or len(size) != 2:
        raise TypeError("Size must be list with two entries.")
    condition = inputs[:, -y_dim:]
    condition_image = replicate_vector_layer(inputs=condition, size=(size[0], size[1]))
    if rfunc == "normal":
        random_image = tf.random.normal(shape=[tf.shape(inputs)[0], size[0], size[1], inputs.get_shape()[1]-y_dim], mean=rparams["loc"], stddev=rparams["scale"])
    elif rfunc == "uniform":
        random_image = tf.random.uniform(shape=[tf.shape(inputs)[0], size[0], size[1], inputs.get_shape()[1]-y_dim], minval=rparams["low"], maxval=rparams["high"])
    else:
        raise NotImplementedError("Sampling distribution only uniform and normal allowed for sample_vector_layer")
    outputs = tf.concat(values=[random_image, condition_image], axis=3, name=name)
    return outputs


def append_output_to_nn_layers(nn, layer):
    if nn is not None:
        nn._layers.append(layer)


def concatenate_with(inputs, layer, nn=None, name=None):
    if not isinstance(layer, int) and not isinstance(layer, str):
        raise TypeError("layer has type {}. Needed int or string.".format(type(layer)))
    if isinstance(layer, int):
        layer_concat = nn._layers[layer+1]
    else:
        layer_concat = tf.get_default_graph().get_tensor_by_name(layer)

    shape_in = inputs.get_shape()
    shape_concat = layer_concat.get_shape()
    assert ( (shape_in[1] == shape_concat[1]) and (shape_in[2] == shape_concat[2]) ), (
        "Input and Output not compatible in 'concatenate_with' layer. Inputs: {}. Concat: {}.".format(shape_in, shape_concat)
    )
    outputs = tf.concat(values=[inputs, layer_concat], axis=3)
    append_output_to_nn_layers(nn, outputs)
    return outputs


def residual_block(inputs, filters, kernel_size, activation, skip_layers, nn=None, name=None):
    if isinstance(filters, int):
        filters = [filters]*skip_layers
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]*skip_layers
    if callable(activation):
        activation = [activation]*skip_layers

    if len(filters) != skip_layers:
        raise AssertionError("Wrong number of filters. Expected 'int' or list of length {}. Received: {}".format(skip_layers, len(filters)))
    if len(kernel_size) != skip_layers:
        raise AssertionError("Wrong number of kernel_size. Expected 'int' or list of length {}. Received: {}".format(skip_layers, len(kernel_size)))
    if len(activation) != skip_layers:
        raise AssertionError("Wrong number of activation. Expected list of length {}. Received: {}".format(skip_layers, len(activation)))

    outputs = tf.layers.conv2d(inputs=inputs, filters=filters[0], kernel_size=kernel_size[0],
                               activation=activation[0], padding="SAME")
    append_output_to_nn_layers(nn, outputs)
    for skip_layer in range(skip_layers-1):
        outputs = tf.layers.conv2d(inputs=outputs, filters=filters[skip_layer+1], kernel_size=kernel_size[skip_layer+1],
                                   activation=activation[skip_layer+1], padding="SAME")
        append_output_to_nn_layers(nn, outputs)
    outputs = tf.concat(values=[outputs, inputs], axis=3, name=name)
    append_output_to_nn_layers(nn, outputs)
    return(outputs)


def unet(inputs, depth, activation, filters, nn=None, name=None):
    assert callable(activation), "activation must be callable."
    downward_layers = [inputs]

    ### Downward sampling
    for i in range(depth):
        power = 2**(i+1)
        outputs = tf.layers.conv2d(inputs=downward_layers[-1], filters=filters*power, kernel_size=2, strides=2, activation=activation)
        downward_layers.append(outputs)
        append_output_to_nn_layers(nn, outputs)
    downward_layers = downward_layers[::-1]

    ### Middle part
    outputs = tf.layers.conv2d(inputs=outputs, filters=filters*power, kernel_size=2, strides=1,
                               activation=activation, padding="SAME")
    append_output_to_nn_layers(nn, outputs)
    outputs = tf.layers.conv2d(inputs=outputs, filters=filters*power, kernel_size=2, strides=1,
                               activation=activation, padding="SAME")
    append_output_to_nn_layers(nn, outputs)
    outputs = tf.concat(values=[outputs, downward_layers[0]], axis=3)
    append_output_to_nn_layers(nn, outputs)

    ### Upward sampling
    for i in range(depth):
        outputs = tf.layers.conv2d_transpose(inputs=outputs, filters=int(filters*power/2**(i+1)), kernel_size=2, strides=2, activation=activation)
        append_output_to_nn_layers(nn, outputs)
        if i == (depth-1):
            outputs = tf.concat(values=[outputs, downward_layers[i+1]], axis=3, name=name)
        else:
            outputs = tf.concat(values=[outputs, downward_layers[i+1]], axis=3)
        append_output_to_nn_layers(nn, outputs)

    return(outputs)


def unet_original(inputs, depth, activation, filters, training, nn=None, name=None, logged=False):
    assert callable(activation), "activation must be callable."
    if logged:
        layer = conv2d_logged
    else:
        layer = tf.layers.conv2d
    downward_layers = []
    nr_filters = []
    # start_filters = int(inputs.shape[3])
    outputs = inputs
    layer_nr = 0
    base_name = "u_" + layer.__name__

    ### Downward sampling
    for i in range(depth):
        power = 2**i
        outputs = layer(inputs=outputs, filters=filters*power, kernel_size=3,
                                   strides=1, activation=activation, padding="SAME", name=base_name+str(layer_nr))
        layer_nr += 1
        append_output_to_nn_layers(nn, outputs)
        outputs = layer(inputs=outputs, filters=filters*power, kernel_size=3,
                                   strides=1, activation=activation, padding="SAME", name=base_name+str(layer_nr))
        layer_nr += 1
        append_output_to_nn_layers(nn, outputs)
        if i == depth-1:
            outputs = tf.layers.dropout(inputs=outputs, rate=0.5, training=training)
            append_output_to_nn_layers(nn, outputs)
        downward_layers.append(outputs)
        outputs = tf.layers.max_pooling2d(inputs=outputs, pool_size=2, strides=2)
        append_output_to_nn_layers(nn, outputs)
        nr_filters.append(filters*power)
    downward_layers = downward_layers[::-1]
    nr_filters = nr_filters[::-1]

    ### Middle part
    outputs = layer(inputs=outputs, filters=nr_filters[0], kernel_size=3, strides=1,
                               activation=tf.nn.relu, padding="SAME", name=base_name+str(layer_nr))
    layer_nr += 1
    append_output_to_nn_layers(nn, outputs)
    outputs = tf.layers.dropout(inputs=outputs, rate=0.5, training=training)
    append_output_to_nn_layers(nn, outputs)
    outputs = layer(inputs=outputs, filters=nr_filters[0], kernel_size=3, strides=1,
                               activation=tf.nn.relu, padding="SAME", name=base_name+str(layer_nr))
    layer_nr += 1
    append_output_to_nn_layers(nn, outputs)
    outputs = tf.layers.dropout(inputs=outputs, rate=0.5, training=training)
    append_output_to_nn_layers(nn, outputs)
    ### Upward sampling
    for i in range(depth):
        outputs = tf.image.resize_nearest_neighbor(images=outputs, size=(outputs.shape[1]*2, outputs.shape[2]*2))
        append_output_to_nn_layers(nn, outputs)
        outputs = layer(inputs=outputs, filters=nr_filters[i], kernel_size=2, strides=1,
                                   activation=activation, padding="SAME", name=base_name+str(layer_nr))
        layer_nr += 1
        append_output_to_nn_layers(nn, outputs)
        if i == (depth-1):
            outputs = tf.concat(values=[outputs, downward_layers[i]], axis=3, name=name)
        else:
            outputs = tf.concat(values=[outputs, downward_layers[i]], axis=3)
        append_output_to_nn_layers(nn, outputs)
        outputs = layer(inputs=outputs, filters=nr_filters[i], kernel_size=3, strides=1,
                                   activation=activation, padding="SAME", name=base_name+str(layer_nr))
        layer_nr += 1
        append_output_to_nn_layers(nn, outputs)
        outputs = layer(inputs=outputs, filters=nr_filters[i], kernel_size=3, strides=1,
                                   activation=activation, padding="SAME", name=base_name+str(layer_nr))
        layer_nr += 1
        append_output_to_nn_layers(nn, outputs)

    return(outputs)


def inception_block(inputs, filters, nn=None, name=None):
    l1 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=1, padding="SAME", activation=tf.nn.relu)
    append_output_to_nn_layers(nn, l1)

    l2 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=1, padding="SAME", activation=tf.nn.relu)
    append_output_to_nn_layers(nn, l2)
    l2 = tf.layers.conv2d(inputs=l2, filters=filters, kernel_size=3, padding="SAME", activation=tf.nn.relu)
    append_output_to_nn_layers(nn, l2)

    l3 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=1, padding="SAME", activation=tf.nn.relu)
    append_output_to_nn_layers(nn, l3)
    l3 = tf.layers.conv2d(inputs=l3, filters=filters, kernel_size=5, padding="SAME", activation=tf.nn.relu)
    append_output_to_nn_layers(nn, l3)

    l4 = tf.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=1, padding="SAME")
    append_output_to_nn_layers(nn, l4)
    l4 = tf.layers.conv2d(inputs=l4, filters=filters, kernel_size=1, padding="SAME", activation=tf.nn.relu)
    append_output_to_nn_layers(nn, l4)

    outputs = tf.concat(values=[l1, l2, l3, l4], axis=3, name=name)
    append_output_to_nn_layers(nn, outputs)

    return(outputs)



