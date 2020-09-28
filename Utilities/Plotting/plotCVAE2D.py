#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-08-26 22:15:23
    # Description :
####################################################################################
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


meta_path = "/home/tneuer/Backup/Uni/Masterarbeit/Programs/Autoencoders/Tensorflow/CVAE_log/Graphs/Graph_80.meta"
model_path = "/home/tneuer/Backup/Uni/Masterarbeit/Programs/Autoencoders/Tensorflow/CVAE_log/Graphs/"

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_train = x_train[:10000]
y_train = y_train[:10000]

nr_labels = len(set(y_train))
latent_dim = 2
number_to_draw = np.zeros(shape=[1, nr_labels])
number_to_draw[:, 4] = 1
number_to_draw[:, 5] = 1

def onclick(event):
    latent_space_input = graph.get_tensor_by_name("Inputs/encoded_input:0")
    decoder_out = graph.get_tensor_by_name("Decoder_1/decoder_output:0")
    my_noise = np.array([[event.xdata, event.ydata]])
    scale = nr_images * 28 / 2
    my_noise_scaled = ((my_noise - scale) / scale) * 3
    my_noise_scaled[0, 1] *= -1
    print(my_noise, my_noise_scaled)
    decoded_number = sess.run(decoder, feed_dict={encoded_input: my_noise_scaled, label: number_to_draw})

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(decoded_number.reshape(28, 28), cmap="gray", origin=None)
    plt.title("(x, y) = ({}, {})".format(round(event.xdata, 2), round(event.ydata, 2)))
    plt.show()


with tf.Session() as sess:
    restorer = tf.train.import_meta_graph(meta_path)
    restorer.restore(sess, tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()

    label = graph.get_tensor_by_name("Inputs/input_label:0")
    encoded_input = graph.get_tensor_by_name("Inputs/encoded_input:0")
    decoder = graph.get_tensor_by_name("Decoder_1/decoder_output:0")

    fig = plt.figure(figsize=(7, 7))
    nr_images = 5
    x_limit = np.linspace(-3, 3, nr_images)
    y_limit = np.linspace(3, -3, nr_images)
    empty_image = np.empty((28*nr_images, 28*nr_images))
    for j, pi in enumerate(y_limit):
        for i, zi in enumerate(x_limit):
            generated_latent_layer = np.array([[zi, pi]])
            generated_image = sess.run(decoder, feed_dict={encoded_input: generated_latent_layer, label: number_to_draw})
            empty_image[j*28:(j+1)*28, i*28:(i+1)*28] = generated_image[0].reshape(28, 28)
    plt.imshow(empty_image, cmap="gray")
    plt.grid(False)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()