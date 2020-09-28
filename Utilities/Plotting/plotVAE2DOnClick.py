#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-08-25 17:01:19
    # Description :
####################################################################################
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl

meta_path = "/home/tneuer/Backup/Uni/Masterarbeit/Programs/Autoencoders/Tensorflow/VAE_log/Graphs/Graph-140.meta"
model_path = "/home/tneuer/Backup/Uni/Masterarbeit/Programs/Autoencoders/Tensorflow/VAE_log/Graphs/"

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_train = x_train[:10000]
y_train = y_train[:10000]

def onclick(event):
    latent_space_input = graph.get_tensor_by_name("Inputs/encoded_input:0")
    decoder_out = graph.get_tensor_by_name("Decoder_1/decoder_output:0")
    decoded_number = sess.run(decoder_out, {latent_space_input: np.array([[event.xdata, event.ydata]])})

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(decoded_number.reshape(28, 28), cmap="gray", origin=None)
    plt.title("(x, y) = ({}, {})".format(round(event.xdata, 2), round(event.ydata, 2)))
    plt.show()


with tf.Session() as sess:
    restorer = tf.train.import_meta_graph(meta_path)
    restorer.restore(sess, tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()

    input_image = graph.get_tensor_by_name("Inputs/input_image:0")
    encoder_out = graph.get_tensor_by_name("Encoder/mean_layer:0")
    latent_space_vectors = sess.run(encoder_out, feed_dict={input_image: x_train})

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(10)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    plt.scatter(latent_space_vectors[:, 0], latent_space_vectors[:, 1], c=y_train, cmap=cmap, norm=norm)
    ax1.set_title("Latent space")
    ax2 = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    ax2.set_ylabel('Label', size=12)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()