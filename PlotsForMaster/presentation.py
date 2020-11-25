#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-10-26 18:01:12
    # Description :
####################################################################################
"""

import os
import sys
os.chdir("../")
sys.path.insert(1, "TFModels")
sys.path.insert(1, "Preprocessing")
sys.path.insert(1, "TFModels/building_blocks")
sys.path.insert(1, "TFModels/CGAN")
sys.path.insert(1, "Utilities")

import numpy as np
import tensorflow as tf

from CGAN import CGAN
from sklearn.preprocessing import OneHotEncoder
from building_blocks.layers import logged_dense, reshape_layer
from initialization import initialize_folder

#############################################################################################################
############ Load data
#############################################################################################################
nr_examples = 10000
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, y_train = x_train[:nr_examples], y_train[:nr_examples]
x_train, x_test = x_train/255., x_test/255.

x_train = x_train.reshape([-1, 28, 28, 1])
x_test = x_test.reshape([-1, 28, 28, 1])[:100]

enc = OneHotEncoder(sparse=False)
y_train = enc.fit_transform(y_train.reshape(-1, 1))
y_test = enc.transform(y_test.reshape(-1, 1))[:100]

logging_idxs = []

for number in range(10):
    for idx, lbl in enumerate(y_test):
        if np.argmax(lbl) == number:
            logging_idxs.append(idx)
            break

logging_images = x_test[logging_idxs]
logging_labels = y_test[logging_idxs]

#############################################################################################################
############ Build & train conditional network
#############################################################################################################
gen_architecture = [
    [logged_dense, {"units": 7*7*32, "activation": tf.nn.relu}],
    [reshape_layer, {"shape": [7, 7, 32]}],
    [tf.layers.conv2d_transpose, {"filters": 64, "kernel_size": 4, "strides": 2, "activation": tf.nn.relu}],
    [tf.layers.conv2d, {"filters": 64, "kernel_size": 4, "strides": 1, "activation": tf.nn.relu}],
    [tf.layers.conv2d_transpose, {"filters": 1, "kernel_size": 4, "strides": 2, "activation": tf.nn.sigmoid}],
]
disc_architecture = [
    [tf.layers.conv2d, {"filters": 32, "kernel_size": 5, "strides": 2, "activation": tf.nn.relu}],
    [tf.layers.conv2d, {"filters": 16, "kernel_size": 5, "strides": 2, "activation": tf.nn.relu}],
    [tf.layers.flatten, {}],
    [logged_dense, {"units": 128, "activation": tf.nn.relu}],
    [logged_dense, {"units": 64, "activation": tf.nn.relu}],
    [reshape_layer, {"shape": [8, 8, 1]}],
    [tf.layers.conv2d, {"filters": 1, "kernel_size": 3, "strides": 2, "activation": tf.identity}]
]

inpt_dim = [28, 28, 1]
z_dim = 2
batch_size = 32
epochs = 100
base_folder = "./PlotsForMaster/Results/"
nr_batches = nr_examples // batch_size
batch_log_step = nr_batches // 10
loss = "wasserstein"
adv_steps = 5
feature_matching = False
patchgan = True

algorithm = loss+"_patch"+str(patchgan)+"_fm"+str(feature_matching)
folder = initialize_folder(algorithm=algorithm, base_folder=base_folder)
vanillgan = CGAN(
    x_dim=inpt_dim, y_dim=10, z_dim=z_dim, gen_architecture=gen_architecture,
    adversarial_architecture=disc_architecture, folder=folder, is_patchgan=patchgan, is_wasserstein=True
)
vanillgan.show_architecture()
vanillgan.log_architecture()
vanillgan.compile(
    loss="wasserstein", logged_images=logging_images, logged_labels=logging_labels, label_smoothing=0.9,
    feature_matching=feature_matching, optimizer=tf.train.AdamOptimizer
)
vanillgan.train(x_train=x_train, y_train=y_train, x_test=x_test[:500], y_test=y_test[:500],
                batch_size=batch_size, epochs=epochs, adversarial_steps=adv_steps, gen_steps=1,
                batch_log_step=None, log_step=1)

tf.reset_default_graph()
