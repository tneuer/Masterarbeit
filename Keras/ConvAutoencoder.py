#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-08-20 14:18:56
    # Description :
####################################################################################
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

run_denoising_encoder = True
encoding_dim = 16
epochs = 5

if not run_denoising_encoder:
    x_train_in = x_train
    x_train_out = x_train
    x_test_in = x_test
    x_test_out = x_test
else:
    x_train_in = np.clip(x_train + 0.5*np.random.normal(loc=0.0, scale=1.0, size=x_train.shape), 0, 1.)
    x_train_out = x_train
    x_test_in = np.clip(x_test + 0.5*np.random.normal(loc=0.0, scale=1.0, size=x_test.shape), 0, 1.)
    x_test_out = x_test


input_img = tf.keras.layers.Input(shape=(28, 28, 1))

encoder = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(input_img)
encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same")(encoder)

decoder = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(encoder)
decoder = tf.keras.layers.UpSampling2D(size=(2, 2))(decoder)
decoder = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same")(decoder)

autoencoder_model = tf.keras.Model(input_img, decoder)
encoder_model = tf.keras.Model(input_img, encoder)

autoencoder_model.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder_model.fit(x_train_in, x_train_out, epochs=epochs, batch_size=256, validation_data=(x_test_in, x_test_out), verbose=2)

decoded_imgs = autoencoder_model.predict(x_test_in)

n_digits_on_display = 10
plt.figure(figsize=(20, 4))
for i in range(n_digits_on_display):
    ax = plt.subplot(3, n_digits_on_display, i + 1)
    plt.imshow(x_test_out[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    ax = plt.subplot(3, n_digits_on_display, i + n_digits_on_display + 1)
    plt.imshow(x_test_in[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    ax = plt.subplot(3, n_digits_on_display, i + 2*n_digits_on_display + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

plt.suptitle("Conv. Autoencoder with {} latent dimensions".format(encoding_dim))
plt.show()

