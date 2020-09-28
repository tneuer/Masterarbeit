#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-08-19 20:40:00
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

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

run_denoising_encoder = True
encoding_dim = 16
epochs = 10

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

input_img = tf.keras.layers.Input(shape=(784, ))
encoder = tf.keras.layers.Dense(units=128, activation="relu")(input_img)
encoder = tf.keras.layers.Dense(units=32, activation="relu")(input_img)
encoder = tf.keras.layers.Dense(units=encoding_dim, activation="relu")(encoder)
decoder = tf.keras.layers.Dense(units=32, activation="relu")(encoder)
decoder = tf.keras.layers.Dense(units=128, activation="relu")(encoder)
decoder = tf.keras.layers.Dense(units=784, activation="sigmoid")(encoder)

autoencoder_model = tf.keras.Model(input_img, decoder)
encoder_model= tf.keras.Model(input_img, encoder)
encoded_input = tf.keras.Input(shape=(encoding_dim, ))
decoder_layer = autoencoder_model.layers[-1]
decoder = tf.keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder_model.fit(x_train_in, x_train_out, epochs=epochs, batch_size=256, shuffle=True, validation_data=(x_test_in, x_test_out), verbose=2)

encoded_imgs = encoder_model.predict(x_test_in)
decoded_imgs = decoder.predict(encoded_imgs)

n_digits_on_display = 10
plt.figure(figsize=(20, 4))
for i in range(n_digits_on_display):
    # display original
    ax = plt.subplot(2, n_digits_on_display, i + 1)
    plt.imshow(x_test_in[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n_digits_on_display, i + 1 + n_digits_on_display)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.suptitle("Autoencoder with {} latent dimensions".format(encoding_dim))
plt.show()


