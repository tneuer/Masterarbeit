#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-13 12:01:33
    # Description :
####################################################################################
"""
import os
import re
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from functionsOnImages import get_layout, build_images, crop_images, get_mean_image

from TrainedGenerator import TrainedGenerator


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


class TrainedIm2Im(TrainedGenerator):

    def concatenate_noise(self, inputs):
        zs = self.sample_noise(len(inputs))
        zs = np.array([np.tile(z, (*self._image_shape[:-1], 1)) for z in zs])
        valid_inpt = np.concatenate([inputs, zs], axis=3)
        return valid_inpt

    def generate(self, inputs):
        n = len(inputs)
        is_training = self._graph.get_tensor_by_name("Inputs/is_training:0")
        try:
            mod_Z_input = self._graph.get_tensor_by_name("real_1:0")
        except KeyError:
            mod_Z_input = self._graph.get_tensor_by_name("Inputs_2/mod_z:0")
        generator = self._graph.get_tensor_by_name(self._generator_out)
        output = self._sess.run(generator, feed_dict={mod_Z_input: inputs, is_training: False}).reshape(
            [n, self._image_shape[0], self._image_shape[1]]
        )
        return output

    def generate_batches(self, inputs, batch_size):
        nr_batches = int(np.ceil(len(inputs) / batch_size))
        results = [0]*len(inputs)
        for batch in range(nr_batches):
            print(batch+1,"/",nr_batches)
            current_batch = inputs[batch*batch_size:(batch+1)*batch_size]
            generated_samples = self.generate_from_condition(inputs=current_batch)
            results[batch*batch_size:(batch+1)*batch_size] = generated_samples[:]
        results = np.stack(results) #.reshape([-1, *generated_samples.shape[1:], 1])
        return(results)


    def build_simulated_events(self, condition, tracker_image, calo_image, cgan_image, eval_functions, n=10, title=None):

        inputs = [condition for _ in range(n)]
        outputs = self.generate_multiple_overlay_from_condition(list_of_inputs=inputs)
        mean_output = np.mean(outputs, axis=0).reshape(calo_image.shape)

        layout = get_layout(n=4+len(eval_functions)) # Tracker, Calo, GAN, Mean generated
        fig, ax = plt.subplots(nrows=layout[0], ncols=layout[1], figsize=(layout[1]*5, layout[0]*5))
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        ax = ax.ravel()

        E_max = np.max(np.stack((calo_image, cgan_image, mean_output), axis=0))
        titles = ["Tracker", "Calorimeter", "CGAN", "Generated average (n={})".format(n)]
        for i, image in enumerate([tracker_image, calo_image, cgan_image, mean_output]):
            if titles[i] == "Tracker":
                ax[i].imshow(image)
            else:
                im = ax[i].imshow(image, vmin=0, vmax=E_max)
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
            ax[i].set_title(titles[i])
        fig.colorbar(im, ax=ax[:4], shrink=0.6)

        for i, (func, params) in enumerate(eval_functions.items()):
            if ("cells" in func.__name__) or ("max" in func.__name__):
                fake_values = np.array(func(crop_images(outputs, **self._padding), **params))
                real_value = np.array(func(crop_images(np.array([calo_image]), **self._padding), **params))
            else:
                fake_values = np.array(func(outputs, **params))
                real_value = np.array(func(np.array([calo_image]), **params))

            if "mass_x" in func.__name__:
                fake_values /= self._image_shape[1]
                real_value /= self._image_shape[1]
                ax[i+4].set_xlim(-1, 1)
            elif "mass_y" in func.__name__:
                fake_values /= self._image_shape[0]
                real_value /= self._image_shape[0]
                ax[i+4].set_xlim(-1, 1)
            elif "resolution" in func.__name__:
                ax[i+4].set_xlabel("tracker-reco")

            ax[i+4].hist(fake_values, bins=20)
            ax[i+4].axvline(real_value, color='k', linestyle='dashed')
            if "get_energies" == func.__name__:
                ax[i+4].axvline(real_value+0.1*real_value, color='gray', linestyle='dashed')
                ax[i+4].axvline(real_value-0.1*real_value, color='gray', linestyle='dashed')

            ax[i+4].set_title(func.__name__)

        if title is not None:
            plt.text(0.05, 0.95, title, transform=fig.transFigure, size=24)
        return fig, ax

