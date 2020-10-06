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
        mod_Z_input = self._graph.get_tensor_by_name("real_1:0")
        generator = self._graph.get_tensor_by_name(self._generator_out)
        output = self._sess.run(generator, feed_dict={mod_Z_input: inputs, is_training: False}).reshape(
            [n, self._image_shape[0], self._image_shape[1]]
        )
        return output

    def generate_batches(self, list_of_inputs, batch_size):
        assert isinstance(list_of_inputs, list), "list of inputs must be of type list."
        assert isinstance(list_of_inputs[0][0], list), ("List of inputs must have following structure: " +
            "List[ Event[ Track[ x, y, et ] ] ]. But at position list_of_inputs[0][0] is {}.".format(list_of_inputs[0][0])
        )

        nr_batches = int(np.ceil(len(list_of_inputs) / batch_size))
        results = [0]*len(list_of_inputs)
        for batch in range(nr_batches):
            print(batch+1,"/",nr_batches)
            current_batch = list_of_inputs[batch*batch_size:(batch+1)*batch_size]
            generated_samples = self.generate_multiple_overlay_from_condition(list_of_inputs=current_batch)
            results[batch*batch_size:(batch+1)*batch_size] = generated_samples[:]
        results = np.stack(results) #.reshape([-1, *generated_samples.shape[1:], 1])
        return(results)

