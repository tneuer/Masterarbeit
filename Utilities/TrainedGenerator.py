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


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


class TrainedGenerator():
    def __init__(self, path_to_meta, path_to_config):
        self._meta_file = self._get_full_meta_path(path_to_meta)
        self._config_file = self._get_full_config_path(path_to_config)

        with open(self._config_file, "r") as f:
            self._config = json.load(f)

        self._sampler_func = eval(self._config["sampler"].split(' ', 1)[0])
        self._sampler_params = eval(self._config["sampler"].split(' ', 1)[1])
        self._z_dim = self._config["z_dim"]

        self._image_shape = self._config["image_shape"]
        self._padding = self._config["padding"]
        self._generator_out = self._config["generator_out"]

        self._sess = tf.Session()
        self._graphs_path = os.path.split(self._meta_file)[0]
        restorer = tf.train.import_meta_graph(self._meta_file)
        restorer.restore(self._sess, tf.train.latest_checkpoint(self._graphs_path))
        self._graph = tf.get_default_graph()


    @staticmethod
    def _get_full_meta_path(path_to_meta):
        if path_to_meta.endswith(".meta"):
            meta_file = path_to_meta
        elif os.path.isdir(path_to_meta):
            path_to_meta = path_to_meta if path_to_meta.endswith("/") else path_to_meta + "/"
            meta_files = [file for file in os.listdir(path_to_meta) if file.endswith(".meta")]
            meta_files.sort(key=natural_keys)
            meta_file = path_to_meta + meta_files[-1]
        else:
            raise ValueError("path_to_meta must be directory with .meta file or .meta file itself.")
        if not os.path.exists(meta_file):
            raise FileNotFoundError("{} does not exist".format(meta_file))
        return meta_file


    @staticmethod
    def _get_full_config_path(path_to_config):
        if path_to_config.endswith("config.json"):
            config_file = path_to_config
        elif os.path.isdir(path_to_config):
            path_to_config = path_to_config if path_to_config.endswith("/") else path_to_config + "/"
            config_file = path_to_config + "config.json"
        else:
            raise ValueError("path_to_config must be directory with config.json file or config.json file itself.")
        if not os.path.exists(config_file):
            raise FileNotFoundError("{} does not exist".format(config_file))
        return config_file


    @classmethod
    def create_empty_config(self, path_to_config):
        if not os.path.isdir(path_to_config):
            raise FileNotFoundError("{} does not exist.".format(path_to_config))
        path_to_config = path_to_config+"config.json" if path_to_config.endswith("/") else path_to_config + "/config.json"
        if os.path.exists(path_to_config):
            raise FileExistsError("{} already exists.".format(path_to_config))

        empty_config = self._get_empty_config()
        with open(path_to_config, "w") as f:
            f.write(empty_config)
        print(path_to_config+" created.")


    @staticmethod
    def _get_empty_config():
        include_param = ["sampler", "image_shape", "z_dim", "generator_out"]
        empty_config = '{' + "".join(['"{}": ,\n'.format(param) for param in include_param]) + '}'
        return empty_config


    def sample_noise(self, n):
        return self._sampler_func(**self._sampler_params, size=[n, self._z_dim])


    def concatenate_noise(self, inputs):
        z = self.sample_noise(len(inputs))
        valid_inpt = np.concatenate([z, inputs], axis=1)
        return valid_inpt




    def generate(self, inputs):
        n = len(inputs)
        is_training = self._graph.get_tensor_by_name("Inputs/is_training:0")
        mod_Z_input = self._graph.get_tensor_by_name("Inputs_1/mod_z:0")
        generator = self._graph.get_tensor_by_name(self._generator_out)
        output = self._sess.run(generator, feed_dict={mod_Z_input: inputs, is_training: False}).reshape([n, self._image_shape[0], self._image_shape[1]])
        return output


    def generate_from_condition(self, inputs):
        inputs = self.concatenate_noise(inputs=inputs)
        outputs = self.generate(inputs=inputs)
        return outputs

    def generate_mean_from_condition(self, inputs, n):
        outputs = [0]*len(inputs)
        for i, inpt in enumerate(inputs):
            repetitions = [inpt for _ in range(n)]
            outputs[i] = np.mean(self.generate_from_condition(repetitions), axis=0)
        outputs = np.stack(outputs, axis=0)
        return outputs

    def generate_overlay(self, inputs):
        output = self.generate_from_condition(inputs=inputs)
        added_image = np.sum(output, axis=0)
        return added_image


    def generate_multiple(self, list_of_inputs):
        outputs = [self.generate(inputs=inputs) for inputs in list_of_inputs]
        return outputs


    def generate_multiple_from_condition(self, list_of_inputs):
        list_of_inputs = [self.concatenate_noise(inputs=inpt) for inpt in list_of_inputs]
        outputs = self.generate_multiple(list_of_inputs=list_of_inputs)
        return outputs

    def generate_multiple_overlay_from_condition(self, list_of_inputs):
        outputs = self.generate_multiple_from_condition(list_of_inputs=list_of_inputs)
        outputs = np.stack([np.sum(output, axis=0) for output in outputs], axis=0)
        return outputs

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


    def input_to_image_grid(self, inputs, nrows=None):
        nrows, ncols = get_layout(n=len(inputs))
        outputs = self.generate_from_condition(inputs=inputs)
        image_array = [[image.reshape([self._image_shape[0], self._image_shape[1]]) for image in outputs[r*ncols:(r+1)*ncols]] for r in range(nrows)]
        return image_array


    def build_from_condition(self, inputs, nrows=None, column_titles=None, row_titles=None, colorbar=False):
        image_array = self.input_to_image_grid(inputs=inputs, nrows=nrows)
        fig, ax = build_images(image_array, column_titles=column_titles, row_titles=row_titles, colorbar=colorbar)
        return fig, ax


    def build_with_reference(self, inputs, reference, nrows=None, column_titles=None, row_titles=None):
        assert len(inputs) == len(reference), "Different number of logged_image and logged_label"
        inputs = np.repeat(inputs, repeats=10, axis=0)
        outputs = self.input_to_image_grid(inputs=inputs, nrows=nrows)
        reference = [ref.reshape(self._image_shape[:-1]) for ref in reference]

        outputs = [[reference[i], *outputs[i]] for i in range(len(reference))]
        column_titles = [" " for _ in range(len(reference))]
        column_titles.insert(0, "Ref.")
        fig, ax = build_images(outputs, column_titles=column_titles, row_titles=row_titles, colorbar=True)
        return fig, ax


    def build_simulated_events(self, condition, tracker_image, calo_image, eval_functions,
                               gen_scaler=1, n=10, title=None, reference_images=None):

        inputs = [condition for _ in range(n)]
        outputs = self.generate_multiple_overlay_from_condition(list_of_inputs=inputs)*gen_scaler
        mean_output = np.mean(outputs, axis=0).reshape(calo_image.shape)

        layout = get_layout(n=4+len(eval_functions)) # Tracker, Calo, Mean generated, Reference
        fig, ax = plt.subplots(nrows=layout[0], ncols=layout[1], figsize=(layout[1]*5, layout[0]*5))
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        ax = ax.ravel()

        E_max = np.max(np.stack((calo_image, mean_output), axis=0))
        titles = ["Tracker", "Calorimeter", "Generated average (n={})".format(n)]
        for i, image in enumerate([tracker_image, calo_image, mean_output]):
            if titles[i] == "Tracker":
                ax[i].imshow(image)
            else:
                im = ax[i].imshow(image, vmin=0, vmax=E_max)
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
            ax[i].set_title(titles[i])
        fig.colorbar(im, ax=ax[:3], shrink=0.6)

        for i, (func, params) in enumerate(eval_functions.items()):
            if ("cells" in func.__name__) or ("max" in func.__name__):
                fake_values = np.array(func(crop_images(outputs, **self._padding), **params))
                real_value = np.array(func(crop_images(np.array([calo_image]), **self._padding), **params))
                if reference_images is not None:
                    ref_values = np.array(func(crop_images(reference_images, **self._padding), **params))
            else:
                fake_values = np.array(func(outputs, **params))
                real_value = np.array(func(np.array([calo_image]), **params))
                if reference_images is not None:
                    ref_values = np.array(func(reference_images, **params))

            if "mass_x" in func.__name__:
                fake_values /= self._image_shape[1]
                real_value /= self._image_shape[1]
                if reference_images is not None:
                    ref_values /= self._image_shape[1]
                ax[i+3].set_xlim(-1, 1)
            elif "mass_y" in func.__name__:
                fake_values /= self._image_shape[0]
                real_value /= self._image_shape[0]
                if reference_images is not None:
                    ref_values /= self._image_shape[0]
                ax[i+3].set_xlim(-1, 1)
            elif "resolution" in func.__name__:
                ax[i+3].set_xlabel("true-reco")

            if reference_images is not None:
                ax[i+3].hist([ref_values, fake_values], bins=20, histtype="step", stacked=False,
                             label=["reference", "generated"], density=True)
                ax[i+3].legend()
            else:
                ax[i+3].hist(fake_values, bins=20)
            ax[i+3].axvline(real_value, color='k', linestyle='dashed')
            if "get_energies" == func.__name__:
                ax[i+3].axvline(real_value+0.1*real_value, color='gray', linestyle='dashed')
                ax[i+3].axvline(real_value-0.1*real_value, color='gray', linestyle='dashed')

            ax[i+3].set_title(func.__name__)

        if reference_images is not None:
            ax[-1].imshow(get_mean_image(reference_images))
            ax[-1].set_title("{} references".format(len(reference_images)))
        if title is not None:
            plt.text(0.05, 0.95, title, transform=fig.transFigure, size=24)
        return fig, ax

