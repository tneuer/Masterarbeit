#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-26 23:06:11
    # Description :
####################################################################################
"""
import os
import re
import sys
sys.path.insert(1, '../Preprocessing')
import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import initialization as init
import matplotlib.pyplot as plt

from functionsOnImages import padding_zeros, clip_outer, get_layout, build_image, build_images, savefigs, separate_images
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, get_energy_resolution
from functionsOnImages import get_center_of_mass_r
from functionsOnImages import build_histogram_HTOS, crop_images

from TrainedGenerator import TrainedGenerator

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

if __name__ == "__main__":

    nr_test_hist = 2000
    batch_size = 500
    result_path = "../../Results"
    look_at = "PiplusMixedP"
    include_all = ["ServerTemp/PiplusMixedP2/1Good"]
    save_path = "../../Results/{}/Summary.pdf".format(include_all[0])
    models = [
            "{}/{}".format(folder, name) for folder in include_all for name in os.listdir(result_path+"/"+folder)
                    if os.path.isdir("{}/{}".format(result_path+"/"+folder, name))
    ]
    models.sort(key=natural_keys)

    fig, axes = plt.subplots(nrows=len(models), ncols=11, figsize=(6*len(models), 3*len(models)), facecolor='w', edgecolor='k')
    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    figs = [fig]
    colnames = ["Enery", "MaxEnergy", "Cells", "StdEnergy", "X CoM", "Y CoM", "Resolution Tracker vs Other",
                    "Resolution Comp.", "dN/dE", "(dN/dE) / (dN/dE)", "Tracker vs. Calo"]

    datatype = "TestingPurpose"
    with open("../../Data/{}/{}/ProcessedTest.pickle".format(look_at, datatype), "rb") as f:
        debug_data = pickle.load(f)
    with open("../../Data/{}/{}/tracker_images.pickle".format(look_at, datatype), "rb") as f:
        debug_tracker_images = pickle.load(f)
    with open("../../Data/{}/{}/ProcessedScaler.pickle".format(look_at, datatype), "rb") as f:
        scaler = pickle.load(f)
    print("Test data loaded.")

    for model_idx, model in enumerate(models):
        print(""""
              #######################################################################
              ######## Evaluating Model {} / {}:  {}
              #######################################################################
              """.format(model_idx+1, len(models), model))


        #####################################################################################################
        # Model loading
        #####################################################################################################
        model_path = "{}/{}/".format(result_path, model)
        meta_path = model_path + "TFGraphs/"
        config_path = model_path + "config.json"

        with open(config_path, "r") as f:
            config = json.load(f)
        keep_cols = config["keep_cols"]
        padding = config["padding"]
        path_loading = "../"+config["path_loading"]
        image_shape = config["image_shape"]

        debug_calo_images = padding_zeros(debug_data["Calo"], **padding)
        debug_tracker_events = debug_data["Tracker"]
        debug_tracker_events_m = debug_tracker_events[keep_cols]
        debug_tracker_images_m = padding_zeros(debug_tracker_images, **padding)
        debug_tracker_images_m = np.reshape(debug_tracker_images_m, newshape=[-1, image_shape[0], image_shape[1]])

        calo_scaler = scaler["Calo"]

        def invert_standardize_data(data, scaler, exclude=None):
            standardized_data = data.drop(exclude, axis=1, inplace=False)
            colnames = standardized_data.columns.values
            standardized_data = pd.DataFrame(data=scaler.inverse_transform(standardized_data), columns=colnames, index=data.index)
            data = pd.concat([standardized_data, data[exclude]], axis=1, sort=False)
            return data

        orig_data = invert_standardize_data(data=debug_data["Tracker"], scaler=scaler["Tracker"], exclude=["theta", "phi", "region"])
        tracker_real_ET = orig_data["real_ET"][-nr_test_hist:]


        #####################################################################################################
        # Generate images from algorithm
        #####################################################################################################

        if "Keras" in model:
            print(model_path)
            generator_file = [f for f in os.listdir(model_path+"/ModelSave") if f.startswith("Generator")]
            assert len(generator_file) == 1, "Ambiguous generator file."
            with open(model_path+"/ModelSave/"+generator_file[0], "rb") as f:
                Generator = pickle.load(f)

            nr_batches = int(nr_test_hist/batch_size)
            generated_images = []
            for i in range(nr_batches):
                print("Generate", i, "/", nr_batches)
                start = i*batch_size
                end = (i+1)*batch_size
                batch_generated_images = Generator.predict(gan_data_m[start:end]).reshape([-1, 64, 64])
                generated_images.extend(batch_generated_images)

        else:
            Generator = TrainedGenerator(path_to_meta=meta_path, path_to_config=config_path)
            generated_images = Generator.generate_batches(list_of_inputs=gan_data_m, batch_size=100)

        generated_images = np.array(generated_images)
        generated_images = clip_outer(images=crop_images(generated_images, top=2, bottom=2), clipval=1/4)
        print(generated_images.shape)

        #####################################################################################################
        # Build figures
        #####################################################################################################
        use_functions = {get_energies: {"energy_scaler": calo_scaler/1000}, get_max_energy: {"energy_scaler": calo_scaler/1000, "maxclip": 6.12},
                        get_number_of_activated_cells: {"threshold": 5/calo_scaler},
                        get_std_energy: {"energy_scaler": calo_scaler/1000, "threshold": 0.005/calo_scaler},
                        get_center_of_mass_x: {"image_shape": image_shape}, get_center_of_mass_y: {"image_shape": image_shape},
                        get_energy_resolution: {"real_ET": tracker_real_ET, "energy_scaler": calo_scaler}}


        htos_calo_images_real = debug_calo_images[-nr_test_hist:]
        htos_calo_images_fake = generated_images
        build_histogram_HTOS(true=htos_calo_images_real, fake=htos_calo_images_fake,
                             energy_scaler=calo_scaler, threshold=3600, real_ET=tracker_real_ET,
                             ax1=axes[model_idx, -2], ax2=axes[model_idx, -1])

        # Resolution Geant vs Generated
        resolution_geant_nn = (get_energies(htos_calo_images_real) - get_energies(htos_calo_images_fake))
        is_small_discrepancy = np.abs(resolution_geant_nn)<5
        resolution_geant_nn_moderate = resolution_geant_nn[is_small_discrepancy]
        axes[model_idx, -3].hist(resolution_geant_nn_moderate, bins=40, histtype="step")
        axes[model_idx, -3].set_title("Considered: {}".format(sum(is_small_discrepancy)))
        axes[model_idx, -3].set_xlabel("Geant-NN")
        resolution_mean = np.mean(resolution_geant_nn_moderate)
        resolution_std = np.std(resolution_geant_nn_moderate)
        textstr = '\n'.join((
                r'$\mu=%.2f$' % (resolution_mean, ),
                r'$\sigma=%.2f$' % (resolution_std, )
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[model_idx, -3].text(0.05, 0.95, textstr, transform=axes[model_idx, -3].transAxes, fontsize=14,
                                 verticalalignment='top', bbox=props)

        axes[model_idx, -4].scatter(tracker_real_ET, get_energies(htos_calo_images_real), label="true", alpha=0.1)
        axes[model_idx, -4].scatter(tracker_real_ET, get_energies(htos_calo_images_fake), label="fake", alpha=0.1)
        axes[model_idx, -4].legend()

        for func_idx, (func, params) in enumerate(use_functions.items()):
            build_histogram(true=htos_calo_images_real, fake=generated_images, function=func,
                                name=colnames[func_idx], epoch="", folder=None, ax=axes[model_idx, func_idx], **params)
            if func_idx == 0:
                fs = {"6": 9, "7": 10, "9": 10, "12": 12, "17": 13}
                axes[model_idx, func_idx].set_ylabel(model, fontsize=fs[str(len(models))])
            if model_idx != 0:
                axes[model_idx, func_idx].set_title("")
        idx = 42
        use_functions[get_energy_resolution] = {"real_ET": tracker_real_ET.iloc[idx], "energy_scaler": calo_scaler}
        ref = debug_tracker_events.iloc[idx]
        is_reference = (
                (debug_tracker_events["theta"] > ref["theta"]-0.01) &
                (debug_tracker_events["theta"] < ref["theta"]+0.01) &
                (np.abs(debug_tracker_events["momentum_pt"]) > np.abs(ref["momentum_pt"]-0.1*ref["momentum_pt"])) &
                (np.abs(debug_tracker_events["momentum_pt"]) < np.abs(ref["momentum_pt"]+0.1*ref["momentum_pt"]))
        )


        use_functions[get_center_of_mass_r] = {"image_shape": image_shape}
        figs.append(Generator.build_simulated_events(condition=[debug_tracker_events_m.iloc[idx]],
                                 tracker_image=debug_tracker_images[idx],
                                 calo_image=debug_calo_images[idx],
                                 n = 500,
                                 eval_functions=use_functions,
                                 title=model,
                                 reference_images=debug_calo_images[is_reference])[0])

        tf.reset_default_graph()

    savefigs(figures=figs, save_path=save_path)


    # def sample_inputs(inputs, n):
    #     inputs = inputs[np.random.choice(inputs.shape[0], n, replace=False)]
    #     return inputs
    # Generator.build_from_condition(inputs=sample_inputs(n=25, inputs=debug_tracker_events))
    # build_image(Generator.generate_overlay(inputs=sample_inputs(n=4, inputs=debug_tracker_events)))
    # build_images([
    #     Generator.generate_multiple_overlay_from_condition(list_of_inputs=[
    #                                                 sample_inputs(n=2, inputs=debug_tracker_events),
    #                                                 sample_inputs(n=5, inputs=debug_tracker_events)
    #                                             ])
    # ])
    # Generator.build_with_reference(inputs=debug_tracker_events[:10], reference=debug_calo_images[:10])
    # nr_tracks = 3
    # Generator.build_simulated_events(condition=debug_tracker_events[:nr_tracks],
    #                     tracker_image=debug_tracker_images[:nr_tracks],
    #                     calo_image=debug_calo_images[:nr_tracks],
    #                     n = 500,
    #                     eval_functions=use_functions)
    # plt.show()