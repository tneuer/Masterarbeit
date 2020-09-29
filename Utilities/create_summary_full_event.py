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

from TrainedCycleGenerator import TrainedCycleGenerator

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

if __name__ == "__main__":

    nr_test_hist = 300
    batch_size = 50
    result_path = "../../Results"
    include_all = ["ServerTemp/B2Dmunu/"]
    save_path = "../../Results/ServerTemp/B2Dmunu/Summary.pdf"
    models = [
            "{}/{}".format(folder, name) for folder in include_all for name in os.listdir(result_path+"/"+folder)
                    if os.path.isdir("{}/{}".format(result_path+"/"+folder, name))
    ]
    models.sort(key=natural_keys)

    fig, axes = plt.subplots(nrows=len(models), ncols=10, figsize=(7*len(models), 5*len(models)), facecolor='w', edgecolor='k')
    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    figs = [fig]
    colnames = ["Enery", "MaxEnergy", "Cells", "StdEnergy", "X CoM", "Y CoM", "Resolution Tracker vs Other",
                    "Resolution Comp.", "dN/dE", "(dN/dE) / (dN/dE)"]

    with open("../../Data/B2Dmunu/TestingPurpose/calo_images.pickle", "rb") as f:
        mc_data = pickle.load(f)[-nr_test_hist:]
    with open("../../Data/B2Dmunu/TestingPurpose/Trained/PiplusLowerP_CWGANGP8_out_1.pickle", "rb") as f:
        gan_data = pickle.load(f)[-nr_test_hist:]
    with open("../../Data/B2Dmunu/TestingPurpose/tracker_images.pickle", "rb") as f:
        tracker_images = pickle.load(f)[-nr_test_hist:]
    with open("../../Data/B2Dmunu/TestingPurpose/tracker_events.pickle", "rb") as f:
        tracker_events = pickle.load(f)
        tracker_real_ET = tracker_events["real_ET"].apply(sum).to_numpy()[-nr_test_hist:]
    with open("../../Data/Piplus/LargeSample/ProcessedScaler.pickle", "rb") as f:
        scaler = pickle.load(f)
        calo_scaler = scaler["Calo"]
    print("Test data loaded.")

    for model_idx, model in enumerate(models):
        print(""""
              #######################################################################
              ######## Evaluating Model {} / {}:  {}
              #######################################################################
              """.format(model_idx+1, len(models), model))


        image_shape = [64, 64, 1]
        mc_data_images = padding_zeros(mc_data, top=6, bottom=6).reshape(-1, 64, 64, 1)
        gan_data_m = padding_zeros(gan_data, top=4, bottom=4)
        tracker_images_m = padding_zeros(tracker_images, top=6, bottom=6)
        tracker_images_m = np.reshape(tracker_images_m, newshape=[-1, image_shape[0], image_shape[1]])

        #####################################################################################################
        # Model loading
        #####################################################################################################
        model_path = "{}/{}/".format(result_path, model)
        meta_path = model_path + "TFGraphs/"
        config_path = model_path + "config.json"
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
            Generator = TrainedCycleGenerator(path_to_meta=meta_path, path_to_config=config_path)
            nr_batches = int(nr_test_hist/batch_size)
            generated_images = []
            for i in range(nr_batches):
                print("Generate", i+1, "/", nr_batches)
                start = i*batch_size
                end = (i+1)*batch_size
                batch_generated_images = Generator.generate(x_inputs=gan_data_m[start:end], y_inputs=mc_data_images[start:end])
                generated_images.extend(batch_generated_images)

        generated_images = np.array(generated_images)
        generated_images = clip_outer(images=crop_images(generated_images, top=6, bottom=6), clipval=1/4)

        #####################################################################################################
        # Build figures
        #####################################################################################################
        use_functions = {get_energies: {"energy_scaler": calo_scaler/1000}, get_max_energy: {"energy_scaler": calo_scaler/1000, "maxclip": 6.12},
                        get_number_of_activated_cells: {"threshold": 5/calo_scaler},
                        get_std_energy: {"energy_scaler": calo_scaler/1000, "threshold": 0.005/calo_scaler},
                        get_center_of_mass_x: {"image_shape": image_shape}, get_center_of_mass_y: {"image_shape": image_shape},
                        get_energy_resolution: {"real_ET": tracker_real_ET, "energy_scaler": calo_scaler}}



        htos_calo_images_mc = padding_zeros(mc_data, top=6, bottom=6)
        htos_calo_images_im2im = padding_zeros(generated_images, top=6, bottom=6)
        htos_calo_images_gan = padding_zeros(gan_data.reshape([-1, 56, 64]), top=4, bottom=4)

        assert htos_calo_images_mc.shape == htos_calo_images_im2im.shape == htos_calo_images_gan.shape, "Shape mismatch."

        build_histogram_HTOS(true=htos_calo_images_mc, fake=htos_calo_images_im2im,
                             energy_scaler=calo_scaler, threshold=3600, real_ET=tracker_real_ET,
                             ax1=axes[model_idx, -2], ax2=axes[model_idx, -1])

        # Resolution Geant vs Generated
        resolution_geant_nn = (get_energies(htos_calo_images_mc) - get_energies(htos_calo_images_im2im))
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


        for func_idx, (func, params) in enumerate(use_functions.items()):
            build_histogram(true=htos_calo_images_mc, fake=htos_calo_images_im2im, fake2=htos_calo_images_gan,
                            function=func, name=colnames[func_idx], epoch="", folder=None, ax=axes[model_idx, func_idx],
                            labels=["MCTruth", "Im2Im", "GAN"], **params)
            if func_idx == 0:
                fs = {"7": 10, "12": 12, "17": 13}
                axes[model_idx, func_idx].set_ylabel(model, fontsize=fs[str(len(models))])
            if model_idx != 0:
                axes[model_idx, func_idx].set_title("")
        idx = 9
        use_functions[get_energy_resolution] = {"real_ET": tracker_real_ET[idx], "energy_scaler": calo_scaler}
        use_functions[get_center_of_mass_r] = {"image_shape": image_shape}
        # figs.append(Generator.build_simulated_events(condition=htos_calo_images_gan[idx].reshape([-1, 64, 64, 1]),
        #                          tracker_image=[tracker_images[idx]],
        #                          calo_image=np.array([htos_calo_images_mc[idx]]),
        #                          n = 500,
        #                          eval_functions=use_functions,
        #                          title=model,
        #                          reference_images=None)[0])

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