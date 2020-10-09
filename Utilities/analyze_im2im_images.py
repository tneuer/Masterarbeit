#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-15 20:06:57
    # Description :
####################################################################################
"""
import os
import sys
sys.path.insert(1, "../Preprocessing")
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import initialization as init
import matplotlib.pyplot as plt

from TrainedIm2Im import TrainedIm2Im
from functionsOnImages import padding_zeros, get_layout, build_image, build_images, savefigs
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, get_energy_resolution, get_center_of_mass_r
from functionsOnImages import build_histogram_HTOS, crop_images, clip_outer


#####################################################################################################
# Model loading
#####################################################################################################
source_dir = "../../Results/B2Dmunu"
model_paths = [f.path for f in os.scandir(source_dir) if os.path.isdir(f.path)]
nr_test_hist = 1000
batch_size = 100

#####################################################################################################
# Data loading
#####################################################################################################

path_loading = "../../Data/B2Dmunu/Debug"
image_shape = [64, 64, 1]

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

mc_data_images = padding_zeros(mc_data, top=6, bottom=6).reshape(-1, 64, 64, 1)
gan_data_m = padding_zeros(gan_data, top=4, bottom=4)
gan_data_m = np.clip(gan_data_m, a_min=0, a_max=calo_scaler)
gan_data_m /= calo_scaler
tracker_images_m = padding_zeros(tracker_images, top=6, bottom=6)
tracker_images_m = np.reshape(tracker_images_m, newshape=[-1, image_shape[0], image_shape[1]])
#####################################################################################################
# Model loading
#####################################################################################################

for model_idx, model_path in enumerate(model_paths):
    print("Working on {} / {}: {}...".format(model_idx+1, len(model_paths), model_path))
    if os.path.exists(model_path+"/Evaluation.pdf"):
        print("{} already exists.".format(model_path+"/Evaluation.pdf"))
        continue

    meta_path = model_path + "/TFGraphs/"
    config_path = model_path + "/config.json"
    Generator = TrainedIm2Im(path_to_meta=meta_path, path_to_config=config_path)
    generated_images = Generator.generate_batches(inputs=gan_data_m, batch_size=batch_size)

    #####################################################################################################
    # Evaluation
    #####################################################################################################
    figs = []
    fig2, axes = plt.subplots(2, 4, figsize=(20, 20))
    axes = np.ravel(axes)
    # use_functions = {get_energies: {"energy_scaler": calo_scaler/1000}, get_max_energy: {"energy_scaler": calo_scaler/1000, "maxclip": 6.12},
    #                 get_number_of_activated_cells: {"threshold": 5/calo_scaler},
    #                 get_std_energy: {"energy_scaler": calo_scaler/1000, "threshold": 0.005/calo_scaler},
    #                 get_center_of_mass_x: {"image_shape": image_shape}, get_center_of_mass_y: {"image_shape": image_shape},
    #                 get_energy_resolution: {"real_ET": tracker_real_ET, "energy_scaler": calo_scaler}}
    # colnames = ["Enery [GeV]", "MaxEnergy [GeV]", "Cells", "StdEnergy [GeV]", "X CoM", "Y CoM", "Tracker-GAN"]
    use_functions = {get_energies: {"energy_scaler": calo_scaler/1000}, get_max_energy: {"energy_scaler": calo_scaler/1000, "maxclip": 6.12},
                    get_number_of_activated_cells: {"threshold": 5/calo_scaler},
                    get_std_energy: {"energy_scaler": calo_scaler/1000, "threshold": 5/calo_scaler},
                    get_energy_resolution: {"real_ET": tracker_real_ET, "energy_scaler": calo_scaler}}
    colnames = [
        r"$\sum_{i} E_{Ti}$ [GeV]", r"$\max (E_{Ti})$ [GeV]", r"$\sum_{i} [E_{Ti} > 6MeV]$",
        r"std($E_{Ti}$) [GeV]", r"$E_{T,Tracker} - \sum_{i} E_{Ti}$ [MeV]"
    ]

    htos_calo_images_mc = padding_zeros(mc_data, top=6, bottom=6) / calo_scaler
    htos_calo_images_im2im = generated_images
    htos_calo_images_gan = padding_zeros(gan_data.reshape([-1, 56, 64]), top=4, bottom=4)
    htos_calo_images_gan = np.clip(htos_calo_images_gan, a_min=0, a_max=calo_scaler) / calo_scaler

    for func_idx, (func, params) in enumerate(use_functions.items()):
        if func.__name__ in ["get_number_of_activated_cells", "get_max_energy"]:
            build_histogram(true=htos_calo_images_mc, fake=htos_calo_images_im2im, fake2=htos_calo_images_gan, function=func,
                            name=colnames[func_idx], epoch="", folder=None, ax=axes[func_idx], labels=["Geant4", "Im2Im", "CGAN"], **params)
        else:
            build_histogram(true=htos_calo_images_mc, fake=htos_calo_images_im2im, fake2=htos_calo_images_gan, function=func,
                            name=colnames[func_idx], epoch="", folder=None, ax=axes[func_idx], labels=["Geant4", "Im2Im", "CGAN"], **params)

    build_histogram_HTOS(true=htos_calo_images_mc, fake=htos_calo_images_im2im, fake2=htos_calo_images_gan,
                         energy_scaler=calo_scaler, threshold=3600, real_ET=tracker_real_ET, labels=["Geant4", "Im2Im", "CGAN"],
                         ax1=axes[-3], ax2=axes[-2])

    axes[-1].scatter(tracker_real_ET, get_energies(htos_calo_images_mc), label="Geant4", alpha=0.05)
    axes[-1].scatter(tracker_real_ET, get_energies(htos_calo_images_im2im), label="Im2Im", alpha=0.05)
    axes[-1].scatter(tracker_real_ET, get_energies(htos_calo_images_gan), label="CGAN", alpha=0.05)
    axes[-1].legend()

    figs.append(fig2)

    use_functions[get_center_of_mass_r] = {"image_shape": image_shape}
    for idx in range(10):
        print("Single images:", idx+1, "/", 10)
        use_functions[get_energy_resolution] = {"real_ET": tracker_real_ET[idx], "energy_scaler": calo_scaler}
        figs.append(Generator.build_simulated_events(condition=gan_data_m[idx],
                                     tracker_image=tracker_images[idx],
                                     calo_image=htos_calo_images_mc[idx],
                                     cgan_image=htos_calo_images_gan[idx],
                                     n=500,
                                     eval_functions=use_functions,
                                     title=model_path
        )[0])

    savefigs(figures=figs, save_path=model_path+"/Evaluation.pdf")
    tf.reset_default_graph()