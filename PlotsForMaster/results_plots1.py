#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-10-31 14:31:35
    # Description :
####################################################################################
"""

import os
import sys
import json
sys.path.insert(1, "../Utilities")
sys.path.insert(1, "../Preprocessing")
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import initialization as init
import matplotlib.pyplot as plt

from TrainedGenerator import TrainedGenerator
from functionsOnImages import get_energies, get_max_energy, get_number_of_activated_cells, get_std_energy
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_energy_resolution, get_mean_image
from functionsOnImages import build_histogram, padding_zeros

savefolder = "../../Thesis/figures/Results"
mpl.rcParams["image.cmap"] = "YlOrBr"

def standardize_data(data, scaler, exclude=None):
    standardized_data = data.drop(exclude, axis=1, inplace=False)
    colnames = standardized_data.columns.values
    standardized_data = pd.DataFrame(data=scaler.transform(standardized_data), columns=colnames, index=data.index)
    data = pd.concat([standardized_data, data[exclude]], axis=1, sort=False)
    return data

#####################################################################################################
# Parameter definition
#####################################################################################################
datasize = "TestingPurpose"
piplus_energy = "PiplusLowerP"
model = "CWGANGP8"

data_path = "../../Data/{}/{}".format(piplus_energy, datasize)
model_path = "../../Results/{}/{}/".format(piplus_energy, model)
meta_path = model_path + "TFGraphs/"
config_path = model_path + "config.json"

#####################################################################################################
# Data loading
#####################################################################################################
Generator = TrainedGenerator(path_to_meta=meta_path, path_to_config=config_path)
with open(config_path, "r") as f:
    config = json.load(f)
    padding = config["padding"]
    keep_cols = config["keep_cols"]
data, scaler = init.load_processed_data(data_path=data_path, mode="test", return_scaler=True)
energy_scaler = scaler["Calo"]
x_min = np.min(data["Tracker"]["x_projections"])
x_max = np.max(data["Tracker"]["x_projections"])
y_min = np.min(data["Tracker"]["y_projections"])
y_max = np.max(data["Tracker"]["y_projections"])

tracker_events = init.load_data(data_path=data_path, mode="tracker")["tracker_events"]
tracker_events = tracker_events.iloc[data["Idx"]]

tracker_events_std = standardize_data(data=tracker_events, scaler=scaler["Tracker"], exclude=["theta", "phi", "region"])
is_not_outlier = (tracker_events_std["real_ET"]<4) | (tracker_events_std["momentum_p"]<4)
tracker_events = tracker_events.loc[is_not_outlier, ]

with open(data_path+"/tracker_images.pickle", "rb") as f:
    tracker_images = pickle.load(f)[is_not_outlier]


assert tracker_images.shape[0] == data["Calo"].shape[0] == data["Tracker"].shape[0] == tracker_events.shape[0], (
    "Shape mismatch: {}, {}, {}, {}.".format(tracker_images.shape, data["Calo"].shape, data["Tracker"].shape, tracker_events.shape)
)

#####################################################################################################
# Generate energy distribution for different input energies
#####################################################################################################
# nr_samples = 1000
# x_positions = np.linspace(x_min, x_max, 5)
# position_samples = []
# for x in x_positions:
#     samples_x = [x for i in range(nr_samples)]
#     samples_y = np.random.uniform(y_min, y_max, nr_samples)
#     samples_energy = np.random.uniform(0, 3, nr_samples)
#     inputs = [[[x, y, e]] for x, y, e in zip(samples_x, samples_y, samples_energy)]
#     generated_samples = Generator.generate_batches(list_of_inputs=inputs, batch_size=100)
#     centers_of_mass = get_center_of_mass_x(generated_samples)
#     position_samples.append(centers_of_mass)


# labels = [r"$\pi^+_x: {}$".format(x) for x in x_positions]
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.hist(position_samples, label=labels, bins=30)
# ax.set_xlabel("Pion energy (GeV)")
# ax.legend()
# plt.savefig(savefolder+"/xSpectrum.png")


#####################################################################################################
# Generate evaluation metrics for the cGAN
#####################################################################################################
# nr_samples = 20000
# calo_images = data["Calo"][:nr_samples]*energy_scaler/1000
# inputs = data["Tracker"][keep_cols].values[:nr_samples]
# inputs = [[track] for track in inputs]
# generated_images = Generator.generate_batches(inputs, batch_size=100)*energy_scaler/1000
# functions = [
#     get_energies, get_max_energy, get_number_of_activated_cells, get_center_of_mass_x, get_energy_resolution
# ]
# xlabels = [
#     "Total energy (GeV)", "Maximum energy (GeV)", "Number of activated cells", "x center-of-energy",
#     r"Resolution $E_{tracker} - E_{image}$"
# ]
# kwargs = [{}, {}, {"threshold": 6/1000}, {}, {"real_ET": (tracker_events["real_ET"]/1000)[:nr_samples]}]
# labels = ["Geant4", "cGAN"]
# absvals = [11, 7, 30, np.inf, 20, np.inf]

# fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
# plt.subplots_adjust(wspace=0.5)
# axs = np.ravel(axs)
# for i, func in enumerate(functions):
#     kwarg = kwargs[i]
#     build_histogram(true=calo_images, fake=generated_images, function=func, name="", epoch="", ax=axs[i],
#                     labels=labels, absval=absvals[i], nr_bins=40, **kwarg)
#     axs[i].set_xlabel(xlabels[i])

# axs[-1].scatter((tracker_events["real_ET"]/1000)[:nr_samples], get_energies(calo_images), label=labels[0], alpha=0.8, s=0.5)
# axs[-1].scatter((tracker_events["real_ET"]/1000)[:nr_samples], get_energies(generated_images), label=labels[1], alpha=0.8, s=0.5)
# axs[-1].legend()
# axs[-1].set_xlim((0, 15))
# axs[-1].set_ylim((0, 15))
# axs[-1].plot([0, 1], [0, 1], transform=axs[-1].transAxes, c="black", linestyle="--")
# axs[-1].set_xlabel(r"Tracker $E_T$")
# axs[-1].set_ylabel(r"HCAL $E_T$")
# plt.savefig(savefolder+"/cGAN_energy.png", dpi=300)


#####################################################################################################
# Generate example images
#####################################################################################################
# nr_samples = 500
# calo_images = data["Calo"][:nr_samples]*energy_scaler/1000
# inputs = data["Tracker"][keep_cols].values[:nr_samples]
# inputs = [[track] for track in inputs]
# generated_images = Generator.generate_batches(inputs, batch_size=100)*energy_scaler/1000
# plt_idxs = []

# calo_images = padding_zeros(calo_images, top=2, bottom=2)
# tracker_images = np.flip(padding_zeros(tracker_images, top=2, bottom=2)[:nr_samples] / 1000, axis=1)

# def set_title_for(ax, im):
#     im = im.reshape([1, 56, 64])
#     energy, max_energy = get_energies(im), get_max_energy(im)
#     ax.set_title("Energy: %.2f GeV\nMaximum Energy: %.2f GeV" % (energy, max_energy))

# for i in range(nr_samples):
#     max_e = np.max([np.max(calo_images[i]), np.max(generated_images[i])])
#     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 12))
#     axs[0].imshow(calo_images[i], vmin=0, vmax=max_e)
#     set_title_for(ax=axs[0], im=calo_images[i])
#     axs[1].imshow(generated_images[i], vmin=0, vmax=max_e)
#     set_title_for(ax=axs[1], im=generated_images[i])
#     fig.suptitle("Index: {}".format(i))
#     plt.show()


#####################################################################################################
# Image variety
#####################################################################################################
# idx = 8
# n = 1000
# eval_functions = {
#     get_energies: {}, get_max_energy: {}, get_number_of_activated_cells: {"threshold": 6/1000},
#     get_std_energy: {}, get_energy_resolution: {"real_ET": tracker_events.iloc[idx]["real_ET"]}
# }
# xlabels = [
#     "Total energy (GeV)", "Maximum energy (GeV)", "Number of activated cells", r"Standard deviation $E_T$ (GeV)",
#     r"Resolution $E_{tracker} - E_{image}$"
# ]
# func_titles = ["" for _ in eval_functions]

# ref = tracker_events.iloc[idx]
# inpt = [data["Tracker"][keep_cols].values[idx]]
# calo_images = padding_zeros(data["Calo"]*energy_scaler/1000, top=2, bottom=2)
# is_reference = (
#         (tracker_events["theta"] > ref["theta"]-0.01) &
#         (tracker_events["theta"] < ref["theta"]+0.01) &
#         (np.abs(tracker_events["momentum_pt"]) > np.abs(ref["momentum_pt"]-0.1*ref["momentum_pt"])) &
#         (np.abs(tracker_events["momentum_pt"]) < np.abs(ref["momentum_pt"]+0.1*ref["momentum_pt"]))
# )
# references = calo_images[is_reference]

# fig, ax = Generator.build_simulated_events(
#     condition=inpt,
#     tracker_image=tracker_images[idx],
#     calo_image=calo_images[idx],
#     eval_functions=eval_functions,
#     n=n, title="",  reference_images=references, func_titles=func_titles, x_labels=xlabels, scaler=energy_scaler/1000
# )
# plt.savefig(savefolder+"/piplus_variety_idx_{}.png".format(idx))