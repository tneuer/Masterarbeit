#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-11-04 12:29:59
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
import tensorflow as tf
import matplotlib as mpl
import initialization as init
import matplotlib.pyplot as plt

from TrainedGenerator import TrainedGenerator
from TrainedIm2Im import TrainedIm2Im
from functionsOnImages import get_energies, get_max_energy, get_number_of_activated_cells, get_std_energy
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_energy_resolution, get_mean_image
from functionsOnImages import build_histogram, padding_zeros, clip_outer, separate_images, build_histogram_HTOS
from find_HTOS import plot_calo_image_with_triggered_and_signal_frame

savefolder = "../../Thesis/presentation/figures/Results"
mpl.rcParams["image.cmap"] = "viridis"
def scale_to_grid_coordinate(values, minval, maxval, globalmin=None, globalmax=None):
    """ Covert continuous measurements to index grid with indices from minval to maxval in both dimensions.
    So for a two dimensional image with x and y measurements between 0-1000 and example measurement of
    (250, 730) which should be scaled to a 10x10 pixel image, the output is (2, 7). So if a particle is at
    position (250, 730) the pixel (2, 7) is lit in a 10x10 image.
    """
    if globalmin is None:
        globalmin = values.min()
    if globalmax is None:
        globalmax = values.max()
    scaled_values = (values - globalmin) / (globalmax - globalmin) # Scale to [0, 1]
    index_values = scaled_values * (maxval - minval) + minval # Scale to [minval, maxval]
    index_values = index_values.astype(dtype=int) # Convert to index
    return index_values

#####################################################################################################
# Parameter definition
#####################################################################################################
datasize = "TestingPurpose"
model_vec = "CWGANGP8"
model_im = "BiCycleGAN1" #33

data_path = "../../Data/B2Dmunu/{}".format(datasize)
model_path_vec = "../../Results/PiplusLowerP/{}/".format(model_vec)
meta_path_vec = model_path_vec + "TFGraphs/"
config_path_vec = model_path_vec + "config.json"

model_path_im = "../../Results/B2Dmunu/{}/".format(model_im)
meta_path_im = model_path_im + "TFGraphs/"
config_path_im = model_path_im + "config.json"

model_path_direct = "../../Results/ServerTemp/B2DmunuTracker/1Good/BiCycleGANTracker14/"
meta_path_direct = model_path_direct + "TFGraphs/"
config_path_direct = model_path_direct + "config.json"

nr_samples = 20000
compare_to = "STP"

#####################################################################################################
# Data loading
#####################################################################################################

with open(config_path_vec, "r") as f:
    config = json.load(f)
    padding = config["padding"]
    keep_cols = config["keep_cols"]

energy_scaler = 6120
with open(data_path+"/calo_images.pickle", "rb") as f:
    calo_images = padding_zeros(pickle.load(f)[:nr_samples], top=6, bottom=6) / 1000
with open(data_path+"/tracker_input.pickle", "rb") as f:
    tracker_input = pickle.load(f)[:nr_samples]
with open(data_path+"/tracker_events.pickle", "rb") as f:
    tracker_events = pickle.load(f)[:nr_samples]
    tracker_real_ET = tracker_events["real_ET"].apply(sum).to_numpy() / 1000
with open(data_path+"/Trained/PiplusLowerP_CWGANGP8_out_1.pickle", "rb") as f:
    cgan_images = pickle.load(f)[:nr_samples]
with open(data_path+"/tracker_images.pickle", "rb") as f:
    tracker_images = padding_zeros(pickle.load(f)[:nr_samples], top=6, bottom=6) / energy_scaler

assert tracker_input.shape[0] == calo_images.shape[0] == tracker_events.shape[0], (
    "Shape mismatch: {}, {}, {}.".format(tracker_input.shape, calo_images.shape, tracker_events.shape)
)

#####################################################################################################
# Generate images with all three GANs
#   1) Vector to image
#   2) Indirect image to image
#   3) Direct image to image
#####################################################################################################
# Generator_vec = TrainedGenerator(path_to_meta=meta_path_vec, path_to_config=config_path_vec)
# generated_images_cgan = Generator_vec.generate_batches(list_of_inputs=tracker_input, batch_size=100)
# generated_images_cgan = np.clip(padding_zeros(generated_images_cgan, top=4, bottom=4), a_min=0, a_max=1)
generated_images_cgan = np.clip(padding_zeros(cgan_images, top=4, bottom=4), a_min=0, a_max=energy_scaler).reshape([nr_samples, 64, 64]) / energy_scaler

if os.path.exists(data_path+"/Trained/im2im_images.pickle"):
    with open(data_path+"/Trained/im2im_images.pickle", "rb") as f:
        generated_images_im2im = pickle.load(f)
    with open(data_path+"/Trained/direct_images.pickle", "rb") as f:
        generated_images_direct = pickle.load(f)
else:
    tf.reset_default_graph()
    Generator_im  = TrainedIm2Im(path_to_meta=meta_path_im, path_to_config=config_path_im)
    generated_images_im2im = Generator_im.generate_batches(list_of_inputs=generated_images_cgan.reshape([nr_samples, 64, 64, 1]), batch_size=100)

    if compare_to == "TIP":
        tf.reset_default_graph()
        Generator_im_direct  = TrainedIm2Im(path_to_meta=meta_path_direct, path_to_config=config_path_direct)
        generated_images_direct = Generator_im_direct.generate_batches(list_of_inputs=tracker_images.reshape([nr_samples, 64, 64, 1]),
                                                                       batch_size=100)
        generated_images_direct = clip_outer(generated_images_direct, clipval=0.25)*energy_scaler/1000

    generated_images_im2im = clip_outer(generated_images_im2im, clipval=0.25)*energy_scaler/1000

if compare_to == "TIP":
    generated_images_comparison = generated_images_direct
elif compare_to == "STP":
    generated_images_comparison = generated_images_cgan
else:
    raise ValueError("compare_to must be 'TIP' or 'STP'.")

if nr_samples == 20000:
    with open(data_path+"/Trained/im2im_images.pickle", "wb") as f:
        pickle.dump(generated_images_im2im, f)
    with open(data_path+"/Trained/direct_images.pickle", "wb") as f:
        pickle.dump(generated_images_direct, f)

#####################################################################################################
# Check statistics
#####################################################################################################
# generated_images_cgan = clip_outer(generated_images_cgan, clipval=0.25)*energy_scaler/1000
# print(calo_images.shape, np.mean(calo_images), np.min(calo_images), np.max(calo_images))
# print(generated_images_cgan.shape, np.mean(generated_images_cgan), np.min(generated_images_cgan), np.max(generated_images_cgan))
# print(generated_images_im2im.shape, np.mean(generated_images_im2im), np.min(generated_images_im2im), np.max(generated_images_im2im))

# inner, outer = separate_images(calo_images)
# print(outer.shape, np.mean(outer), np.min(outer), np.max(outer))
# inner, outer = separate_images(generated_images_cgan)
# print(outer.shape, np.mean(outer), np.min(outer), np.max(outer))
# inner, outer = separate_images(generated_images_im2im)
# print(outer.shape, np.mean(outer), np.min(outer), np.max(outer))


#####################################################################################################
# Generate evaluation metrics for the cGAN
#####################################################################################################
# if compare_to == "STP":
#     generated_images_comparison = clip_outer(generated_images_comparison, clipval=0.25)*energy_scaler/1000
# functions = [
#     get_energies, get_max_energy, get_number_of_activated_cells, get_std_energy, get_energy_resolution
# ]
# xlabels = [
#     "Total energy (GeV)", "Maximum energy (GeV)", "Number of activated cells", r"Standard deviation $E_T$ (GeV)",
#     r"Resolution $E_{tracker} - E_{image}$"
# ]
# kwargs = [{}, {}, {"threshold": 6/1000}, {}, {"real_ET": tracker_real_ET}]
# labels = ["Geant4", compare_to, "CIR"]
# absvals = [100, np.inf, 1000, np.inf, np.inf, 100]

# fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
# fig.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.99, wspace=0.3, hspace=0.4)
# axs = np.ravel(axs)
# for i, func in enumerate(functions):
#     kwarg = kwargs[i]
#     build_histogram(true=calo_images, fake=generated_images_comparison, fake2=generated_images_im2im, function=func,
#                     name="", epoch="", ax=axs[i], labels=labels, absval=absvals[i], nr_bins=40, **kwarg)
#     axs[i].set_xlabel(xlabels[i])

# axs[-1].scatter(tracker_real_ET, get_energies(calo_images), label=labels[0], alpha=0.8, s=0.5)
# axs[-1].scatter(tracker_real_ET, get_energies(generated_images_comparison), label=labels[1], alpha=0.8, s=0.5)
# axs[-1].scatter(tracker_real_ET, get_energies(generated_images_im2im), label=labels[2], alpha=0.8, s=0.5)
# axs[-1].legend()
# axs[-1].set_xlim((0, 110))
# axs[-1].set_ylim((0, 110))
# axs[-1].plot([0, 1], [0, 1], transform=axs[-1].transAxes, c="black", linestyle="--")
# axs[-1].set_xlabel(r"Tracker $E_T$")
# axs[-1].set_ylabel(r"HCAL $E_T$")

# plt.savefig(savefolder+"/im2im_energy_{}.png".format(compare_to), dpi=300)


#####################################################################################################
# Generate example images
#####################################################################################################
if compare_to == "STP":
    generated_images_cgan = clip_outer(generated_images_cgan, clipval=0.25)*energy_scaler/1000
def set_title_for(ax, im, name):
    im = im.reshape([1, 64, 64])
    energy, max_energy = get_energies(im), get_max_energy(im)
    ax.set_title("%s\nEnergy: %.2f GeV\nMaximum Energy: %.2f GeV" % (name, energy, max_energy))

np.random.seed(20201004)
nr_plotted = 0
while nr_plotted < 5:
    i = np.random.randint(0, nr_samples, size=1)[0]
    calo_max, im2im_max, direct_max = np.max(calo_images[i]), np.max(generated_images_im2im[i]), np.max(generated_images_direct[i])
    if ( abs(calo_max-im2im_max)/calo_max > 0.4 )  or ( abs(calo_max-direct_max)/calo_max > 0.4 ):
        continue
    max_e = np.max([calo_max, im2im_max, direct_max])
    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(12, 3.5))
    fig.subplots_adjust(left=0.04, bottom=0.0, right=0.98, top=1.0, wspace=None, hspace=None)
    axs[0].imshow(tracker_images[i])
    # set_title_for(ax=axs[0], im=tracker_images[i], name="Tracker")
    axs[1].imshow(generated_images_cgan[i])
    # set_title_for(ax=axs[1], im=generated_images_cgan[i], name="STP")
    axs[2].imshow(calo_images[i], vmin=0, vmax=max_e)
    # set_title_for(ax=axs[2], im=calo_images[i], name="Geant4")
    axs[3].imshow(generated_images_im2im[i], vmin=0, vmax=max_e)
    # set_title_for(ax=axs[3], im=generated_images_im2im[i], name="CIR")
    im = axs[4].imshow(generated_images_direct[i], vmin=0, vmax=max_e)
    # set_title_for(ax=axs[4], im=generated_images_direct[i], name="TIP")
    axs[-1].axis("off")
    plt.colorbar(im, ax=axs[-1], shrink=0.4, anchor=-0.2, fraction=1.1, label="Transverse Energy (GeV)")
    plt.savefig(savefolder+"/full_event_idx_{}.png".format(i))
    nr_plotted += 1


#####################################################################################################
# Image variety
#####################################################################################################
# n = 1000
# idx = 57
# eval_functions = {
#     get_energies: {}, get_max_energy: {}, get_number_of_activated_cells: {"threshold": 6/1000},
#     get_std_energy: {}, get_energy_resolution: {"real_ET": tracker_real_ET[idx]}
# }
# xlabels = [
#     "Total energy (GeV)", "Maximum energy (GeV)", "Number of activated cells", r"Standard deviation $E_T$ (GeV)",
#     r"Resolution $E_{tracker} - E_{image}$"
# ]
# func_titles = ["" for _ in eval_functions]

# tf.reset_default_graph()
# Generator_im_direct  = TrainedIm2Im(path_to_meta=meta_path_direct, path_to_config=config_path_direct)
# references = np.array([Generator_im_direct.generate_from_condition([tracker_images[idx].reshape([64, 64, 1])]) for _ in range(n)])
# references = np.array([r.reshape([64, 64]) for r in references])
# tf.reset_default_graph()
# Generator_im  = TrainedIm2Im(path_to_meta=meta_path_im, path_to_config=config_path_im)

# fig, ax = Generator_im.build_simulated_events(
#     condition=generated_images_cgan[idx].reshape([64, 64, 1]),
#     tracker_image=tracker_images[idx],
#     calo_image=calo_images[idx],
#     cgan_image=generated_images_cgan[idx],
#     eval_functions=eval_functions,
#     n=n, title="",  reference_images=references*energy_scaler/1000, func_titles=func_titles, x_labels=xlabels, scaler=energy_scaler/1000
# )
# plt.savefig(savefolder+"/event_variety_idx_{}.png".format(idx))


#####################################################################################################
# HTIS & HTOS
#####################################################################################################
# x_coordinates = ["Pi1_x_projection", "Pi2_x_projection", "K_x_projection"]
# y_coordinates = ["Pi1_y_projection", "Pi2_y_projection", "K_y_projection"]

# for x_coordinate in x_coordinates:
#     globalmin_x = np.min(tracker_events["x_projections"].apply(np.min))
#     globalmax_x = np.max(tracker_events["x_projections"].apply(np.max))
#     tracker_events[x_coordinate+"_pixel"] = scale_to_grid_coordinate(
#         values=tracker_events[x_coordinate], minval=0, maxval=tracker_images.shape[2]-1, globalmin=globalmin_x, globalmax=globalmax_x
#     )
# for y_coordinate in y_coordinates:
#     globalmin_y = np.min(tracker_events["y_projections"].apply(np.min))
#     globalmax_y = np.max(tracker_events["y_projections"].apply(np.max))
#     tracker_events[y_coordinate+"_pixel"] = scale_to_grid_coordinate(
#         values=tracker_events[y_coordinate], minval=0, maxval=tracker_images.shape[1]-1, globalmin=globalmin_y, globalmax=globalmax_y
#     )


# fr_width = 7
# is_TIS_or_TOS_calo = np.array([plot_calo_image_with_triggered_and_signal_frame(
#         tracker_images=tracker_images, tracker_events=tracker_events, fr_width=7, idx=idx,
#         calo_images=calo_images, threshold=3.2, padding=None, show=False, axs=None
#     )[3:5] for idx in range(nr_samples)])

# is_TIS_or_TOS_im2im = np.array([plot_calo_image_with_triggered_and_signal_frame(
#         tracker_images=tracker_images, tracker_events=tracker_events, fr_width=7, idx=idx,
#         calo_images=generated_images_im2im, threshold=3.2, padding=None, show=False, axs=None
#     )[3:5] for idx in range(nr_samples)])

# is_TIS_or_TOS_direct = np.array([plot_calo_image_with_triggered_and_signal_frame(
#         tracker_images=tracker_images, tracker_events=tracker_events, fr_width=7, idx=idx,
#         calo_images=generated_images_direct, threshold=3.2, padding=None, show=False, axs=None
#     )[3:5] for idx in range(nr_samples)])

# labels = ["Geant4", "TIP", "CIR"]
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 14))
# fig.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.99, wspace=0.4)
# build_histogram_HTOS(true=calo_images, fake=generated_images_direct, energy_scaler=1, threshold=None,
#                      real_ET=tracker_real_ET, fake2=generated_images_im2im, labels=labels, ax1=axs[0, 0], ax2=axs[0, 1],
#                      triggered=[is_TIS_or_TOS_calo[:, 0], is_TIS_or_TOS_direct[:, 0], is_TIS_or_TOS_im2im[:, 0]])
# build_histogram_HTOS(true=calo_images, fake=generated_images_direct, energy_scaler=1, threshold=None,
#                      real_ET=tracker_real_ET, fake2=generated_images_im2im, labels=labels, ax1=axs[1, 0], ax2=axs[1, 1],
#                      triggered=[is_TIS_or_TOS_calo[:, 1], is_TIS_or_TOS_direct[:, 1], is_TIS_or_TOS_im2im[:, 1]])
# axs[0, 0].set_ylabel("HTIS count")
# axs[0, 1].set_ylabel("HTIS rate")
# axs[1, 0].set_ylabel("HTOS count")
# axs[1, 1].set_ylabel("HTOS rate")
# for ax in np.ravel(axs):
#     ax.set_xlabel(r"Tracker $E_T$")
#     ax.set_title("")
# plt.savefig(savefolder+"/HTIS_HTOS.png")


#####################################################################################################
# Timing plot
#####################################################################################################

# load_path = "../../Results/Timing"
# files = [f.path for f in os.scandir(load_path)]
# all_data = pd.DataFrame(columns=pd.read_csv(files[0]).columns)
# for f in files:
#     this_data = pd.read_csv(f)
#     if "Image" in f:
#         this_data["Method"] = "CIR"
#     elif "Vector" in f:
#         this_data["Method"] = "STP"
#     elif "Direct" in f:
#         this_data["Method"] = "TIP"
#     else:
#         raise ValueError("Wrong filename.")
#     all_data = all_data.append(this_data, ignore_index=True)

# all_data.fillna(1, inplace=True)
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
# unique_data = all_data[["GPU", "Method", "batch_size"]].drop_duplicates().sort_values(by=["GPU", "Method", "batch_size"], ascending=[True, False, True])

# colors = {"STP": "orange", "CIR": "green", "TIP": "blue"}
# linestyles = {50: "--", 100: "-", 200: ":", 500: "-."}
# for idx, (gpu, method, bs) in unique_data.iterrows():
#     current_data = all_data.loc[(all_data.GPU==gpu) & (all_data.Method==method) & (all_data.batch_size==bs)]
#     label = method + " --- " + str(bs)
#     print(label, np.max(current_data.cum_time.values))
#     if bs != 200:
#         continue
#     if "GTX" in gpu:
#         if method == "STP":
#             continue
#         axs[0].plot(current_data.cum_batch.values, current_data.cum_time.values, color=colors[method], linestyle=linestyles[bs], label=label)
#     else:
#         axs[1].plot(current_data.cum_batch.values, current_data.cum_time.values, color=colors[method], linestyle=linestyles[bs], label=label)

# x_max = max([axs[0].get_xlim()[1], axs[1].get_xlim()[1]])
# y_max = max([axs[0].get_ylim()[1], axs[1].get_ylim()[1]])

# axs[0].set_xlabel("Created images")
# axs[0].set_ylabel("Cumulative time (s)")
# axs[0].set_title("GeForce GTX 950M")
# axs[0].set_xlim((0, x_max))
# axs[0].set_ylim((0, y_max))
# axs[0].legend(title="Method --- batch size")

# axs[1].set_xlabel("Created images")
# axs[1].set_ylabel("Cumulative time (s)")
# axs[1].set_title("P100 PCIe")
# axs[1].set_xlim((0, x_max))
# axs[1].set_ylim((0, y_max))
# axs[1].legend(title="Method --- batch size")
# plt.savefig(savefolder+"/timing.png", dpi=300)



#####################################################################################################
# Approximating distributions
#####################################################################################################
# position_samples = []
# approximate_distr = {get_energies: [20, 40, 60, 80, 100, 120, 140], get_number_of_activated_cells: [300, 400, 500, 600, 700, 800]}
# kwargs = [{}, {"threshold": 6/1000}]
# legend_titles = ["Tracker energy (GeV)", "Number of cells"]
# x_labels = ["Energy (GeV)", "Number of activated cells"]
# filenames = ["Energy", "Cells"]

# for i, (func, values) in enumerate(approximate_distr.items()):
#     if func == get_energies:
#         fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True, sharey=True)
#         for energy in values:
#             is_valid_e = np.logical_and(0.95*energy < tracker_real_ET, tracker_real_ET < 1.05*energy)
#             if sum(is_valid_e) > 1:
#                 axs[0].hist(get_energies(calo_images[is_valid_e]), density=True, range=(min(values), max(values)), bins=30,
#                             label=str(energy)+" (n={})".format(sum(is_valid_e)), histtype="step")
#                 axs[1].hist(get_energies(generated_images_im2im[is_valid_e]), density=True, range=(min(values), max(values)), bins=30,
#                             label=str(energy)+" (n={})".format(sum(is_valid_e)), histtype="step")
#                 axs[2].hist(get_energies(generated_images_direct[is_valid_e]), density=True, range=(min(values), max(values)), bins=30,
#                             label=str(energy)+" (n={})".format(sum(is_valid_e)), histtype="step")
#     else:
#         fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, sharey=True)
#         for val in values:
#             calo_values = func(calo_images)
#             is_valid = np.logical_and(0.95*val < calo_values, calo_values < 1.05*val)
#             if sum(is_valid) > 1:
#                 axs[0].hist(func(generated_images_im2im[is_valid], **kwargs[i]), density=True, range=(min(values), max(values)), bins=30,
#                             label=str(val)+" (n={})".format(sum(is_valid)), histtype="step")
#                 axs[1].hist(func(generated_images_direct[is_valid], **kwargs[i]), density=True, range=(min(values), max(values)), bins=30,
#                             label=str(val)+" (n={})".format(sum(is_valid)), histtype="step")

#     axs[0].legend(title=legend_titles[i])
#     axs[-1].set_xlabel(x_labels[i])
#     plt.savefig(savefolder+"/full_evenet_{}_spectrum.png".format(filenames[i]))