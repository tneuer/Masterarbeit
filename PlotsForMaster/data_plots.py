#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-10-26 18:01:12
    # Description :
####################################################################################
"""

import os
import sys
sys.path.insert(1, "../Utilities")
sys.path.insert(1, "../Preprocessing")
import pickle
import matplotlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from functionsOnImages import get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_energy_resolution, get_mean_image

data_path = "../../Data/PiplusMixedP/TestingPurpose"
savefolder = "../../Thesis/presentation/figures/Data"
# mpl.rcParams["image.cmap"] = "YlOrBr"

if os.path.exists(data_path+"/tracker_events.csv"):
    tracker_events = pd.read_csv(data_path+"/tracker_events.csv")
else:
    with open(data_path+"/tracker_events.pickle", "rb") as f:
        tracker_events = pickle.load(f)
with open(data_path+"/tracker_images.pickle", "rb") as f:
    tracker_images = pickle.load(f)
with open(data_path+"/calo_images.pickle", "rb") as f:
    calo_images = pickle.load(f)
with open(data_path+"/calo_events.pickle", "rb") as f:
    calo_events = pickle.load(f)

def scale_to_grid_coordinate(values, minval, maxval, globalmin=None, globalmax=None):
    if globalmin is None:
        globalmin = values.min()
    if globalmax is None:
        globalmax = values.max()
    scaled_values = (values - globalmin) / (globalmax - globalmin) # Scale to [0, 1]
    index_values = scaled_values * (maxval - minval) + minval # Scale to [minval, maxval]
    index_values = index_values.astype(dtype=int) # Convert to index
    return index_values

def my_hist(ax, values, nr_bins, x_label, y_label, x_rot=30, y_rot=60, nr_x_ticks=5, nr_y_ticks=5, props=None,
            descr=None, descr_props=None, digit=-3):
    n, _, _ = ax.hist(values, bins=nr_bins)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    min_x, max_x = np.round(np.min(values), digit), np.round(np.max(values), digit)
    x_stepsize = np.round( (max_x - min_x) / nr_x_ticks, digit)
    xticks = np.arange(min_x, max_x+x_stepsize, x_stepsize).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels=xticks, rotation=x_rot, ha='right')

    max_y = np.round(np.max(n), -3)
    y_stepsize = int(max_y / nr_y_ticks)
    y_stepsize = np.round(y_stepsize, -(len(str(y_stepsize))-1))
    yticks = np.arange(0, max_y+y_stepsize, y_stepsize).astype(int)
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels=yticks, rotation=y_rot, ha='right')
    if props is not None:
        this_props = props.copy()
        textstr_x = '\n'.join((
                r'$\min=%.0f$' % (np.min(values), ),
                r'$\mu=%.0f$' % (np.mean(values), ),
                r'$\sigma=%.0f$' % (np.std(values), ),
                r'$\max=%.0f$' % (np.max(values), )
        ))
        x_pos = this_props.pop("x")
        y_pos = this_props.pop("y")
        fs = this_props.pop("fs")
        ax.text(x_pos, y_pos, textstr_x, transform=ax.transAxes, fontsize=fs,
                verticalalignment='top', bbox=this_props)

    if descr is not None:
        this_descr_props = descr_props.copy()
        x_pos = this_descr_props.pop("x")
        y_pos = this_descr_props.pop("y")
        fs = this_descr_props.pop("fs")
        ax.text(x_pos, y_pos, descr, transform=ax.transAxes, fontsize=fs,
                verticalalignment='top', bbox=this_descr_props)


#############################################################################################################
############ Build mean calo images
#############################################################################################################
# mpl.rcParams.update({"font.size": 6})
# fig, axs = plt.subplots(nrows=1, ncols=4, gridspec_kw={'width_ratios': [5, 5, 5, 1]}, figsize=(8,2.6))
# fig.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.99, wspace=0.3, hspace=0.4)
# inner_mean = get_mean_image(calo_events["calo_ET_inner"])
# outer_mean = get_mean_image(calo_events["calo_ET_outer"])
# calo_mean = get_mean_image(calo_images)
# E_max = np.max([np.max(inner_mean), np.max(outer_mean), np.max(calo_mean)])

# axs[0].imshow(inner_mean, vmin=0, vmax=E_max)
# axs[0].set_title("Inner:\nGeometry: {}\nactive cells: {} ".format(inner_mean.shape, np.sum(inner_mean>0)))
# axs[1].imshow(outer_mean, vmin=0, vmax=E_max)
# axs[1].set_title("Outer:\nGeometry: {}\nactive cells: {} ".format(outer_mean.shape, np.sum(outer_mean>0)))
# im = axs[2].imshow(calo_mean, vmin=0, vmax=E_max)
# axs[2].set_title("Total:\nGeometry: {}\nactive cells: {} ".format(calo_mean.shape, np.sum(calo_mean>0)))
# axs[3].axis("off")
# cbar = plt.colorbar(im, ax=axs[-1], shrink=0.4, anchor=0.2, fraction=1)
# cbar.set_label("Mean energy (MeV)", rotation=270)
# plt.tight_layout()
# plt.savefig(savefolder+"/mean_calorimeters.png", dpi=400)

#############################################################################################################
############ Plot calorimeter distribution
#############################################################################################################

# mpl.rcParams.update({"font.size": 6})
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5, x=0.65, y=0.95, fs=6)
# descr_props = dict(facecolor='white', x=0.65, y=0.6, fs=8)
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 5))
# plt.subplots_adjust(wspace=0.4, hspace=0.45)
# axs = np.ravel(axs)
# nr_bins = 30

# my_hist(
#         ax=axs[0], values=get_energies(calo_images)/1000, nr_bins=nr_bins, props=props,
#         x_label="Image transverse energy [GeV]", y_label="Count",
#         descr=r"$\sum_{ij} E_{T,ijk}$", descr_props=descr_props, digit=0
# )
# my_hist(
#         ax=axs[1], values=get_max_energy(calo_images)/1000, nr_bins=nr_bins, props=props,
#         x_label=r"Image max transverse energy [GeV]", y_label="Count",
#         descr=r"$\max_{ij} (E_{T,ijk})$", descr_props=descr_props, digit=0
# )
# nr_cells = get_number_of_activated_cells(calo_images)
# my_hist(
#         ax=axs[2], values=nr_cells[nr_cells<=30], nr_bins=30, props=props,
#         x_label=r"Number of activated cells", y_label="Count",
#         descr=r"$\sum_{ij} [E_{T,ijk} > 6MeV]$", descr_props=descr_props, digit=0
# )
# my_hist(
#         ax=axs[3], values=get_center_of_mass_x(calo_images), nr_bins=nr_bins, props=props,
#         x_label=r"Image x-center-of-mass [x pixel]", y_label="Count",
#         descr=r"$\dfrac{\sum_{ij} x_{ij} \cdot E_{T,ijk}}{\sum_{ij} E_{T,ijk}}$", descr_props=dict(facecolor='white', x=0.05, y=0.95, fs=5),
#         digit=-1
# )
# resolution = get_energy_resolution(calo_images, tracker_events["real_ET"])/1000
# my_hist(
#         ax=axs[4], values=resolution[np.abs(resolution)<15], nr_bins=nr_bins, props=props,
#         x_label=r"Image resolution [GeV]", y_label="Count",
#         descr=r"$E_{T,Tracker} - \sum_{ij} E_{T,ijk}$", descr_props=dict(facecolor='white', x=0.55, y=0.6, fs=7), digit=0
# )
# use_images = np.logical_and(get_energies(calo_images[:5000]) < 15000, tracker_events["real_ET"][:5000] < 30000)
# axs[5].scatter(tracker_events["real_ET"][:5000][use_images]/1000, get_energies(calo_images[:5000])[use_images]/1000, s=0.5)
# axs[5].plot([0, 1], [0, 1], transform=axs[5].transAxes, c="black", linestyle="--")
# axs[5].set_xlabel("Tracker energies [GeV]")
# axs[5].set_ylabel("Calorimeter energies [GeV]")
# axs[5].set_xlim((0, 20))
# ticks = [0, 5, 10, 15, 20]
# axs[5].set_xticks(ticks)
# axs[5].set_xticklabels(labels=ticks, rotation=30, ha='right')
# axs[5].set_yticks(ticks)
# axs[5].set_yticklabels(labels=ticks, rotation=60, ha='right')
# plt.savefig(savefolder+"/dist_calorimeter.png", dpi=400)

#############################################################################################################
############ Plot example tracker and calo image
#############################################################################################################
# example_idx = 7

# mpl.rcParams.update({"font.size": 30})
# te_idx = tracker_events.iloc[example_idx, ][["x_projections", "y_projections", "real_ET"]]
# min_x, max_x = np.min(tracker_events.x_projections), np.max(tracker_events.x_projections)
# min_y, max_y = np.min(tracker_events.y_projections), np.max(tracker_events.y_projections)
# max_e = np.max(tracker_events.real_ET)
# grid_x = scale_to_grid_coordinate(te_idx["x_projections"], minval=0, maxval=64, globalmin=min_x, globalmax=max_x)
# grid_y = scale_to_grid_coordinate(te_idx["y_projections"], minval=0, maxval=52, globalmin=min_y, globalmax=max_y)
# print(te_idx, "-->", (grid_x, grid_y, te_idx["real_ET"]/max_e))

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.imshow(calo_images[example_idx])
# ax.set_xlabel("X cell")
# ax.set_ylabel("Y cell")
# ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
# plt.savefig(savefolder+"/ex_calorimeter.png", dpi=600)


# fig, ax = plt.subplots(figsize=(12, 8))
# ax.imshow(tracker_images[example_idx])
# ax.set_xlabel("X cell")
# ax.set_ylabel("Y cell")
# ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
# plt.savefig(savefolder+"/ex_tracker.png", dpi=600)


#############################################################################################################
############ Plot tracker distribution
#############################################################################################################
# mpl.rcParams.update({"font.size": 14})
# nr_bins = 20
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# fig, ax = plt.subplots(figsize=(16, 8), nrows=1, ncols=3)
# plt.subplots_adjust(wspace=0.25)
# ax[0].hist(tracker_events["x_projections"], bins=nr_bins)
# ax[0].set_xlabel("x_projections (mm)")
# ax[0].set_ylabel("Count")
# xticks = np.arange(-4500, 4501, 1500)
# ax[0].set_xticks(xticks)
# ax[0].set_xticklabels(labels=xticks, rotation=30, ha='right')
# yticks = np.arange(500, 3501, 500)
# ax[0].set_yticks(yticks)
# ax[0].set_yticklabels(labels=yticks, rotation=60, ha='right')
# textstr_x = '\n'.join((
#         r'$\min=%.0f$' % (np.min(tracker_events["x_projections"]), ),
#         r'$\mu=%.0f$' % (np.mean(tracker_events["x_projections"]), ),
#         r'$\sigma=%.0f$' % (np.std(tracker_events["x_projections"]), ),
#         r'$\max=%.0f$' % (np.max(tracker_events["x_projections"]), )
# ))
# ax[0].text(0.62, 0.95, textstr_x, transform=ax[0].transAxes, fontsize=13,
#                          verticalalignment='top', bbox=props)

# ax[1].hist(tracker_events["y_projections"], bins=nr_bins)
# ax[1].set_xlabel("y_projections (mm)")
# ax[1].set_ylabel("Count")
# xticks = np.arange(-3000, 3001, 1500)
# ax[1].set_xticks(xticks)
# ax[1].set_xticklabels(labels=xticks, rotation=30, ha='right')
# yticks = np.arange(500, 3501, 500)
# ax[1].set_yticks(yticks)
# ax[1].set_yticklabels(labels=yticks, rotation=60, ha='right')
# textstr_y = '\n'.join((
#         r'$\min=%.0f$' % (np.min(tracker_events["y_projections"]), ),
#         r'$\mu=%.0f$' % (np.mean(tracker_events["y_projections"]), ),
#         r'$\sigma=%.0f$' % (np.std(tracker_events["y_projections"]), ),
#         r'$\max=%.0f$' % (np.max(tracker_events["y_projections"]), )
# ))
# ax[1].text(0.62, 0.95, textstr_y, transform=ax[1].transAxes, fontsize=13,
#                          verticalalignment='top', bbox=props)

# ax[2].hist(tracker_events["real_ET"][tracker_events.real_ET < 20000], bins=nr_bins)
# ax[2].set_xlabel("real_ET (MeV)")
# ax[2].set_ylabel("Count")
# xticks = np.arange(0, 20000, 4000)
# ax[2].set_xticks(xticks)
# ax[2].set_xticklabels(labels=xticks, rotation=30, ha='right')
# yticks = yticks = np.arange(1000, 6001, 1000)
# ax[2].set_yticks(yticks)
# ax[2].set_yticklabels(labels=yticks, rotation=60, ha='right')
# textstr_et = '\n'.join((
#         r'$\min=%.0f$' % (np.min(tracker_events["real_ET"]), ),
#         r'$\mu=%.0f$' % (np.mean(tracker_events["real_ET"]), ),
#         r'$\sigma=%.0f$' % (np.std(tracker_events["real_ET"]), ),
#         r'$\max=%.0f$' % (np.max(tracker_events["real_ET"]), )
# ))
# ax[2].text(0.62, 0.95, textstr_et, transform=ax[2].transAxes, fontsize=13,
#                          verticalalignment='top', bbox=props)
# plt.savefig(savefolder+"/dist_tracker.png", dpi=600)


#############################################################################################################
############ Plot images for second part visualization
#############################################################################################################

#contained in Utilities/create_from_CGAN.py
