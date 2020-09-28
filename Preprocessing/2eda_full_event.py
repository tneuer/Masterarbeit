#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-18 23:25:24
    # Description :
####################################################################################
"""
import sys
sys.path.insert(1, '../Utilities/')
import functionsOnImages

import numpy as np
import pandas as pd
import initialization as init
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from functionsOnImages import padding_zeros, get_layout, build_image, build_images, savefigs
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, get_energy_resolution
from functionsOnImages import build_histogram_HTOS, crop_images

#############################################################################################################
############ Preprocess and load
#############################################################################################################
data_load_path = "../../Data/B2Dmunu/LargeSample"
figure_save_path = data_load_path
data = init.load_data(data_load_path, mode="all")

tracker_events = data["tracker_events"]
tracker_images = data["tracker_images"]
calo_events = data["calo_events"]
calo_images = data["calo_images"]

figs = []

#############################################################################################################
############ Outlier
#############################################################################################################

flattened_energies = [item for sublist in tracker_events["real_ET"].ravel() for item in sublist]

# et_cutoff = np.mean(flattened_energies) + 4*np.std(flattened_energies)
et_cutoff = 7282.0142822265625
def contains_outlier(row):
    return np.any(row > et_cutoff)

is_no_outlier = ~tracker_events["real_ET"].apply(contains_outlier)
tracker_events = tracker_events[is_no_outlier]
tracker_images = tracker_images[is_no_outlier]
calo_events["calo_ET_inner"] = calo_events["calo_ET_inner"][is_no_outlier]
calo_events["calo_ET_outer"] = calo_events["calo_ET_outer"][is_no_outlier]
calo_images = calo_images[is_no_outlier]

flatten_tracker_df = pd.DataFrame({
                           "x_projections": [item for sublist in tracker_events["x_projections"].ravel() for item in sublist],
                           "y_projections": [item for sublist in tracker_events["y_projections"].ravel() for item in sublist],
                           "phi": [item for sublist in tracker_events["phi"].ravel() for item in sublist],
                           "theta": [item for sublist in tracker_events["theta"].ravel() for item in sublist],
                           "real_ET": [item for sublist in tracker_events["real_ET"].ravel() for item in sublist],
                           })

#############################################################################################################
############ Tracker
#############################################################################################################
print("Tracker...")
fig_tracker_distribution, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), facecolor='w', edgecolor='k')
fig_tracker_distribution.subplots_adjust(wspace=0.2, hspace=0.4)
ax = np.ravel(ax)

for i, column in enumerate(flatten_tracker_df):
    ax[i].hist(flatten_tracker_df[column], bins=30)
    ax[i].set_xlabel("")
    ax[i].set_title(column)
figs.append(fig_tracker_distribution)

# Correlation
sns_pp = sns.pairplot(flatten_tracker_df, plot_kws={'alpha': 0.1})
sns_pp.savefig(figure_save_path+"/pairplot.png")

#############################################################################################################
############ Energy
#############################################################################################################
print("Calorimeter...")
### Check calo statistics
fig_caloStat, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 9), facecolor='w', edgecolor='k')
axes = np.ravel(axes)
fig_caloStat.subplots_adjust(wspace=0.2, hspace=0.3)
colnames = ["X CoM", "Y CoM", "Cells", "Enery", "MaxEnergy", "StdEnergy", "Resolution", "dN/dE", "(dN/dE) / (dN/dE)"]
calo_scaler = 1
image_shape = calo_images[0].shape
tracker_real_ET = tracker_events["real_ET"].apply(sum)
use_functions = {
                get_center_of_mass_x: {"image_shape": image_shape}, get_center_of_mass_y: {"image_shape": image_shape},
                get_number_of_activated_cells: {"threshold": 5/calo_scaler},
                get_energies: {"energy_scaler": calo_scaler}, get_max_energy: {"energy_scaler": calo_scaler},
                get_std_energy: {"energy_scaler": calo_scaler},
                get_energy_resolution: {"real_ET": tracker_real_ET, "energy_scaler": calo_scaler}
                }


build_histogram_HTOS(true=calo_images, fake=None, energy_scaler=calo_scaler,
                     threshold=3600, real_ET=tracker_real_ET, ax1=axes[7], ax2=axes[8])

for func_idx, (func, params) in enumerate(use_functions.items()):
    build_histogram(true=calo_images, fake=None, function=func, name=colnames[func_idx], epoch="",
                    folder=None, ax=axes[func_idx], **params)

figs.append(fig_caloStat)



fig_calo_distribution, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 8), facecolor='w', edgecolor='k')
fig_calo_distribution.subplots_adjust(wspace=0.2, hspace=0.05)

## Distribution total energy
calo_energies = functionsOnImages.get_energies(calo_images)
ax[0].hist(calo_energies, bins=20)
ax[0].set_xlabel("Calorimeter energy")
ax[0].set_title("Calorimeter energy density")

## Distribution resolution
energy_resolution = (tracker_real_ET.values - calo_energies) / tracker_real_ET.values
ax[1].hist(energy_resolution, bins=20)
ax[1].set_xlabel("(true - reco) / true")
ax[1].set_title("Calorimeter resolution")


## ECDF
unique_calo_energies = np.sort(np.unique(calo_energies))
cumulative_sum = [0]*len(unique_calo_energies)

for i, energy in enumerate(unique_calo_energies):
    cumulative_sum[i] = np.sum(calo_energies<energy)/len(calo_energies)*100

ax[2].plot(unique_calo_energies, cumulative_sum)
ax[2].set_xlabel("Calorimeter energy")
ax[2].set_ylabel("Percentage")
ax[2].set_title("Calorimeter energy distribution")
figs.append(fig_calo_distribution)


## Check mean images
fig_mean_images_calo, _ = functionsOnImages.build_mean_images([
                                calo_events["calo_ET_inner"],
                                calo_events["calo_ET_outer"],
                                calo_images
                                ], ["Inner", "Outer", "Reconstructed"], save_path=None)
figs.append(fig_mean_images_calo)


#############################################################################################################
############ Tracker vs Calorimeter
#############################################################################################################
print("Tracker vs. Calorimeter...")

## Energy
tracker_energies = tracker_events["real_ET"].apply(sum).values
calo_energies = functionsOnImages.get_energies(calo_images)

fig_E_comparison, ax = plt.subplots()
ax.scatter(tracker_energies, calo_energies)
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
ax.set_title("Tracker vs. Calorimeter energies")
ax.set_xlabel("Tracker")
ax.set_ylabel("Calorimeter")
figs.append(fig_E_comparison)

## Position in tracker and calorimeter for 10 random examples to see if the rescaling in the tracker worked as expected
fig_xy_comparison, ax = functionsOnImages.build_tracker_calo_images(data, 10, calo_components=True, seed=42, save_path=None)
figs.append(fig_xy_comparison)


#############################################################################################################
############ Scan input space of tracker -> Look at calo images
#############################################################################################################
# def find_closest_element(data, number, column):
#     return data.iloc[(data[column]-number).abs().argsort()[:1]]

# quantiles = tracker_events.quantile(q=np.linspace(0, 1, 100))
# means = np.mean(tracker_events, axis=0)
# attributes = tracker_events.columns.values
# for i, attribute in enumerate(attributes):
#     print("Scanning: {}".format(attribute))
#     image_array = []
#     scanned_conditions = means.copy()
#     for j, quantile in enumerate(quantiles[attribute]):
#         if j % 10 == 0:
#             image_array.append([])
#         image_idx = find_closest_element(tracker_events, quantile, attribute).index.tolist()[0]
#         image_array[-1].append(calo_images[image_idx])

#     fig, ax = functionsOnImages.build_images(image_array)
#     plt.savefig(figure_save_path+"/Scan_{}.png".format(attribute))

# plt.show()

functionsOnImages.savefigs(figs, figure_save_path+"/Summary.pdf")
