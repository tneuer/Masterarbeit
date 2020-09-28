#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-19 16:57:42
    # Description :
####################################################################################
"""
import os
import sys
sys.path.insert(0, "../Utilities/")
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from initialization import load_data, load_processed_data
from functionsOnImages import crop_images, get_mean_image
from functionsOnImages import get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import savefigs



def get_phi(x, y):
    return np.arctan2(x, y)


def check_angles(angle1, angle2, name):
    """ Plot angle computed from method 1 against angle from method 2 (position vs. momentum)
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].scatter(angle1, angle2)
    ax[0].plot(ax[0].get_xlim(), ax[0].get_ylim(), ls="--", c=".3")
    ax[0].set_xlabel("{} from projection.".format(name))
    ax[0].set_ylabel("{} from momentum.".format(name))
    ax[1].hist(angle1)
    ax[1].set_xlabel("{} from projection.".format(name))

    return fig, ax


def plot_in_range(data, images, column, range_of_values, ax=None):
    """ Plot data in certain range of angles
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    else:
        fig = plt.gcf()
    is_in_range = (data[column] >= range_of_values[0]) & (data[column] <= range_of_values[1])
    images_in_range = images[is_in_range]
    mean_image = get_mean_image(images_in_range)
    ax.imshow(mean_image)
    title_range = [np.round(val, 2) for val in range_of_values]
    ax.set_title("{} in [{}, {}]".format(column, title_range[0], title_range[1]))

    return fig, ax


def build_symmetry_plot(tracker, calo, plot_cols, theta_range, phi_range):
    fig, axs = plt.subplots(len(theta_range)-1, len(plot_cols)+1, figsize=(16, 9))
    for i in range(len(theta_range)-1):
        is_in_theta_range = (tracker["theta"] >= theta_range[i]) & (tracker["theta"] <= theta_range[i+1])
        for j in range(len(plot_cols)):
            column = plot_cols[j]
            ax = axs[i, j]
            bin_lowest = min(tracker[column])
            bin_highest = max(tracker[column])
            bins = np.linspace(bin_lowest, bin_highest, 20)
            for k in range(len(phi_range)-1):
                is_in_phi_range = (tracker["phi"] >= phi_range[k]) & (tracker["phi"] <= phi_range[k+1])
                is_in_range = np.logical_and(is_in_theta_range, is_in_phi_range)

                ax.hist(
                    tracker[is_in_range][column], bins=bins,
                    label="[{}, {}]".format(np.round(phi_range[k], 2), np.round(phi_range[k+1], 2)),
                    histtype="step", stacked=False
                )

            if i == 0:
                ax.set_title(column)
            if j == 0:
                lower = np.round(theta_range[i], 2)
                upper = np.round(theta_range[i+1], 2)
                ax.set_ylabel("Theta in [{}, {}]".format(lower, upper))

        axs[i, j+1].imshow(get_mean_image(calo[is_in_theta_range]))
        axs[i, j+1].set_title("{} images".format(sum(is_in_theta_range)))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')

    return fig, axs


if __name__ == '__main__':
    ############################################################################################################
    # Original images
    ############################################################################################################
    path_loading = "../../Data/Piplus/LargeSample"
    data, scaler = load_processed_data(data_path=path_loading, mode="train")
    tracker = data["Tracker"]
    calo = data["Calo"]*scaler["Calo"]

    exclude = ["phi", "theta", "region"]
    columns = tracker.drop(exclude, axis=1).columns.values
    tracker_inverse = scaler["Tracker"].inverse_transform(tracker[columns])
    tracker_inverse = pd.DataFrame(data=tracker_inverse, columns=columns, index=tracker.index)
    tracker = pd.concat([tracker_inverse, tracker[exclude]], axis=1)


    fig_scan, axs = plt.subplots(2, 4, figsize=(16, 9))
    phi_range = np.linspace(0, 2*np.pi, 5)
    for i in range(4):
        plot_in_range(tracker, calo, "phi", [phi_range[i], phi_range[i+1]], ax=axs[0, i])

    theta_range = np.linspace(min(tracker["theta"]), max(tracker["theta"]), 5)
    for i in range(4):
        plot_in_range(tracker, calo, "theta", [theta_range[i], theta_range[i+1]], ax=axs[1, i])
    fig_scan.suptitle("Calculated from position")
    plt.show()
    raise

    ############################################################################################################
    # Created images
    ############################################################################################################
    # path_loading = "../../Data/Piplus/Sample/Trained"
    # with open(path_loading+"/CWGANGP12_out_1.pickle", "rb") as f:
    #     calo = pickle.load(f).reshape([-1, 56, 64])
    #     calo = crop_images(calo, top=2, bottom=2)


    ############################################################################################################
    # Run images
    ############################################################################################################
    figs = []

    phi1 = get_phi(x=tracker["x_projections"], y=tracker["y_projections"])
    phi2 = get_phi(x=tracker["momentum_px"], y=tracker["momentum_py"])
    fig_phi, _ = check_angles(phi1, phi2, "Phi")
    figs.append(fig_phi)

    z = 13330 + 1220 / 2
    theta1 = np.arcsin(np.sqrt(tracker["x_projections"]**2+tracker["y_projections"]**2)/np.sqrt(tracker["x_projections"]**2+tracker["y_projections"]**2+z**2))
    theta2 = np.arcsin(tracker["momentum_pt"]/tracker["momentum_p"])
    fig_theta, _ = check_angles(theta1, theta2, "Theta")
    figs.append(fig_theta)

    tracker["CaloEnergy"] = get_energies(calo)
    tracker["CaloMaxEnergy"] = get_max_energy(calo)
    tracker["Cells"] = get_number_of_activated_cells(calo)

    tracker["theta"] = theta1
    tracker["phi"] = phi1 + np.pi
    fig_scan, axs = plt.subplots(2, 4, figsize=(16, 9))
    phi_range = np.linspace(0, 2*np.pi, 5)
    for i in range(4):
        plot_in_range(tracker, calo, "phi", [phi_range[i], phi_range[i+1]], ax=axs[0, i])

    theta_range = np.linspace(min(tracker["theta"]), max(tracker["theta"]), 5)
    for i in range(4):
        plot_in_range(tracker, calo, "theta", [theta_range[i], theta_range[i+1]], ax=axs[1, i])
    fig_scan.suptitle("Calculated from position")
    figs.append(fig_scan)

    plot_cols = ["momentum_p", "momentum_pt", "CaloEnergy", "CaloMaxEnergy", "Cells", "phi"]
    fig_sym, axs = build_symmetry_plot(tracker, calo, plot_cols, theta_range, phi_range)
    fig_sym.suptitle("Calculated from position")
    figs.append(fig_sym)

    tracker["theta"] = theta2
    tracker["phi"] = phi2 + np.pi
    fig_scan, axs = plt.subplots(2, 4, figsize=(16, 9))
    phi_range = np.linspace(0, 2*np.pi, 5)
    for i in range(4):
        plot_in_range(tracker, calo, "phi", [phi_range[i], phi_range[i+1]], ax=axs[0, i])

    theta_range = np.linspace(min(tracker["theta"]), max(tracker["theta"]), 5)
    for i in range(4):
        plot_in_range(tracker, calo, "theta", [theta_range[i], theta_range[i+1]], ax=axs[1, i])
    fig_scan.suptitle("Calculated from momentum")
    figs.append(fig_scan)


    plot_cols = ["momentum_p", "momentum_pt", "CaloEnergy", "CaloMaxEnergy", "Cells", "phi"]
    fig_sym, axs = build_symmetry_plot(tracker, calo, plot_cols, theta_range, phi_range)
    fig_sym.suptitle("Calculated from momentum")
    figs.append(fig_sym)

    savefigs(figs, path_loading+"/Angles.pdf")
