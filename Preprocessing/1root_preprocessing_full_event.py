#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-17 12:46:41
    # Description :
####################################################################################
"""
import os
import sys
sys.path.insert(1, "../Utilities")
import pickle

import ROOT as r
import numpy as np
import pandas as pd
import root_numpy as rn
import matplotlib.pyplot as plt

from collections import Counter

from functionsOnImages import double_images, insert

#############################################################################################################
############ Global variables
#############################################################################################################
file = 'B2Dmunu.root'
data_path = "../../Data/B2Dmunu"
file_path = '{}/{}'.format(data_path, file)
tree_name = 'Bd2Dmu/DecayTree'

tracker_dim = [52, 64]
read_tracker = True
read_calo = True
cleaning = True
convert_to_image = True

mode = "LargeSample"

nr_max = 93510

#############################################################################################################
############ Initilaization
#############################################################################################################
data_save_path = data_path + "/"  + mode
start = 0
if mode == "Debug":
    stop = 5000
elif mode == "Test":
    stop = 10000
elif mode == "LargeSample":
    stop = 60000
elif mode == "Full":
    stop = None
elif mode == "TestingPurpose":
    start = nr_max - 30000
    stop = nr_max
else:
    raise ValueError("Variable 'mode' has to be in {}".format(["Logging", "Debug", "Sample", "Full"]))

if not os.path.exists(data_save_path):
    os.mkdir(data_save_path)


#############################################################################################################
############ Tracker
#############################################################################################################
def extract_tracker_info_from_root(path_to_file, tree_name, save_path=None):

    f = r.TFile(file_path)
    t = f.Get(tree_name)

    branch_to_name = {"x_projections": "K_L0Calo_HCAL_xProjections",
                        "y_projections": "K_L0Calo_HCAL_yProjections",
                        "real_ET": "K_L0Calo_HCAL_realETs",
                        "K_x_projection": "K_L0Calo_HCAL_xProjection",
                        "K_y_projection": "K_L0Calo_HCAL_yProjection",
                        "K_real_ET": "K_L0Calo_HCAL_realET",
                        "Pi1_x_projection": "pi1_L0Calo_HCAL_xProjection",
                        "Pi1_y_projection": "pi1_L0Calo_HCAL_yProjection",
                        "Pi1_real_ET": "pi1_L0Calo_HCAL_realET",
                        "Pi2_x_projection": "pi2_L0Calo_HCAL_xProjection",
                        "Pi2_y_projection": "pi2_L0Calo_HCAL_yProjection",
                        "Pi2_real_ET": "pi2_L0Calo_HCAL_realET",
                        "Mu_x_projection": "mu_L0Calo_HCAL_xProjection",
                        "Mu_y_projection": "mu_L0Calo_HCAL_yProjection",
                        "Mu_real_ET": "mu_L0Calo_HCAL_realET",
                        "n_Particles": "K_n_Particles"
                        }

    events = {name: rn.root2array(filenames=file_path, treename=tree_name, branches=branch, start=start, stop=stop)
                                                    for name, branch in branch_to_name.items()
                }
    df = pd.DataFrame.from_dict(events)

    def get_phi(row):
        return np.arctan2(row["x_projections"], row["y_projections"]) + np.pi
    df["phi"] = df.apply(get_phi, axis=1)

    def get_theta(row):
        z = 13330 # From LHCb HCAL Paper (distance to calorimeter) - The LHCb Hadron Calorimeter, Yu. Guz, Conference Series 160 (2009)
        return np.arcsin(np.sqrt(row["x_projections"]**2+row["y_projections"]**2)/np.sqrt(row["x_projections"]**2+row["y_projections"]**2+z**2))
    df["theta"] = df.apply(get_theta, axis=1)

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(df, f)
    print("Tracker extracted.")

    return df


def convert_tracker_event_to_image(events, tracker_dim, save_path=None, show_id=None):
    """ Tracker measurements are on continuous scale. We need to assign them a cell coordination (i, j) to get
    the pixelated image.
    X_grid has to lie between 0,..., n-1, because n would be out of bounds
    """
    X_grid_coord = events["x_projections"].apply(scale_to_grid_coordinate, args=[0, tracker_dim[1]-1])
    Y_grid_coord = events["y_projections"].apply(scale_to_grid_coordinate, args=[0, tracker_dim[0]-1])

    nr_events = len(X_grid_coord)
    pics_idxs = np.arange(nr_events).tolist()

    pics = np.array([np.zeros(shape=tracker_dim) for _ in pics_idxs])
    for Id in range(nr_events):
        pics[Id, Y_grid_coord[Id], X_grid_coord[Id]] = events.loc[Id, "real_ET"]

        if (Id % 10000) == 0:
            print(Id, "/", nr_events)

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(pics, f)
        print("Saved to {}".format(save_path))

    if show_id is not None:
        plt.imshow(pics[show_id])
        plt.show()

    return pics


def scale_to_grid_coordinate(values, minval, maxval):
    """ Covert continuous measurements to index grid with indices from minval to maxval in both dimensions.
    So for a two dimensional image with x and y measurements between 0-1000 and example measurement of
    (250, 730) which should be scaled to a 10x10 pixel image, the output is (2, 7). So if a particle is at
    position (250, 730) the pixel (2, 7) is lit in a 10x10 image.
    """
    scaled_values = (values - values.min()) / (values.max() - values.min()) # Scale to [0, 1]
    index_values = scaled_values * (maxval - minval) + minval # Scale to [minval, maxval]
    index_values = index_values.astype(dtype=int) # Convert to index
    return index_values




#############################################################################################################
############ Calorimeter
#############################################################################################################
def extract_calo_info_from_root(path_to_file, tree_name, save_path=None):
    f = r.TFile(file_path)
    t = f.Get(tree_name)

    branch_to_name = {"calo_ET_inner": "K_L0Calo_HCAL_CellsET_inner",
                        "calo_ET_outer": "K_L0Calo_HCAL_CellsET_outer"
                        }

    events = {name:
                np.stack(
                    rn.root2array(
                        filenames=file_path, treename=tree_name, branches=branch, start=start, stop=stop), axis=0
                )
                for name, branch in branch_to_name.items()
                }

    for key, value in events.items():
        events[key] = set_to_zero(events[key], maxval=6)
        events[key] = delete_empty_rows(events[key])

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(events, f)
        print("Saved to {}".format(save_path))

    return events


def set_to_zero(events, maxval):
    events[events<maxval] = 0
    return events


def delete_empty_rows(events):
    """ Delete rows if they are empty over ALL inputs.
    """
    sum_images = np.sum(events, axis=0)
    sum_columns = np.sum(sum_images, axis=1)
    remove_rows = np.where(sum_columns==0)[0]
    print(remove_rows)
    events = np.delete(events, remove_rows, axis=1)
    return events


def convert_calo_event_to_image(events, save_path=None, show_id=None):
    """ Insert inner calo image into outer, by first doubling the outer image.
    """
    pics_outer = double_images(events["calo_ET_outer"])
    pics = insert(inner=events["calo_ET_inner"], outer=pics_outer)

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(pics, f)
            print("Saved to {}".format(save_path))

    if show_id is not None:
        plt.imshow(pics[show_id])
        plt.show()

    return pics




#############################################################################################################
############ Clean data
#############################################################################################################

def clean_data(tracker_events, calo_events, save_path_tracker=None, save_path_calo=None, save_path_log=None):
    calo_events_inner = calo_events["calo_ET_inner"]
    calo_events_outer = calo_events["calo_ET_outer"]
    original_number_events = len(tracker_events)

    # calo_events_inner, nr_noise_inner = remove_noise(calo_events_inner)
    # calo_events_outer, nr_noise_outer = remove_noise(calo_events_outer)
    nr_noise_inner = nr_noise_outer = np.array([0])

    is_empty_inner = is_empty(calo_events_inner)
    is_empty_outer = is_empty(calo_events_outer)
    is_empty_calo = np.logical_and(is_empty_inner, is_empty_outer)

    is_clean = ~is_empty_calo

    tracker_events = tracker_events[is_clean]
    calo_events_inner = calo_events_inner[is_clean]
    calo_events_outer = calo_events_outer[is_clean]

    print("Empty: {}\n".format(sum(is_empty_calo)))
    print("\nDeleted: {} / {}".format(sum(~is_clean), original_number_events))

    calo_events["calo_ET_inner"] = calo_events_inner
    calo_events["calo_ET_outer"] = calo_events_outer
    assert len(tracker_events) == len(calo_events["calo_ET_inner"]) == len(calo_events["calo_ET_outer"])

    if save_path_tracker is not None:
        with open(save_path_tracker, "wb") as f:
            pickle.dump(tracker_events, f)
    if save_path_calo is not None:
        with open(save_path_calo, "wb") as f:
            pickle.dump(calo_events, f)
    if save_path_log is not None:
        log_cleaning_process(nr_noise_inner, nr_noise_outer, sum(is_empty_calo),
                         sum(~is_clean), original_number_events, save_path_log)

    return tracker_events, calo_events


def remove_noise(images, noise_value=0):
    nr_noise_pixels = [0]*len(images)
    denoised_images = [0]*len(images)
    for i, image in enumerate(images):
        try:
            pixel_value_counter = Counter(image.ravel()).most_common(2)[1]
            if pixel_value_counter[0] == 0:
                print("Careful. 0 is not most common value in image!")
                pixel_value_counter = Counter(image.ravel()).most_common(2)[0]

            if pixel_value_counter[1] > 30:
                image[image==pixel_value_counter[0]] = 0
                nr_noise_pixels[i] = pixel_value_counter[1]

        except IndexError:
            pass

        denoised_images[i] = image

    denoised_images = np.stack(denoised_images, axis=0)
    return denoised_images, np.array(nr_noise_pixels)


def remove_noise_(images, noise_value=0):
    is_noisy = is_noise(images)
    nr_noise_pixels = [np.sum(image>0) for image in images[is_noisy]]
    images[is_noisy, :, :] = 0
    return images, np.array(nr_noise_pixels)


def is_noise(images):
    is_noisy = [(len(np.unique(image))==2) & (np.sum(image>0)>30) for image in images]
    return is_noisy


def is_empty(images):
    energies = np.sum(np.sum(images, axis=2), axis=1)
    return energies == 0


def log_cleaning_process(noise_inner, noise_outer, nr_empty, nr_clean, nr_events, save_path):
    with open(save_path+"/README.txt", "w") as f:
        f.write(
            "Noise inner: {}\nNoise outer: {}\n".format(sum(noise_inner!=0), sum(noise_outer!=0)) +
            "\nEmpty: {}\n".format(nr_empty) +
            "\nDeleted: {} / {}".format(nr_clean, nr_events)
        )
    fig_calo_distribution, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), facecolor='w', edgecolor='k')
    fig_calo_distribution.subplots_adjust(wspace=0.2, hspace=0.05)
    ax[0].hist(noise_inner, bins=20)
    ax[0].set_xlabel("Numper of noise pixels")
    ax[0].set_title("Inner calorimeter")

    ax[1].hist(noise_outer, bins=20)
    ax[1].set_xlabel("Numper of noise pixels")
    ax[1].set_title("Outer calorimeter")
    plt.savefig(save_path+"/NoiseDistribution.png")





if __name__ == "__main__":
    if (read_tracker):
        tracker_events = extract_tracker_info_from_root(path_to_file=file_path, tree_name=tree_name,
                                                               save_path="{}/tracker_events.pickle".format(data_save_path))

    if (read_calo):
        calo_events = extract_calo_info_from_root(path_to_file=file_path, tree_name=tree_name,
                                                                save_path="{}/calo_events.pickle".format(data_save_path))
        print("\nExtracted.\n")

    if (cleaning):
        tracker_events, calo_events = clean_data(tracker_events, calo_events,
                                                 save_path_tracker="{}/tracker_events.pickle".format(data_save_path),
                                                 save_path_calo="{}/calo_events.pickle".format(data_save_path),
                                                 save_path_log=data_save_path)
        print("\nCleaned and saved\n")


    if (convert_to_image):
        pics_tracker_piplus = convert_tracker_event_to_image(events=tracker_events, tracker_dim=tracker_dim,
                                                             save_path="{}/tracker_images.pickle".format(data_save_path))
        calo_pics_piplus = convert_calo_event_to_image(calo_events,
                                                          save_path="{}/calo_images.pickle".format(data_save_path))

        print("\nImages created.\n")