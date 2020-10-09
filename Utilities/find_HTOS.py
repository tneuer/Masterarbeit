#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-10-08 14:53:18
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
import matplotlib.patches as patches

from TrainedIm2Im import TrainedIm2Im
from functionsOnImages import padding_zeros, get_layout, build_image, build_images, savefigs
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, get_energy_resolution, get_center_of_mass_r
from functionsOnImages import build_histogram_HTOS, crop_images, clip_outer


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


def plot_tracker_image_with_signal_frame(tracker_images, tracker_events, pixel_width, idx):
    assert isinstance(pixel_width, int), "pixel_width must be integer. Given: {}.".format(type(pixel_width))
    assert np.min(tracker_images[0].shape)*0.1 > pixel_width, (
        "pixel_width must not be more than 5% of image size. Max: {}. Given: {}.".format(np.min(tracker_images[0].shape)*0.1, pixel_width)
    )

    particles = ["Pi1", "Pi2", "K"]
    tracker_events = tracker_events.reset_index()
    frame_centers = [
            [tracker_events.loc[idx, "Pi1_x_projection_pixel"], tracker_events.loc[idx, "Pi1_y_projection_pixel"]],
            [tracker_events.loc[idx, "Pi2_x_projection_pixel"], tracker_events.loc[idx, "Pi2_y_projection_pixel"]],
            [tracker_events.loc[idx, "K_x_projection_pixel"], tracker_events.loc[idx, "K_y_projection_pixel"]]
    ]
    tracker_image = tracker_images[idx]

    im_shape = tracker_image.shape
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    im = ax.imshow(tracker_image)
    # ax.axis("off")
    image_title = ""

    for i, (center_x, center_y) in enumerate(frame_centers):
        left_x = np.min([np.max([center_x-pixel_width-0.5, 0]), im_shape[1]])
        top_y = np.min([np.max([center_y-pixel_width-0.5, 0]), im_shape[0]])
        frame = patches.Rectangle((left_x, top_y), pixel_width*2+1, pixel_width*2+1,
                                  linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(frame)
        image_title += "{}: {}\n".format(particles[i], tracker_events.loc[idx, "{}_real_ET".format(particles[i])])
    ax.set_title(image_title)
    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.show()
    raise
    return ax


if __name__ == "__main__":

    #####################################################################################################
    # Model loading
    #####################################################################################################
    source_dir = "../../Results/B2Dmunu"
    model_path = [f.path for f in os.scandir(source_dir) if os.path.isdir(f.path)][0]
    nr_test_hist = 100
    batch_size = 100

    #####################################################################################################
    # Data loading
    #####################################################################################################

    path_loading = "../../Data/B2Dmunu/Debug"
    image_shape = [64, 64, 1]

    with open("../../Data/B2Dmunu/TestingPurpose/calo_images.pickle", "rb") as f:
        mc_data = pickle.load(f)
    with open("../../Data/B2Dmunu/TestingPurpose/Trained/PiplusLowerP_CWGANGP8_out_1.pickle", "rb") as f:
        gan_data = pickle.load(f)
    with open("../../Data/B2Dmunu/TestingPurpose/tracker_images.pickle", "rb") as f:
        tracker_images = pickle.load(f)
    with open("../../Data/B2Dmunu/TestingPurpose/tracker_events.pickle", "rb") as f:
        tracker_events = pickle.load(f)
        tracker_real_ET = tracker_events["real_ET"].apply(sum).to_numpy()
    with open("../../Data/Piplus/LargeSample/ProcessedScaler.pickle", "rb") as f:
        scaler = pickle.load(f)
        calo_scaler = scaler["Calo"]

    x_coordinates = ["Pi1_x_projection", "Pi2_x_projection", "K_x_projection"]
    y_coordinates = ["Pi1_y_projection", "Pi2_y_projection", "K_y_projection"]

    for x_coordinate in x_coordinates:
        globalmin_x = np.min(tracker_events["x_projections"].apply(np.min))
        globalmax_x = np.max(tracker_events["x_projections"].apply(np.max))
        tracker_events[x_coordinate+"_pixel"] = scale_to_grid_coordinate(
            values=tracker_events[x_coordinate], minval=0, maxval=tracker_images.shape[2]-1, globalmin=globalmin_x, globalmax=globalmax_x
        )
    for y_coordinate in y_coordinates:
        globalmin_y = np.min(tracker_events["y_projections"].apply(np.min))
        globalmax_y = np.max(tracker_events["y_projections"].apply(np.max))
        tracker_events[y_coordinate+"_pixel"] = scale_to_grid_coordinate(
            values=tracker_events[y_coordinate], minval=0, maxval=tracker_images.shape[1]-1, globalmin=globalmin_y, globalmax=globalmax_y
        )

    plot_tracker_image_with_signal_frame(
        tracker_images=tracker_images,
        tracker_events=tracker_events,
        pixel_width=3,
        idx = 42
    )
    raise


    mc_data_images = padding_zeros(mc_data[-nr_test_hist:], top=6, bottom=6).reshape(-1, 64, 64, 1)
    gan_data_m = np.clip(padding_zeros(gan_data[-nr_test_hist:], top=4, bottom=4), a_min=0, a_max=calo_scaler) / calo_scaler
    tracker_images_m = padding_zeros(tracker_images[-nr_test_hist:], top=6, bottom=6).reshape([-1, image_shape[0], image_shape[1]])
    tracker_events_m = tracker_events.loc[-nr_test_hist:, :]

    #####################################################################################################
    # Image generation
    #####################################################################################################

    meta_path = model_path + "/TFGraphs/"
    config_path = model_path + "/config.json"
    Generator = TrainedIm2Im(path_to_meta=meta_path, path_to_config=config_path)
    im2im_data = Generator.generate_batches(inputs=gan_data_m, batch_size=batch_size) * calo_scaler

    print(tracker_images.shape, mc_data.shape, gan_data.shape, im2im_data.shape)
    print(np.max(tracker_images), np.max(mc_data), np.max(gan_data), np.max(im2im_data))
    print(np.mean(tracker_images), np.mean(mc_data), np.mean(gan_data), np.mean(im2im_data))

