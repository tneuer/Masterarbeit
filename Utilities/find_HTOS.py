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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from TrainedIm2Im import TrainedIm2Im
from functionsOnImages import padding_zeros, separate_images, halve_images, halve_image


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


def plot_tracker_image_with_signal_frame(images, events, pixel_width, idx, padding=None, show=True, ax=None):
    assert isinstance(pixel_width, int), "pixel_width must be integer. Given: {}.".format(type(pixel_width))
    assert np.min(images[0].shape)*0.1 > pixel_width, (
        "pixel_width must not be more than 5% of image size. Max: {}. Given: {}.".format(np.min(images[0].shape)*0.1, pixel_width)
    )

    tracker_image = images[idx]
    im_shape = tracker_image.shape
    events = events.reset_index()
    def truncate_value_x(value):
        return np.min([np.max([value, 0]), im_shape[1]])
    def truncate_value_y(value):
        return np.min([np.max([value, 0]), im_shape[0]])
    if padding is None:
        padding = {"top": 0, "bottom": 0, "left": 0, "right": 0}

    particles = ["Pi1", "Pi2", "K"]
    centers = [
        [events.iloc[idx]["Pi1_x_projection_pixel"], events.iloc[idx]["Pi1_y_projection_pixel"]],
        [events.iloc[idx]["Pi2_x_projection_pixel"], events.iloc[idx]["Pi2_y_projection_pixel"]],
        [events.iloc[idx]["K_x_projection_pixel"], events.iloc[idx]["K_y_projection_pixel"]]
    ]
    corners = [
        [truncate_value_x(centers[0][0]-pixel_width-0.5+padding["left"]), truncate_value_y(centers[0][1]-pixel_width-0.5+padding["top"])],
        [truncate_value_x(centers[1][0]-pixel_width-0.5+padding["left"]), truncate_value_y(centers[1][1]-pixel_width-0.5+padding["top"])],
        [truncate_value_x(centers[2][0]-pixel_width-0.5+padding["left"]), truncate_value_y(centers[2][1]-pixel_width-0.5+padding["top"])]
    ]


    if show or (ax is not None):

        add_colorbar = False
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            add_colorbar = True

        im = ax.imshow(tracker_image)
        image_title = ""
        for i, (left_x, top_y) in enumerate(corners):
            frame = patches.Rectangle((left_x, top_y), pixel_width*2+1, pixel_width*2+1,
                                      linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(frame)
            image_title += "{}: {}\n".format(particles[i], events.loc[idx, "{}_real_ET".format(particles[i])])

        ax.set_title(image_title)

        if add_colorbar:
            fig.colorbar(im, ax=ax, shrink=0.6)

        if show:
            plt.show()
        return corners, ax
    else:
        return corners


def plot_calo_image_with_triggered_frame(images, idx, threshold, show=True, ax=None):
    assert images.shape[1] == images.shape[2] == 64, (
        "Only shape (64, 64) supported. If you want to change, take care of the shift for the inner image!!"
    )
    image = images[idx].reshape([1, images.shape[1], images.shape[2]])
    im_shape = image.shape[1:]
    def truncate_value_x(value):
        return np.min([np.max([value, 0]), im_shape[1]])
    def truncate_value_y(value):
        return np.min([np.max([value, 0]), im_shape[0]])

    inner, outer = separate_images(image)
    inner = inner.reshape([inner.shape[1], inner.shape[2]])
    outer = outer.reshape([outer.shape[1], outer.shape[2]])
    inner_triggered = []
    outer_triggered = []
    title = ""
    max_cells_sum = -np.inf

    #### Check trigger in inner image
    for i in range(inner.shape[0]-1):
        for j in range(inner.shape[1]-1):
            cells_sum = inner[i, j] + inner[i+1, j] + inner[i, j+1] + inner[i+1, j+1]
            if cells_sum > threshold:
                inner_triggered.append([
                    truncate_value_x(j+16-0.5), truncate_value_y(i+18-0.5)
                ])
            if cells_sum > max_cells_sum:
                max_cells_sum = cells_sum

    #### Check trigger in outer image
    cells_sums = []
    for i in range(outer.shape[0]-1):
        for j in range(outer.shape[1]-1):
            cells_sum = outer[i, j] + outer[i+1, j] + outer[i, j+1] + outer[i+1, j+1]
            cells_sums.append(cells_sum)
            if cells_sum > threshold:
                outer_triggered.append([
                    truncate_value_x(j*2-0.5), truncate_value_y(i*2-0.5)
                ])
            if cells_sum > max_cells_sum:
                max_cells_sum = cells_sum

    if show or (ax is not None):

        add_colorbar = False
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            add_colorbar = True

        if len(inner_triggered) > 0:
            title += "Inner triggered"
        if len(outer_triggered) > 0:
            title += "\nOuter triggered"
        elif title == "":
            title = "Not triggered"
        title += "\nMax sum: {}.".format(max_cells_sum)
        im = ax.imshow(image.reshape([images.shape[1], images.shape[2]]))
        for i, (left_x, top_y) in enumerate(inner_triggered):
            frame = patches.Rectangle((left_x, top_y), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(frame)

        for i, (left_x, top_y) in enumerate(outer_triggered):
            frame = patches.Rectangle((left_x, top_y), 4, 4, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(frame)

        ax.set_title(title)

        if add_colorbar:
            fig.colorbar(im, ax=ax, shrink=0.6)
        if show:
            plt.show()
        return inner_triggered, outer_triggered, ax
    else:
        return inner_triggered, outer_triggered


def plot_calo_image_with_triggered_and_signal_frame(tracker_images, tracker_events, pixel_width, idx,
                                                calo_images, threshold, padding=None, show="all"):
    assert tracker_images.shape == calo_images.shape[:3], (
        "Same shape for tracker and calorimeter neeeded. Tracker: {}. Calorimeter: {}.".format(tracker_images.shape, calo_images.shape)
    )

    if show:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 12))

    signal_corners, _ = plot_tracker_image_with_signal_frame(
        images=tracker_images, events=tracker_events, pixel_width=pixel_width, idx=idx, padding=padding, show=False, ax=axs[0]
    )
    inner_corners, outer_corners, _ = plot_calo_image_with_triggered_frame(
        images=calo_images, idx=idx, threshold=threshold, show=False, ax=axs[1]
    )

    image = calo_images[idx].reshape([calo_images.shape[1], calo_images.shape[2]])

    is_TOS = False
    is_TIS = False
    for trigger_corner in inner_corners:
        breaks = False
        left_top_trigger = trigger_corner
        right_bottom_trigger = [left_top_trigger[0]+2, left_top_trigger[1]+2]
        for signal_corner in signal_corners:
            left_top_signal = signal_corner
            right_bottom_signal = [left_top_signal[0]+pixel_width*2+1, left_top_signal[1]+pixel_width*2+1]
            if is_rectangle_overlap(left_top1=left_top_trigger, right_bottom1=right_bottom_trigger,
                                    left_top2=left_top_signal, right_bottom2=right_bottom_signal):
                is_TOS = True
                breaks = True
                break
        if not breaks:
            is_TIS = True

    for trigger_corner in outer_corners:
        breaks = False
        left_top_trigger = trigger_corner
        right_bottom_trigger = [left_top_trigger[0]+4, left_top_trigger[1]+4]
        for signal_corner in signal_corners:
            left_top_signal = signal_corner
            right_bottom_signal = [left_top_signal[0]+pixel_width*2+1, left_top_signal[1]+pixel_width*2+1]
            if is_rectangle_overlap(left_top1=left_top_trigger, right_bottom1=right_bottom_trigger,
                                    left_top2=left_top_signal, right_bottom2=right_bottom_signal):
                is_TOS = True
                breaks = True
                break
        if not breaks:
            is_TIS = True

    if show:
        im = axs[2].imshow(image)

        for i, (left_x, top_y) in enumerate(signal_corners):
            frame = patches.Rectangle((left_x, top_y), pixel_width*2+1, pixel_width*2+1, linewidth=1, edgecolor='g', facecolor='none')
            axs[2].add_patch(frame)

        for i, (left_x, top_y) in enumerate(inner_corners):
            frame = patches.Rectangle((left_x, top_y), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
            axs[2].add_patch(frame)

        for i, (left_x, top_y) in enumerate(outer_corners):
            frame = patches.Rectangle((left_x, top_y), 4, 4, linewidth=1, edgecolor='r', facecolor='none')
            axs[2].add_patch(frame)

        axs[2].set_title(
            "TOS: {} - TIS: {}\nTruth TOS: {} - TIS: {}".format(
                is_TOS, is_TIS, tracker_events.iloc[idx]["is_TOS"], tracker_events.iloc[idx]["is_TIS"]
            )
        )

        plt.show()
    else:
        return signal_corners, inner_corners, outer_corners

def is_rectangle_overlap(left_top1, right_bottom1, left_top2, right_bottom2):
    if (left_top1[0] >= right_bottom2[0]) or (left_top2[0] >= right_bottom1[0]):
        return False
    if (left_top1[1] >= right_bottom2[1]) or (left_top2[1] >= right_bottom1[1]):
        return False
    return True


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

    padding = {"top": 6, "bottom": 6, "left": 0, "right": 0}
    mc_data_images = padding_zeros(mc_data[:], **padding).reshape(-1, 64, 64, 1)
    gan_data_m = np.clip(padding_zeros(gan_data[:], top=4, bottom=4), a_min=0, a_max=calo_scaler) / calo_scaler
    tracker_images_m = padding_zeros(tracker_images[:], **padding).reshape([-1, image_shape[0], image_shape[1]])
    tracker_events_m = tracker_events.iloc[:, :]

    # plot_tracker_image_with_signal_frame(images=tracker_images_m, events=tracker_events_m, pixel_width=3, idx=4, padding=padding, show=True)
    # plot_calo_image_with_triggered_frame(images=mc_data_images, idx=16, threshold=3600, show=True)
    for idx in range(100, 1000):
        if tracker_events_m.iloc[idx]["is_TOS"]:
            plot_calo_image_with_triggered_and_signal_frame(
                tracker_images=tracker_images_m, tracker_events=tracker_events_m, pixel_width=3, idx=idx,
                calo_images=mc_data_images, threshold=3200, padding=padding, show=True
            )
    raise


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

