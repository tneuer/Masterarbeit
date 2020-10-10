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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from TrainedIm2Im import TrainedIm2Im
from functionsOnImages import padding_zeros, separate_images, double_image


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


class Square():
    def __init__(self, center, width, bounds_x=None, bounds_y=None):
        self.center = list(center)
        self.width = width
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y

        self.top_left = self.compute_top_left()
        self.bottom_right = self.compute_bottom_right()

        if bounds_x is not None or bounds_y is not None:
            self.adjust_bounds()

    def compute_top_left(self):
        return [self.center[0] - self.width/2, self.center[1] - self.width/2]

    def compute_bottom_right(self):
        return [self.center[0] + self.width/2, self.center[1] + self.width/2]

    def adjust_bounds(self):
        self.top_left[1] = self.truncate_value(self.top_left[1], minval=self.bounds_x[0], maxval=self.bounds_x[1])
        self.top_left[0] = self.truncate_value(self.top_left[0], minval=self.bounds_y[0], maxval=self.bounds_y[1])
        self.bottom_right[1] = self.truncate_value(self.bottom_right[1], minval=self.bounds_x[0], maxval=self.bounds_x[1])
        self.bottom_right[0] = self.truncate_value(self.bottom_right[0], minval=self.bounds_y[0], maxval=self.bounds_y[1])

    def truncate_value(self, value, minval, maxval):
        return np.min([np.max([value, minval]), maxval])

    def shift(self, x=0, y=0):
        self._shift_x(value=x)
        self._shift_y(value=y)
        self.top_left = self.compute_top_left()
        self.bottom_right = self.compute_bottom_right()

    def _shift_x(self, value):
        self.center[0] = self.center[0] + value

    def _shift_y(self, value):
        self.center[1] = self.center[1] + value

    def overlaps(self, other):
        if (self.top_left[1] >= other.bottom_right[1]) or (other.top_left[1] >= self.bottom_right[1]):
            return False
        if (self.top_left[0] >= other.bottom_right[0]) or (other.top_left[0] >= self.bottom_right[0]):
            return False
        return True

    def get_corner(self):
        return self.top_left

    def get_center(self):
        return self.center



def get_squares_from_tracker(events, width, max_x, max_y):
    assert isinstance(events, pd.DataFrame), "Events is {}, but pd.DataFrame neeeded.".format(type(events))

    coordinate_columns = [
        "Pi1_x_projection_pixel", "Pi1_y_projection_pixel", "Pi2_x_projection_pixel", "Pi2_y_projection_pixel",
        "K_x_projection_pixel", "K_y_projection_pixel"
    ]

    rectangles = []
    for idx, (iPi1X, iPi1Y, iPi2X, iPi2Y, iKX, iKY) in events[coordinate_columns].iterrows():
        Pi1_Rectangle = Square(center=(iPi1X, iPi1Y), width=width, bounds_x=(0, max_x), bounds_y=(0, max_y))
        Pi2_Rectangle = Square(center=(iPi2X, iPi2Y), width=width, bounds_x=(0, max_x), bounds_y=(0, max_y))
        K_Rectangle = Square(center=(iKX, iKY), width=width, bounds_x=(0, max_x), bounds_y=(0, max_y))
        rectangles.append([Pi1_Rectangle, Pi2_Rectangle, K_Rectangle])

    return rectangles


def plot_tracker_image_with_signal_frame(images, events, fr_width, idx, padding=None, show=True, ax=None):
    assert isinstance(fr_width, int), "fr_width must be integer. Given: {}.".format(type(fr_width))
    assert np.min(images[0].shape)*0.2 > fr_width, (
        "fr_width must not be more than 20% of image size. Max: {}. Given: {}.".format(np.min(images[0].shape)*0.2, fr_width)
    )
    if padding is None:
        padding = {"top": 0, "bottom": 0, "left": 0, "right": 0}

    tracker_image = images[idx]
    im_shape = tracker_image.shape
    signal_rects = get_squares_from_tracker(
        events=tracker_events.iloc[[idx]], width=fr_width, max_x=im_shape[1], max_y=im_shape[0]
    )[0]
    for rect in signal_rects:
        rect.shift(x=padding["left"], y=padding["top"])

    particles = ["Pi1", "Pi2", "K"]
    if show or (ax is not None):
        add_colorbar = False
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            add_colorbar = True

        im = ax.imshow(tracker_image)
        image_title = ""
        for i, rect in enumerate(signal_rects):
            frame = patches.Rectangle(rect.get_corner(), fr_width, fr_width, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(frame)
            image_title += "{}: {}\n".format(particles[i], events.loc[idx, "{}_real_ET".format(particles[i])])
        ax.set_title(image_title)

        if add_colorbar:
            fig.colorbar(im, ax=ax, shrink=0.6)
        if show:
            plt.show()
        return signal_rects, ax
    else:
        return signal_rects


def get_triggered_squares(image, width, threshold, shift_x=0, shift_y=0, stride=1):
    triggered_squares = []
    max_cells_sum = -np.inf
    for i in range(0, image.shape[0]-width+1, stride):
        for j in range(0, image.shape[1]-width+1, stride):
            if width == 2:
                cells_sum = image[i, j] + image[i+1, j] + image[i, j+1] + image[i+1, j+1]
            else:
                cells_sum = np.sum(image[i:(i+width), j:(j+width)])
            if cells_sum > threshold:
                rect = Square(center=[j+width/2-0.5, i+width/2-0.5], width=width, bounds_x=(0, image.shape[1]), bounds_y=(0, image.shape[0]))
                rect.shift(x=shift_x, y=shift_y)
                triggered_squares.append(rect)
            if cells_sum > max_cells_sum:
                max_cells_sum = cells_sum
    return triggered_squares, max_cells_sum


def plot_calo_image_with_triggered_frame(images, idx, threshold, show=True, ax=None):
    assert images.shape[1] == images.shape[2] == 64, (
        "Only shape (64, 64) supported. If you want to change, take care of the shift for the inner image!!"
    )
    image = images[idx].reshape([images.shape[1], images.shape[2]])
    inner, outer = separate_images(image.reshape([1, image.shape[0], image.shape[1]]))
    inner = inner.reshape([inner.shape[1], inner.shape[2]])
    outer = outer.reshape([outer.shape[1], outer.shape[2]])
    outer = double_image(outer)

    inner_triggered, max_cells_sum1 = get_triggered_squares(image=inner, width=2, threshold=threshold, shift_x=16, shift_y=18)
    outer_triggered, max_cells_sum2 = get_triggered_squares(image=outer, width=4, threshold=threshold, stride=2)

    max_cells_sum = np.max([max_cells_sum1, max_cells_sum2])
    if show or (ax is not None):
        add_colorbar = False
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            add_colorbar = True

        title = ""
        if len(inner_triggered) > 0:
            title += "Inner triggered"
        if len(outer_triggered) > 0:
            title += "\nOuter triggered"
        elif title == "":
            title = "Not triggered"
        title += "\nMax sum: {}.".format(max_cells_sum)
        im = ax.imshow(image)
        for i, square in enumerate(inner_triggered):
            frame = patches.Rectangle(square.get_corner(), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(frame)
        for i, square in enumerate(outer_triggered):
            frame = patches.Rectangle(square.get_corner(), 4, 4, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(frame)
        ax.set_title(title)

        if add_colorbar:
            fig.colorbar(im, ax=ax, shrink=0.6)
        if show:
            plt.show()
        return inner_triggered, outer_triggered, ax
    else:
        return inner_triggered, outer_triggered


def is_TIS_or_TOS(signal_squares, inner_squares, outer_squares):
    is_TOS = False
    is_TIS = False
    trigger_squares = inner_squares[:]
    trigger_squares.extend(outer_squares)
    for trigger_square in trigger_squares:
        breaks = False
        for signal_square in signal_squares:
            if trigger_square.overlaps(signal_square):
                is_TOS = True
                breaks = True
                break
        if not breaks:
            is_TIS = True

    return is_TIS, is_TOS


def plot_calo_image_with_triggered_and_signal_frame(
        tracker_images, tracker_events, fr_width, idx, calo_images, threshold, padding=None, show=True, axs=None
    ):
    assert tracker_images.shape == calo_images.shape[:3], (
        "Same shape for tracker and calorimeter neeeded. Tracker: {}. Calorimeter: {}.".format(tracker_images.shape, calo_images.shape)
    )
    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 12))

    signal_squares, _ = plot_tracker_image_with_signal_frame(
        images=tracker_images, events=tracker_events, fr_width=fr_width, idx=idx, padding=padding, show=False, ax=axs[0]
    )
    inner_squares, outer_squares, _ = plot_calo_image_with_triggered_frame(
        images=calo_images, idx=idx, threshold=threshold, show=False, ax=axs[1]
    )

    is_TIS, is_TOS = is_TIS_or_TOS(signal_squares=signal_squares, inner_squares=inner_squares, outer_squares=outer_squares)

    image = calo_images[idx].reshape([calo_images.shape[1], calo_images.shape[2]])
    if show or (axs is not None):
        im = axs[2].imshow(image)
        for i, square in enumerate(signal_squares):
            frame = patches.Rectangle(square.get_corner(), fr_width, fr_width, linewidth=1, edgecolor='g', facecolor='none')
            axs[2].add_patch(frame)
        for i, square in enumerate(inner_squares):
            frame = patches.Rectangle(square.get_corner(), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
            axs[2].add_patch(frame)
        for i, square in enumerate(outer_squares):
            frame = patches.Rectangle(square.get_corner(), 4, 4, linewidth=1, edgecolor='r', facecolor='none')
            axs[2].add_patch(frame)

        axs[2].set_title(
            "TOS: {} - TIS: {}\nTruth TOS: {} - TIS: {}".format(
                is_TOS, is_TIS, tracker_events.iloc[idx]["is_TOS"], tracker_events.iloc[idx]["is_TIS"]
            )
        )
        if show:
            plt.show()
        return signal_squares, inner_squares, outer_squares, fig, axs
    else:
        return signal_squares, inner_squares, outer_squares



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


    # plot_tracker_image_with_signal_frame(images=tracker_images_m, events=tracker_events_m, fr_width=7, idx=4, padding=padding, show=True)
    # plot_calo_image_with_triggered_frame(images=mc_data_images, idx=16, threshold=3600, show=True)
    for idx in range(0, 1000):
        if tracker_events_m.iloc[idx]["is_TOS"]:
            print(idx)
            plot_calo_image_with_triggered_and_signal_frame(
                tracker_images=tracker_images_m, tracker_events=tracker_events_m, fr_width=7, idx=idx,
                calo_images=mc_data_images, threshold=3200, padding=padding, show=False, axs=None
            )
        plt.show()
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

