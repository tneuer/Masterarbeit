#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-20 13:30:57
    # Description :
####################################################################################
"""

import os
import matplotlib.backends.backend_pdf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numba import njit
from scipy import signal

def savefigs(figures, save_path):
    pdf = matplotlib.backends.backend_pdf.PdfPages(save_path)
    for figure in figures:
        pdf.savefig(figure)
    pdf.close()


#####################################################################################################
# Preprocessing images
#####################################################################################################


def scale_images(images, scaler):
    if not isinstance(scaler, (int, float, bool)):
        raise ValueError("Wrong input for image scaler. Needs boolean or numeric.")

    if scaler is True:
        scaler = np.max(images)
    elif scaler is False:
        scaler = 1

    scaled_images = images/scaler
    return scaled_images, scaler


def flatten_images(images):
    return images.reshape((len(images), np.prod(images.shape[1:])))


def padding_zeros(images, top=0, bottom=0, left=0, right=0):
    new_shape = list(images[0].shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    resized_images = np.zeros((len(images), *new_shape))
    resized_images[:, top:(new_shape[0]-bottom), left:(new_shape[1]-right)] = images[:, :, :]
    return resized_images


def crop_images(images, top=0, bottom=0, left=0, right=0):
    resized_images = images[:, top:images.shape[1]-bottom, left:images.shape[2]-right]
    return resized_images


def double_images(images):
    resized_images = np.array([double_image(image) for image in images])
    return resized_images


def double_image(image):
    resized_size = (image.shape[0]*2, image.shape[1]*2)
    resized_image = np.zeros(shape=resized_size)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_pixels_y = (i*2, i*2,
                            i*2 + 1, i*2 + 1)
            new_pixels_x = (j*2, j*2 + 1,
                            j*2, j*2 + 1)
            new_pixel_value = image[i, j] / 4
            resized_image[new_pixels_y, new_pixels_x] = new_pixel_value
    return resized_image


def insert(inner, outer):
    assert len(inner) == len(outer), "Not same amount of pictures in inner and outer"
    pics_idxs = [i for i in range(len(inner))]

    dim_x_outer = outer[0].shape[0]
    dim_y_outer = outer[0].shape[1]
    dim_x_inner = inner[0].shape[0]
    dim_y_inner = inner[0].shape[1]

    start_inner_x = dim_x_outer//2 - dim_x_inner//2
    stop_inner_x = dim_x_outer//2 + dim_x_inner//2
    start_inner_y = dim_y_outer//2 - dim_y_inner//2
    stop_inner_y = dim_y_outer//2 + dim_y_inner//2

    outer[pics_idxs, start_inner_x:stop_inner_x, start_inner_y:stop_inner_y] = inner[pics_idxs, :, :]
    assert np.array_equal(inner, outer[pics_idxs, start_inner_x:stop_inner_x, start_inner_y:stop_inner_y])
    return outer


def separate_images(images):
    if (images.shape[1] == 52) and (images.shape[2] == 64):
        inner_images = images[:, 12:40, 16:48]
        outer_images = np.copy(images)
        outer_images[:, 12:40, 16:48] = 0
        otr_resized_size = (26, 32)
    elif (images.shape[1] == 56) and (images.shape[2] == 64):
        inner_images = images[:, 14:42, 16:48]
        outer_images = np.copy(images)
        outer_images[:, 14:42, 16:48] = 0
        otr_resized_size = (28, 32)
    elif (images.shape[1] == 64) and (images.shape[2] == 64):
        inner_images = images[:, 18:46, 16:48]
        outer_images = np.copy(images)
        outer_images[:, 18:46, 16:48] = 0
        otr_resized_size = (32, 32)
    else:
        raise ValueError("Wrong image shape. Given {}.".format(images.shape))
    outer_images = halve_images(outer_images)

    assert inner_images.shape[1] == 28 and inner_images.shape[2] == 32, (
                "Wrong inner shape. Expected: {}. Given {}.".format("[-1, 26, 32]", inner_images.shape)
    )
    assert outer_images.shape[1] == otr_resized_size[0] and outer_images.shape[2] == otr_resized_size[1], (
                "Wrong outer shape. Expected: {}. Given {}.".format("[-1, 26, 32]", outer_images.shape)
    )
    return inner_images, outer_images


def halve_images(images):
    """ Halve image by summing up over four adjacent cells
    """
    halved_images = np.stack([halve_image(image) for image in images], axis=0)
    return halved_images


def halve_image(image):
    """ Halving image by summing up 4 pixels is the same as a convolution with a matrix of ones and stride two
    """
    assert image.shape[0] % 2 == 0 and image.shape[1] % 2 == 0
    def strideConv(image, mask, stride=1):
        return signal.convolve2d(image, mask[::-1, ::-1], mode='valid')[::stride, ::stride]

    resized_image = strideConv(image, mask=np.array([[1, 1], [1, 1]]), stride=2)
    return resized_image


def clip_outer(images, clipval):
    clipped_images = np.copy(images)
    if (images.shape[1] == 52) and (images.shape[2] == 64):
        inner_images = images[:, 12:40, 16:48]
        clipped_images = np.clip(images, a_min=0, a_max=clipval)
        clipped_images[:, 12:40, 16:48] = inner_images
    elif (images.shape[1] == 56) and (images.shape[2] == 64):
        inner_images = images[:, 14:42, 16:48]
        clipped_images = np.clip(images, a_min=0, a_max=clipval)
        clipped_images[:, 14:42, 16:48] = inner_images
    elif (images.shape[1] == 64) and (images.shape[2] == 64):
        inner_images = images[:, 18:46, 16:48]
        clipped_images = np.clip(images, a_min=0, a_max=clipval)
        clipped_images[:, 18:46, 16:48] = inner_images
    else:
        raise ValueError("Wrong image shape. Given {}.".format(images.shape))
    return clipped_images


#####################################################################################################
# Evaluating images
#####################################################################################################

def create_x_mask(image_shape):
    size_x = image_shape[1]
    size_y = image_shape[0]
    row_mask_x = np.arange(-size_x//2, size_x//2+1)
    row_mask_x = row_mask_x[ row_mask_x!=0 ]
    x_mask = np.repeat(row_mask_x, repeats=size_y).reshape([size_y, size_x], order="F")
    return x_mask


def create_y_mask(image_shape):
    size_x = image_shape[1]
    size_y = image_shape[0]
    row_mask_y = np.arange(-size_y//2, size_y//2+1)
    row_mask_y = row_mask_y[ row_mask_y != 0 ]
    y_mask = np.repeat(row_mask_y, repeats=size_x).reshape([size_y, size_x])
    return y_mask


def get_energies(images, energy_scaler=1):
    return np.array([np.sum(image)*energy_scaler for image in images])


def get_triggered_energies(images, energy_scaler, cells, threshold):
    triggered_images = get_triggered_images(images, energy_scaler, cells, threshold)
    return get_energies(triggered_images)


def get_triggered_images(images, energy_scaler, cells, threshold):
    is_triggered = is_triggered_images(images, energy_scaler, cells, threshold)
    return images[is_triggered]


def is_triggered_images(images, energy_scaler, threshold):
    inner, outer = separate_images(images)
    is_triggered_inner = [is_triggered_image(inr, energy_scaler, threshold) for inr in inner]
    is_triggered_outer = [is_triggered_image(otr, energy_scaler, threshold) for otr in outer]
    is_triggered = [inr or otr for inr, otr in zip(is_triggered_inner, is_triggered_outer)]
    return is_triggered


def is_triggered_image(image, energy_scaler, threshold):
    """ Take 4 highest adjacent energy cells. If they are above the threshold it is triggered on this image.
    """
    def strideConv(image, mask, stride=1):
        return signal.convolve2d(image, mask[::-1, ::-1], mode='valid')[::stride, ::stride]
    resized_image = strideConv(image, mask=np.array([[1, 1], [1, 1]]), stride=1)
    if np.max(resized_image)*energy_scaler > threshold:
        return True
    return False


def get_max_energy(images, energy_scaler=1, maxclip=np.Inf):
    inner, outer = separate_images(images*energy_scaler)
    return np.clip([np.max([np.max(inr), np.max(otr)]) for inr, otr in zip(inner, outer)], 0, maxclip)


def get_std_energy(images, energy_scaler=1, threshold=-1):
    return [np.std(image[(image*energy_scaler)>threshold])*energy_scaler for image in images]


def get_number_of_activated_cells(images, threshold=0, energy_scaler=1):
    inner, outer = separate_images(images*energy_scaler)
    return np.array([np.sum(inr>threshold) + np.sum(otr>threshold) for inr, otr in zip(inner, outer)])


def get_center_of_mass_x(images, image_shape=None):
    if image_shape is None:
        image_shape = images[0].shape
        center = [np.sum(image*create_x_mask(image_shape))/np.sum(image) for image in images]
    else:
        center = [np.sum(image.reshape(image_shape[0], image_shape[1])*create_x_mask(image_shape))/np.sum(image) for image in images]
    return np.nan_to_num(center, nan=0)

def get_center_of_mass_y(images, image_shape=None):
    if image_shape is None:
        image_shape = images[0].shape
        center = [np.sum(image*create_y_mask(image_shape))/np.sum(image) for image in images]
    else:
        center = [np.sum(image.reshape(image_shape[0], image_shape[1])*create_y_mask(image_shape))/np.sum(image) for image in images]
    return np.nan_to_num(center, nan=0)

def get_center_of_mass_r(images, image_shape=None):
    if image_shape is None:
        image_shape = images[0].shape
        center = [np.sum(image*np.sqrt(create_x_mask(image_shape)**2+create_y_mask(image_shape)**2))/np.sum(image) for image in images]
    else:
        center = [np.sum(image.reshape(image_shape[0], image_shape[1])*np.sqrt(create_x_mask(image_shape)**2+
                                                                            create_y_mask(image_shape)**2))/np.sum(image) for image in images]
    return np.nan_to_num(center, nan=0)


def get_mean_image(images):
    return np.mean(images, axis=0, keepdims=False)


def get_energy_resolution(images, real_ET, energy_scaler=1):
    energies = np.array(get_energies(images, energy_scaler=energy_scaler))
    return (real_ET - energies)



#####################################################################################################
# Plotting
#####################################################################################################


def get_layout(n, nrows=None):
    if nrows is None:
        nrows = int(np.sqrt(n))
    ncols = int(np.ceil(n / nrows))
    return nrows, ncols


def build_image(image, colorbar=False):
    fig, ax = plt.subplots()
    im = ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=1, anchor=0.6)
    return fig, ax


def build_mean_images(data, column_titles=None, save_path=None):
    images_to_plot = [[get_mean_image(images) for images in data]]

    fig, ax = build_images(images_to_plot, column_titles, fs_x=20, fs_y=10)
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def build_images(array_of_images, column_titles=None, row_titles=None, colorbar=False, fs_x=45, fs_y=25):
    layout = (len(array_of_images), len(array_of_images[0]))
    if column_titles is None:
        column_titles = [" ".format(i+1) for i in range(layout[1])]
    if row_titles is None:
        row_titles = [" ".format(i+1) for i in range(layout[0])]


    fig, ax = plt.subplots(nrows=layout[0], ncols=layout[1], figsize=(layout[1]*4, layout[0]*4), facecolor='w', edgecolor='k')
    fig.subplots_adjust(wspace=0.01, hspace=0.05)

    E_min = 0
    ax = ax.reshape(layout)
    for i, row in enumerate(ax):
        E_max = np.max([np.max(image) for image in array_of_images[i]])
        for j, col in enumerate(row):
            im = col.imshow(array_of_images[i][j], vmin=E_min, vmax=E_max)
            col.get_xaxis().set_visible(False)

            if i == 0:
                col.set_title(column_titles[j], fontsize=fs_x)
            if j == 0:
                col.set_ylabel(row_titles[i], fontsize=fs_y)
            else:
                col.get_yaxis().set_visible(False)
        if colorbar:
            fig.colorbar(im, ax=ax.ravel()[layout[1]*i], shrink=1, anchor=0.6)
    return fig, ax


def build_tracker_calo_images(data, n, calo_components=False, seed=42, save_path=None):

    if seed is not None:
        np.random.seed(seed)

    nr_images = len(data["tracker_images"])
    indices = [i for i in range(nr_images)]
    rand_indices = np.random.choice(indices, size=n, replace=False)

    if calo_components:
        images_to_plot = [
            [
            data["tracker_images"][i],
            data["calo_images"][i],
            data["calo_events"]["calo_ET_inner"][i],
            data["calo_events"]["calo_ET_outer"][i]
            ]
            for i in rand_indices
        ]
        column_titles = ["Tracker", "Reconstructed Calo", "Inner Calo", "Outer Calo"]
    else:
        images_to_plot = [
            [
            data["tracker_images"][i],
            data["calo_images"][i]
            ]
            for i in rand_indices
        ]
        column_titles = ["Tracker", "Reconstructed Calo"]

    fig, ax = build_images(images_to_plot, column_titles, fs_x=20, fs_y=20)
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def build_histogram(true, function, name, fake=None, fake2=None, epoch=None, folder=None, ax=None, use_legend=False, labels=None,
                     **kwargs):
    eval_true = function(true, **kwargs)
    if labels is None:
        labels = ["true", "fake", "fake2"]
    if fake is not None:
        eval_fake = function(fake, **kwargs)

        if fake2 is not None:
            eval_fake2 = function(fake2, **kwargs)
            minval = np.min([np.min(eval_true), np.min(eval_fake), np.min(eval_fake2)])
            maxval = np.max([np.max(eval_true), np.max(eval_fake), np.max(eval_fake2)])
        else:
            minval = np.min([np.min(eval_true), np.min(eval_fake)])
            maxval = np.max([np.max(eval_true), np.max(eval_fake)])
    else:
        minval = np.min(eval_true)
        maxval = np.max(eval_true)

    if name.lower() == "resolution":
        minval = -2.5
    bins = np.linspace(minval, maxval, 20)

    if ax is None:
        _, ax = plt.subplots()

    counts_true, bin_edges = np.histogram(eval_true, bins=bins)
    bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])
    binwidth = (bins[1] - bins[0])
    binerror_true = np.sqrt(counts_true)

    ax.bar(bincenters, counts_true, width=binwidth, label=labels[0], fill=False, yerr=binerror_true, edgecolor="C0", ecolor="C0")

    if use_legend and fake is None:
        the_mean = np.mean(eval_true)
        the_std = np.std(eval_true)
        textstr = '\n'.join((
                r'$\mu=%.2f$' % (the_mean, ),
                r'$\sigma=%.2f$' % (the_std, )
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                                 verticalalignment='top', bbox=props)

    if fake is not None:
        counts_fake, bin_edges = np.histogram(eval_fake, bins=bins)
        binerror_fake = np.sqrt(counts_fake)
        ax.bar(bincenters, counts_fake, width=binwidth, label=labels[1], fill=False, yerr=binerror_fake, edgecolor="C1", ecolor="C1")
    if fake2 is not None:
        counts_fake2, bin_edges2 = np.histogram(eval_fake2, bins=bins)
        binerror_fake2 = np.sqrt(counts_fake2)
        ax.bar(bincenters, counts_fake2, width=binwidth, label=labels[2], fill=False, yerr=binerror_fake2, edgecolor="C2", ecolor="C2")
    ax.legend(loc='upper right')
    ax.set_title("{} {}".format(name, epoch))
    if folder is not None:
        plt.savefig(folder+"/Evaluation/{}/{}Distribution{}.png".format(name, name, epoch))

    return ax


def build_histogram_HTOS(true, fake, energy_scaler, threshold, real_ET, fake2=None, labels=None, ax1=None, ax2=None):
    if fake is not None:
        assert true.shape == fake.shape, "real and fake shape differ. real: {}, fake: {}.".format(true.shape, fake.shape)
        is_triggered_fake = is_triggered_images(fake, energy_scaler, threshold)
    if fake2 is not None:
        assert true.shape == fake2.shape, "real and fake2 shape differ. real: {}, fake2: {}.".format(true.shape, fake2.shape)
        is_triggered_fake2 = is_triggered_images(fake2, energy_scaler, threshold)
    if ax1 is None:
        _, ax1 = plt.subplots()
    if labels is None:
        labels = ["true", "fake", "fake2"]
    is_triggered_true = is_triggered_images(true, energy_scaler, threshold)

    bin_nr = 20
    bins = np.linspace(np.min(real_ET), np.max(real_ET), 20)
    nr_real, _, _ = ax1.hist(real_ET, bins=bins, label="Tracker", histtype="step", color="black")
    nr_true, _, _ = ax1.hist(real_ET[is_triggered_true], bins=bins, label=labels[0], histtype="step")

    if fake is not None:
        nr_fake, _, _ = ax1.hist(real_ET[is_triggered_fake], bins=bins, label=labels[1], histtype="step")
        ax1.set_title("HTOS reco: {}, GAN: {}\nConcordant: {}".format(
            sum(is_triggered_true), sum(is_triggered_fake), sum(np.logical_and(is_triggered_true, is_triggered_fake)))
        )
    if fake2 is not None:
        nr_fake2, _, _ = ax1.hist(real_ET[is_triggered_fake2], bins=bins, label=labels[2], histtype="step")
    else:
        ax1.set_title("HTOS reco: {}".format(sum(is_triggered_true)))
    ax1.legend(loc="upper right")

    midpoints = (bins[1:] + bins[:-1]) / 2
    if ax2 is None:
        _, ax2 = plt.subplots()
    trigger_rate_real = get_trigger_rate(reference=nr_true, real=nr_real)
    ax2.plot(midpoints, trigger_rate_real, label=labels[0])

    if fake is not None:
        trigger_rate_fake = get_trigger_rate(reference=nr_fake, real=nr_real)
        ax2.plot(midpoints, trigger_rate_fake, label=labels[1])
    if fake2 is not None:
        trigger_rate_fake2 = get_trigger_rate(reference=nr_fake2, real=nr_real)
        ax2.plot(midpoints, trigger_rate_fake2, label=labels[2])

    ax2.legend(loc="upper left")
    return ax1, ax2


def get_trigger_rate(reference, real):
    """ Compute trigger rate and if it is zero (mostly due to small sample size in high energy bins) set it to
    the previous trigger rate
    """
    trigger_rate = reference / real
    trigger_rate[np.isnan(trigger_rate)] = 0
    zero_idxs = np.where(trigger_rate==0)[0]
    if (len(zero_idxs) > 0) and (zero_idxs[0] == 0):
        zero_idxs[0] = 1
    for zero_idx in zero_idxs:
        trigger_rate[zero_idx] = trigger_rate[zero_idx-1]
    return trigger_rate
