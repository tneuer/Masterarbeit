#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-03 12:46:41
    # Description :
####################################################################################
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from initialization import load_data
from root_preprocessing import convert_tracker_event_to_image, scale_to_grid_coordinate
from sklearn.base import BaseEstimator, TransformerMixin


data_path = "../../Data/Piplus/Full"
particle = "piplus"
data = load_data(data_path, particle, mode="events")
tracker_events = data["tracker_events"]
calo_images_inner = data["calo_events"]["calo_ET_inner"]
calo_images_outer = data["calo_events"]["calo_ET_outer"]

is_inner = tracker_events["region"] == 1
is_outer = tracker_events["region"] == 0

tracker_dim = [26, 32]
box_x = 4
box_y = 3

#############################################################################################################
############ Data exploration
#############################################################################################################
# is_region = is_outer
# tracker_events = tracker_events[is_region]
# calo_images_inner = calo_images_inner[is_region]
# calo_images_outer = calo_images_outer[is_region]

# tracker_images = convert_tracker_event_to_image(events=tracker_events, tracker_dim=tracker_dim, save_path=None, show_id=None)

# idxs = np.random.choice(np.arange(1000) ,size=10, replace=False)
# fig, axs = plt.subplots(nrows=10, ncols=3, figsize=(3*4, 10*4), facecolor='w', edgecolor='k')
# fig.subplots_adjust(wspace=0.01, hspace=0.05)
# for idx, row in zip(idxs, axs):
#     EMAX = np.max(calo_images_inner[idx])
#     row[0].imshow(tracker_images[idx])
#     row[1].imshow(calo_images_inner[idx], vmin=0, vmax=EMAX)
#     row[2].imshow(calo_images_outer[idx], vmin=0, vmax=EMAX)
# plt.show()


#############################################################################################################
# Cleaning process
#############################################################################################################

def is_out_of_bounds(centers, half_sides, minvals, maxvals):
    is_out_x = is_out_of_bounds_side(centers[0], half_sides[0], minvals[0], maxvals[0])
    is_out_y = is_out_of_bounds_side(centers[1], half_sides[1], minvals[1], maxvals[1])
    return np.logical_or(is_out_x, is_out_y)


def is_out_of_bounds_side(center, half_side, minval, maxval):
    return np.logical_or( (center - half_side) < minval, (center + half_side) > maxval)


def is_empty(calo_images):
    return np.array([np.sum(image)==0 for image in calo_images])


def has_energy_in_inner(inner_calorimeter_images, outer_calorimeter_images=None, threshold=0):
    if outer_calorimeter_images is None:
        return np.array([np.sum(image)>threshold for image in inner_calorimeter_images])
    else:
        assert 0 <= threshold <= 1, "If outer given, threshold has to be in [0, 1]."
        return np.array([ ( np.sum(image_in)/np.sum(image_out) ) > threshold
                        for image_in, image_out in zip(inner_calorimeter_images, outer_calorimeter_images)
                        ])


def has_low_energy_percentage(centered_images, outer_calorimeter_images, threshold=0.95):
    assert 0 <= threshold <= 1, "Threshold has to be in [0, 1]."
    return np.array([ ( np.sum(image_cent)/np.sum(image_out) ) < threshold
                  for image_cent, image_out in zip(centered_images, outer_calorimeter_images)
                  ])


def delete_with_mask(datasets, masf_func, delete_mask=None, **kwargs):
    if delete_mask is None:
        delete_mask = masf_func(**kwargs)
    message = "Keep: {} / {} (-{} {}).".format(sum(~delete_mask), datasets[0].shape[0], sum(delete_mask), masf_func.__name__)
    return [dataset[~delete_mask] for dataset in datasets], message



tracker_x = scale_to_grid_coordinate(values=tracker_events["x_projections"], minval=0, maxval=tracker_dim[1]-1)
tracker_y = scale_to_grid_coordinate(values=tracker_events["y_projections"], minval=0, maxval=tracker_dim[0]-1)

datasets = [tracker_events, calo_images_inner, calo_images_outer, tracker_x, tracker_y]
datasets, m2 = delete_with_mask(datasets=datasets, masf_func=is_out_of_bounds,
                            centers=(datasets[-2], datasets[-1]), half_sides=(box_x, box_y), minvals=(0, 0), maxvals=(tracker_dim[1]-1, tracker_dim[0]-1)
                            )


datasets, m3 = delete_with_mask(datasets=datasets, masf_func=is_empty, calo_images=datasets[2])


datasets, m4 = delete_with_mask(datasets=datasets, masf_func=has_energy_in_inner,
                            inner_calorimeter_images=datasets[1], outer_calorimeter_images=datasets[2], threshold=0.05
                            )


#############################################################################################################
# Centering process
#############################################################################################################
calo_images_outer = datasets[2]
tracker_x = datasets[3].values
tracker_y = datasets[4].values

centered_images = np.stack([image[(y-box_y):(y+box_y+1), (x-box_x):(x+box_x+1)] for image,x,y in zip(calo_images_outer, tracker_x, tracker_y)], axis=0)

datasets.append(centered_images)
datasets, m5 = delete_with_mask(datasets=datasets, masf_func=has_low_energy_percentage,
                            centered_images=centered_images, outer_calorimeter_images=calo_images_outer, threshold=0.95
                            )


message = "\nPixelbounds: ({}, {})".format(box_x, box_y) + "\nNew shape: ({}, {})\n".format(2*box_x+1, 2*box_y+1) + "\n".join([m2, m3, m4, m5])
print(message)


#############################################################################################################
# Saving
#############################################################################################################
datasets[0].to_csv(data_path+"/Centered_{}_tracker_events.csv".format(particle), index=False)
with open(data_path+"/Centered_Original_{}_calo_images.pickle".format(particle), "wb") as f:
    pickle.dump(datasets[2], f)
frame = {"X": datasets[3], "Y": datasets[4]}
pd.DataFrame(frame).to_csv(data_path+"/CenteredXY.csv")
with open(data_path+"/Centered_{}_calo_images.pickle".format(particle), "wb") as f:
    pickle.dump(datasets[5], f)


with open(data_path+"/README.txt", "a") as f:
    f.write(message)


#############################################################################################################
# Evaluation
#############################################################################################################
calo_images_inner = datasets[1]
calo_images_outer = datasets[2]
tracker_x = datasets[3].values
tracker_y = datasets[4].values
centered_images = datasets[5]
fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i, idx in enumerate(np.random.choice(np.arange(1000) ,size=rows, replace=False)):
    EMAX = np.max(calo_images_outer[idx])
    tracker_image = np.zeros_like(calo_images_outer[0])
    tracker_image[tracker_y[idx], tracker_x[idx]] = 1

    fig.add_subplot(rows, columns, i*columns+1)
    plt.imshow(tracker_image)
    plt.axis('off')

    fig.add_subplot(rows, columns, i*columns+2)
    plt.imshow(centered_images[idx], vmin=0, vmax=EMAX)
    plt.axis('off')
    plt.title("{} / {}".format(np.max(centered_images[idx]), np.sum(centered_images[idx])))

    fig.add_subplot(rows, columns, i*columns+3)
    plt.imshow(calo_images_outer[idx], vmin=0, vmax=EMAX)
    plt.axis('off')
    plt.title("{} / {}".format(np.max(calo_images_outer[idx]), np.sum(calo_images_outer[idx])))

    fig.add_subplot(rows, columns, i*columns+4)
    plt.imshow(calo_images_inner[idx], vmin=0, vmax=EMAX)
    plt.axis('off')
    plt.title("{} / {}".format(np.max(calo_images_inner[idx]), np.sum(calo_images_inner[idx])))
plt.savefig(data_path+"/CenteredImages.png")
plt.show()