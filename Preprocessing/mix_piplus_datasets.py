#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-23 14:45:06
    # Description :
####################################################################################
"""
import os
import pickle

import numpy as np
import pandas as pd
import initialization as init


#############################################################################################################
############ Global variables
#############################################################################################################
size = "LargeSample"

fraction_highP = 0.5
fraction_lowP = 0.7
np_seed = 20200811

savepath = "../../Data/PiplusMixedP/{}".format(size)

if not os.path.exists("../../Data/PiplusMixedP/"):
    os.mkdir("../../Data/PiplusMixedP/")
if not os.path.exists(savepath):
    os.mkdir(savepath)


#############################################################################################################
############ Load data
#############################################################################################################
path_loading_highP = "../../Data/Piplus/{}".format(size)
data_highP = init.load_data(data_path=path_loading_highP, mode="all")
tracker_events_highP = data_highP["tracker_events"]
calo_images_highP = data_highP["calo_images"]
tracker_images_highP = data_highP["tracker_images"]
calo_events_highP = data_highP["calo_events"]

print("Higher P Info:")
print("T. events, T. images, C. images, C. inner, C. outer")
print(tracker_events_highP.shape, tracker_images_highP.shape, calo_images_highP.shape,
      calo_events_highP["calo_ET_inner"].shape, calo_events_highP["calo_ET_outer"].shape)


path_loading_lowP = "../../Data/PiplusLowerP/{}".format(size)
data_lowP = init.load_data(data_path=path_loading_lowP, mode="all")
tracker_events_lowP = data_lowP["tracker_events"]
calo_images_lowP = data_lowP["calo_images"]
tracker_images_lowP = data_lowP["tracker_images"]
calo_events_lowP = data_lowP["calo_events"]

print("Lower P Info:")
print("T. events, T. images, C. images, C. inner, C. outer")
print(tracker_events_lowP.shape, tracker_images_lowP.shape, calo_images_lowP.shape,
      calo_events_lowP["calo_ET_inner"].shape, calo_events_lowP["calo_ET_outer"].shape)

#############################################################################################################
############ Sample data
#############################################################################################################
np.random.seed(np_seed)

nr_sampled_highP = int(len(tracker_events_highP) * fraction_highP)
idx_highP = np.random.choice(len(tracker_events_highP), nr_sampled_highP, replace=False)
tracker_events_highP_sampled = tracker_events_highP.iloc[idx_highP, :]
calo_images_highP_sampled = calo_images_highP[idx_highP]
tracker_images_highP_sampled = tracker_images_highP[idx_highP]
calo_events_highP_inner_sampled = calo_events_highP["calo_ET_inner"][idx_highP]
calo_events_highP_outer_sampled = calo_events_highP["calo_ET_outer"][idx_highP]


nr_sampled_lowP = int(len(tracker_events_lowP) * fraction_lowP)
idx_lowP = np.random.choice(len(tracker_events_lowP), nr_sampled_lowP, replace=False)
tracker_events_lowP_sampled = tracker_events_lowP.iloc[idx_lowP, :]
calo_images_lowP_sampled = calo_images_lowP[idx_lowP]
tracker_images_lowP_sampled = tracker_images_lowP[idx_lowP]
calo_events_lowP_inner_sampled = calo_events_lowP["calo_ET_inner"][idx_lowP]
calo_events_lowP_outer_sampled = calo_events_lowP["calo_ET_outer"][idx_lowP]


#############################################################################################################
############ Concatenate data
#############################################################################################################
tracker_events = pd.concat([tracker_events_highP_sampled, tracker_events_lowP_sampled])
calo_images = np.concatenate([calo_images_highP_sampled, calo_images_lowP_sampled])
tracker_images = np.concatenate([tracker_images_highP_sampled, tracker_images_lowP_sampled])
calo_events = {
    "calo_ET_inner": np.concatenate([calo_events_highP_inner_sampled, calo_events_lowP_inner_sampled]),
    "calo_ET_outer": np.concatenate([calo_events_highP_outer_sampled, calo_events_lowP_outer_sampled])
}


print("\n")
print("Tracker events / high / low: ", tracker_events.shape, tracker_events_highP_sampled.shape, tracker_events_lowP_sampled.shape)
print("Tracker images / high / low: ", tracker_images.shape, tracker_images_highP_sampled.shape, tracker_images_lowP_sampled.shape)
print("Calo images / high / low: ", calo_images.shape, calo_images_highP_sampled.shape, calo_images_lowP_sampled.shape)
print("Calo inner / high / low: ", calo_events["calo_ET_inner"].shape, calo_events_highP_inner_sampled.shape, calo_events_lowP_inner_sampled.shape)
print("Calo outer / high / low: ", calo_events["calo_ET_outer"].shape, calo_events_highP_outer_sampled.shape, calo_events_lowP_outer_sampled.shape)


#############################################################################################################
############ Save data
#############################################################################################################

tracker_events.to_csv(savepath+"/tracker_events.csv", index=False)
with open(savepath+"/tracker_images.pickle", "wb") as f:
    pickle.dump(tracker_images, f, protocol=4)
with open(savepath+"/calo_images.pickle", "wb") as f:
    pickle.dump(calo_images, f, protocol=4)
with open(savepath+"/calo_events.pickle", "wb") as f:
    pickle.dump(calo_events, f, protocol=4)


with open(savepath+"/README.txt", "w") as f:
    f.write("Fraction High P: {}\n".format(fraction_highP))
    f.write("Fraction Low P: {}\n".format(fraction_lowP))
    f.write("Seed: {}\n\n".format(np_seed))

    f.write("Tracker events / high / low: {} / {} / {}\n".format(
        tracker_events.shape, tracker_events_highP_sampled.shape, tracker_events_lowP_sampled.shape
    ))
    f.write("Tracker images / high / low: {} / {} / {}\n".format(
        tracker_images.shape, tracker_images_highP_sampled.shape, tracker_images_lowP_sampled.shape
    ))
    f.write("Calo images / high / low: {} / {} / {}\n".format(
        calo_images.shape, calo_images_highP_sampled.shape, calo_images_lowP_sampled.shape
    ))
    f.write("Calo inner / high / low: {} / {} / {}\n".format(
        calo_events["calo_ET_inner"].shape, calo_events_highP_inner_sampled.shape, calo_events_lowP_inner_sampled.shape
    ))
    f.write("Calo outer / high / low: {} / {} / {}".format(
        calo_events["calo_ET_outer"].shape, calo_events_highP_outer_sampled.shape, calo_events_lowP_outer_sampled.shape
    ))
