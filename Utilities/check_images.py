#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-10 11:39:45
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


load_path = "../../Data/B2Dmunu/TestingPurpose"
with open(load_path+"/tracker_images.pickle", "rb") as f:
    tracker_images = pickle.load(f)[:15000]
with open(load_path+"/calo_images.pickle", "rb") as f:
    mc_data = pickle.load(f)[:15000]

if os.path.exists(load_path+"/identical_images.pickle"):
    with open(load_path+"/identical_images.pickle", "rb") as f:
        identical_images = pickle.load(f)
else:
    look_at = [True for i in range(len(mc_data))]
    identical_images = {}
    for i in range(len(mc_data)):
        if i % 100 == 0:
            print(i, "/", len(mc_data))
        if not look_at[i]:
            continue
        diffs = np.array([np.sum(np.abs(mc_data[j]-mc_data[i])) for j in range(i+1, len(mc_data))])
        diffs0 = i+1+np.where(diffs==0)[0]
        if len(diffs0)>0:
            identical_images[i] = list(diffs0)
            for j in diffs0:
                look_at[j] = False
    with open(load_path+"/identical_images.pickle", "wb") as f:
        pickle.dump(identical_images, f)
        raise

nr_copies = []
for key in identical_images:
    nr_copies.append(len(identical_images[key]))

plt.figure()
plt.hist(nr_copies, bins=60)
plt.show()

for key in identical_images:
    items = identical_images[key]
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(tracker_images[key])
    axs[1].imshow(mc_data[key])
    fig.suptitle("index:"+str(key))
    for item in items:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(tracker_images[item])
        axs[1].imshow(mc_data[item])
        fig.suptitle("Index: {}. Tracker difference: {}".format(item, np.sum(np.abs(tracker_images[key]-tracker_images[item]))))
    plt.show()
