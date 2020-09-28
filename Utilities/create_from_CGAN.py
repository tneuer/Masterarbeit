#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-15 18:34:12
    # Description :
####################################################################################
"""

import os
import re
import sys
sys.path.insert(1, '../Preprocessing')
import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import initialization as init
import matplotlib.pyplot as plt
import initialization as init

from TrainedGenerator import TrainedGenerator
from functionsOnImages import padding_zeros, savefigs
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, get_energy_resolution
from functionsOnImages import build_histogram_HTOS, crop_images


#####################################################################################################
# Parameter definition
#####################################################################################################
do = 0
datasize = "TestingPurpose"
piplus_energy = ["PiplusLowerP", "PiplusMixedP"][do]
model = ["CWGANGP8", "CWGANGP7"][do]
batch_size = 100
generate_images = False
evaluate_network = True
nr_samples = 10

data_path = "../../Data/{}/{}".format(piplus_energy, datasize)
saving_path = data_path + "/Trained"
model_path = "../../Results/{}/{}/".format(piplus_energy, model)
meta_path = model_path + "TFGraphs/"
config_path = model_path + "config.json"

if not os.path.exists(saving_path):
    os.mkdir(saving_path)

#####################################################################################################
# Data loading
#####################################################################################################
Generator = TrainedGenerator(path_to_meta=meta_path, path_to_config=config_path)
with open(config_path, "r") as f:
    config = json.load(f)
    padding = config["padding"]
    keep_cols = config["keep_cols"]

data, scaler = init.load_processed_data(data_path=data_path, mode="test", return_scaler=True)
calo_images = padding_zeros(data["Calo"], **padding)
tracker_events = data["Tracker"]
tracker_events_list = []
for _, row in tracker_events.iterrows():
    tracker_events_list.append([row[keep_cols].tolist()])


#####################################################################################################
# Generation of predictions
#####################################################################################################
if generate_images:
  print("""
        #############################
        # Predicting
        #############################
        """)
  generated_images = Generator.generate_batches(list_of_inputs=tracker_events_list, batch_size=batch_size)
  with open(saving_path+"/"+piplus_energy+"_"+model+"_out.pickle", "wb") as f:
      pickle.dump(generated_images, f)


#####################################################################################################
# Evaluation of predictions
#####################################################################################################
if evaluate_network:
  print("""
        #############################
        # Evaluating
        #############################
        """)
  if not generate_images:
      with open(saving_path+"/"+piplus_energy+"_"+model+"_out.pickle", "rb") as f:
          generated_images = pickle.load(f)

  with open(data_path+"/tracker_images.pickle", "rb") as f:
      tracker_images = pickle.load(f)
  tracker_et = scaler["Tracker"].inverse_transform(tracker_events[scaler["Names"]])
  tracker_et = tracker_et[:, scaler["Names"]=="real_ET"] / 1000

  tracker_images = tracker_images[data["Idx"]]
  calo_images = calo_images*scaler["Calo"]/1000
  generated_images = generated_images*scaler["Calo"]/1000
  image_shape = calo_images.shape[1:]

  print(np.max(calo_images), np.max(generated_images))
  print(calo_images.shape, generated_images.shape, tracker_images.shape, tracker_et.shape)

  figs = []
  fig2, axes = plt.subplots(2, 5, figsize=(20, 12))
  axes = np.ravel(axes)
  use_functions = {
      get_energies: {"energy_scaler": 1}, get_max_energy: {"energy_scaler": 1, "maxclip": 6.120},
      get_number_of_activated_cells: {"threshold": 5/1000}, get_center_of_mass_x: {"image_shape": image_shape},
      get_center_of_mass_y: {"image_shape": image_shape}, get_std_energy: {"energy_scaler": 1},
      get_energy_resolution: {"real_ET": tracker_et/1000, "energy_scaler": 1}
  }

  colnames = ["Energy", "MaxEnergy", "Cells", "X CoM", "Y CoM", "StdEnergy", "Resolution", "dN/dE", "(dN/dE) / (dN/dE)"]
  for func_idx, (func, params) in enumerate(use_functions.items()):
      if func.__name__ in ["get_number_of_activated_cells", "get_max_energy"]:
          build_histogram(true=calo_images, fake=generated_images, function=func,
                          name=colnames[func_idx], epoch="", folder=None, ax=axes[func_idx], **params)
      else:
          build_histogram(true=calo_images, fake=generated_images, function=func,
                          name=colnames[func_idx], epoch="", folder=None, ax=axes[func_idx], **params)
  build_histogram_HTOS(true=calo_images, fake=generated_images,
                       energy_scaler=1, threshold=3.6, real_ET=tracker_et,
                       ax1=axes[7], ax2=axes[8])
  axes[-1].scatter(tracker_et, get_energies(calo_images), label="true", alpha=0.1)
  axes[-1].scatter(tracker_et, get_energies(generated_images), label="fake", alpha=0.1)
  axes[-1].legend()
  axes[-1].set_xlim(0, np.max(tracker_et))
  axes[-1].set_ylim(0, np.max(tracker_et))
  axes[-1].set_xlabel("Tracker")
  axes[-1].set_ylabel("Calorimeter")
  figs.append(fig2)

  for idx in range(nr_samples):
      print(idx+1, "/", nr_samples)
      ref = tracker_events.iloc[idx]
      is_reference = (
              (tracker_events["theta"] > ref["theta"]-0.01) &
              (tracker_events["theta"] < ref["theta"]+0.01) &
              (np.abs(tracker_events["real_ET"]) > np.abs(ref["real_ET"]-0.3*ref["real_ET"])) &
              (np.abs(tracker_events["real_ET"]) < np.abs(ref["real_ET"]+0.3*ref["real_ET"]))
      )
      use_functions[get_energy_resolution]["real_ET"] = ref["real_ET"]

      fig, ax = Generator.build_simulated_events(
          condition=tracker_events_list[idx],
          tracker_image=tracker_images[idx],
          calo_image=calo_images[idx], n=500, eval_functions=use_functions, title=model+": "+str(idx),
          reference_images=calo_images[is_reference]
      )
      figs.append(fig)

  savefigs(figures=figs, save_path=saving_path+"/"+piplus_energy+"_"+model+"_Evaluation.pdf")


