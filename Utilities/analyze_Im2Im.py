#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-15 20:06:57
    # Description :
####################################################################################
"""
import sys
sys.path.insert(1, "../Preprocessing")
import pickle

import numpy as np
import pandas as pd
import initialization as init
import matplotlib.pyplot as plt

from TrainedGenerator import TrainedGenerator
from functionsOnImages import padding_zeros, get_layout, build_image, build_images, savefigs
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, get_energy_resolution
from functionsOnImages import build_histogram_HTOS, crop_images, clip_outer


#####################################################################################################
# Model loading
#####################################################################################################
do = 0
piplus_energy = ["PiplusLowerP", "PiplusMixedP"][do]
model = ["CWGANGP8", "CWGANGP7"][do]
model_path = "../../Results/{}/{}/".format(piplus_energy, model)
meta_path = model_path + "TFGraphs/"
config_path = model_path + "config.json"
Generator = TrainedGenerator(path_to_meta=meta_path, path_to_config=config_path)
keep_cols = Generator._config["keep_cols"]

#####################################################################################################
# Data loadng
#####################################################################################################

simulation = "B2Dmunu"
path_loading = "../../Data/{}/LargeSample".format(simulation)
nr_sim = 1
image_shape = [56, 64]
padding = {"top": 2, "bottom": 2, "left":0, "right":0}


if simulation == "Piplus":
    data, scaler = init.load_processed_data(path_loading, mode="train")
    calo_images = data["Calo"]*scaler["Calo"] / 1000
    calo_images = padding_zeros(calo_images, **padding)
    with open(path_loading+"/tracker_images.pickle".format(nr_sim), "rb") as f:
        tracker_images = pickle.load(f)[data["Idx"]]
    tracker_events = pd.read_csv(path_loading+"/tracker_events.csv").loc[data["Idx"]]

    tracker_real_ET = tracker_events["real_ET"].values
    exclude = ["phi", "theta", "region"]
    columns = tracker_events.drop(exclude, axis=1).columns
    transformed_events = scaler["Tracker"].transform(tracker_events.drop(exclude, axis=1))
    transformed_events = pd.DataFrame(data=transformed_events, columns=columns, index=tracker_events.index)
    tracker_events = pd.concat([transformed_events, tracker_events[exclude]], axis=1)

elif simulation == "B2Dmunu":
    with open("../../Data/Piplus/LargeSample/ProcessedScaler.pickle", "rb") as f:
        scaler = pickle.load(f)
    with open(path_loading+"/calo_images.pickle", "rb") as f:
        calo_images = pickle.load(f) / 1000
        calo_images = padding_zeros(calo_images, **padding)
    with open(path_loading+"/tracker_images.pickle", "rb") as f:
        tracker_images = pickle.load(f)
    with open(path_loading+"/tracker_input.pickle", "rb") as f:
        tracker_events = pickle.load(f)
    with open(path_loading+"/tracker_events.pickle", "rb") as f:
        tracker_real_ET = pickle.load(f)["real_ET"].apply(np.sum).values

else:
    raise ValueError("Non valid value for 'simulation'.")


with open(path_loading+"/Trained/{}_{}_out_{}.pickle".format(piplus_energy, model, nr_sim), "rb") as f:
    generated_images = pickle.load(f) / 1000 # Convert to GeV
    generated_images = clip_outer(generated_images, scaler["Calo"]/4/1000)

#####################################################################################################
# Verification
#####################################################################################################
figs = []
plt.figure()
calo_images = calo_images.reshape([-1, *image_shape])
generated_images = generated_images.reshape([-1, *image_shape])


fig2, axes = plt.subplots(3, 3, figsize=(20, 20))
axes = np.ravel(axes)
use_functions = {get_energies: {"energy_scaler": 1}, get_max_energy: {"energy_scaler": 1, "maxclip": 6.120},
                get_number_of_activated_cells: {"threshold": 5/1000},
                get_center_of_mass_x: {"image_shape": image_shape}, get_center_of_mass_y: {"image_shape": image_shape},
                get_std_energy: {"energy_scaler": 1}, get_energy_resolution: {"real_ET": tracker_real_ET/1000, "energy_scaler": 1}}
colnames = ["Enery", "MaxEnergy", "Cells", "X CoM", "Y CoM", "StdEnergy", "Resolution", "dN/dE", "(dN/dE) / (dN/dE)"]

htos_calo_images_real = crop_images(calo_images, **padding)
htos_calo_images_fake = crop_images(generated_images, **padding)
for func_idx, (func, params) in enumerate(use_functions.items()):
    if func.__name__ in ["get_number_of_activated_cells", "get_max_energy"]:
        build_histogram(true=htos_calo_images_real, fake=htos_calo_images_fake, function=func,
                        name=colnames[func_idx], epoch="", folder=None, ax=axes[func_idx], **params)
    else:
        build_histogram(true=calo_images, fake=generated_images, function=func,
                        name=colnames[func_idx], epoch="", folder=None, ax=axes[func_idx], **params)

build_histogram_HTOS(true=htos_calo_images_real, fake=htos_calo_images_fake,
                     energy_scaler=1, threshold=3.6, real_ET=tracker_real_ET,
                     ax1=axes[7], ax2=axes[8])
figs.append(fig2)

for idx in range(10):
    print(idx+1, "/", 10)
    use_functions = {
                    get_energies: {"energy_scaler": 1}, get_max_energy: {"energy_scaler": 1, "maxclip": 6.120},
                    get_number_of_activated_cells: {"threshold": 5/1000},
                    get_center_of_mass_x: {"image_shape": image_shape}, get_center_of_mass_y: {"image_shape": image_shape},
                    get_std_energy: {"energy_scaler": scaler["Calo"]}, get_energy_resolution: {"real_ET": tracker_real_ET[idx]/1000, "energy_scaler": 1}
    }

    figs.append(Generator.build_simulated_events(condition=tracker_events[idx],
                                    tracker_image=[tracker_images[idx]],
                                    calo_image=np.array([calo_images[idx]]),
                                    gen_scaler=scaler["Calo"]/1000,
                                    n=10,
                                    eval_functions=use_functions,
                                    title=model,
                                    reference_images=None)[0]
    )

savefigs(figures=figs, save_path=path_loading+"/Trained/Evaluation_{}_{}_{}.pdf".format(piplus_energy, model, nr_sim))