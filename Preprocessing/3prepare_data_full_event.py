#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-03-05 15:24:19
    # Description :
####################################################################################
"""
import sys
import pickle
sys.path.insert(1, '../Utilities/')

import numpy as np
import pandas as pd
import initialization as init

import matplotlib.pyplot as plt

#############################################################################################################
############ Preprocess and load
#############################################################################################################
ref = "PiplusLowerP"
data_load_path = "../../Data/B2Dmunu/TestingPurpose"
figure_save_path = data_load_path
data = init.load_data(data_load_path, mode="all")

tracker_events = data["tracker_events"]
tracker_images = data["tracker_images"]
calo_events = data["calo_events"]
calo_images = data["calo_images"]

figs = []

# with open(data_load_path+"/tracker_events_original.pickle", "rb") as f:
#     tracker_events = pickle.load(f)
piplus_data = pd.read_csv("../../Data/{}/Debug/tracker_events.csv".format(ref))

#############################################################################################################
############ Outlier
#############################################################################################################
et_cutoff = 20000 #MeV
def contains_outlier(row):
    return np.any(row > et_cutoff)

is_no_outlier = ~tracker_events["real_ET"].apply(contains_outlier)
tracker_events = tracker_events[is_no_outlier]
tracker_images = tracker_images[is_no_outlier]
calo_events["calo_ET_inner"] = calo_events["calo_ET_inner"][is_no_outlier]
calo_events["calo_ET_outer"] = calo_events["calo_ET_outer"][is_no_outlier]
calo_images = calo_images[is_no_outlier]


#############################################################################################################
############ Scale variables and remove out of bounds
#############################################################################################################
with open("../../Data/{}/Debug/ProcessedScaler.pickle".format(ref), "rb") as f:
    scaler = pickle.load(f)

piplus_min_x, piplus_max_x = min(piplus_data["x_projections"]), max(piplus_data["x_projections"])
piplus_min_y, piplus_max_y = min(piplus_data["y_projections"]), max(piplus_data["y_projections"])
colnames = scaler["Names"]

piplus_data_std = piplus_data.copy()
tracker_events_std = tracker_events.copy()
for col in ["x_projections", "y_projections", "real_ET"]:
    col_idx = np.where(colnames==col)[0][0]
    std_mean = scaler["Tracker"].mean_[col_idx]
    std_var = scaler["Tracker"].var_[col_idx]

    if col in ["x_projections"]:
        tracker_events[col] = tracker_events[col].apply(lambda x: x[np.logical_and(x>piplus_min_x, x<piplus_max_x)])
    elif col in ["y_projections"]:
        tracker_events[col] = tracker_events[col].apply(lambda y: y[np.logical_and(y>piplus_min_y, y<piplus_max_y)])

    piplus_data_std[col] = (piplus_data[col] - std_mean) / np.sqrt(std_var)
    tracker_events_std[col] = tracker_events[col].apply(lambda x: (x - std_mean) / np.sqrt(std_var))


#############################################################################################################
############ Compare original piplus data with full event data
#############################################################################################################
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
fig.subplots_adjust(wspace=0.2, hspace=0.3)

for idx, col in enumerate(["x_projections", "y_projections", "real_ET"]):

    if col in ["x_projections", "y_projections"]:
        col_signal = col[:-1]
    else:
        col_signal = col
    full_event_data = np.concatenate(tracker_events[col].values)
    bins = np.linspace(
        np.min([np.min(full_event_data), np.min(piplus_data[col])]),
        np.max([np.max(full_event_data), np.max(piplus_data[col])]),
        50
    )
    axs[0, idx].hist(full_event_data, label="FullEvent", density=True, bins=bins, alpha=0.7)
    axs[0, idx].hist(piplus_data[col], label=ref, density=True, bins=bins, alpha=0.7)
    axs[0, idx].legend()
    axs[0, idx].set_title(col+": Mean / Var"+"\nPiplus: {} / {}\nFull: {} / {}".format(
        np.round(np.mean(piplus_data[col]), 3), np.round(np.var(piplus_data[col]), 3),
        np.round(np.mean(full_event_data)), np.round(np.var(full_event_data)),
    ))

    full_event_data_std = np.concatenate(tracker_events_std[col].values)
    bins_std = np.linspace(
        np.min([np.min(full_event_data_std), np.min(piplus_data_std[col])]),
        np.max([np.max(full_event_data_std), np.max(piplus_data_std[col])]),
        50
    )
    axs[1, idx].hist(full_event_data_std, label="FullEvent", density=True, bins=bins_std, alpha=0.7)
    axs[1, idx].hist(piplus_data_std[col], label=ref, density=True, bins=bins_std, alpha=0.7)
    axs[1, idx].legend()
    axs[1, idx].set_title("Standardized: Mean / Var"+"\nPiplus: {} / {}\nFull: {} / {}".format(
        np.round(np.mean(piplus_data_std[col]), 3), np.round(np.var(piplus_data_std[col]), 3),
        np.round(np.mean(full_event_data_std)), np.round(np.var(full_event_data_std), 3),
    ))

plt.savefig(data_load_path+"/ComparisonTo{}.png".format(ref), dpi=200)


#############################################################################################################
############ Convert to correct format [(x1, x2, x3, ...), (y1, y2, y3, ...), (p1, p2, p3, ...)] --->
############                            [(x1, y1, p1), (x2, y2, p2), (x3, y3, p3), ...]
#############################################################################################################
def concatenate_event(row):
    return list(zip(row["x_projections"], row["y_projections"], row["real_ET"]))

tracker_input = tracker_events.apply(concatenate_event, axis=1)
tracker_input = np.array([np.reshape([list(track) for track in event], (-1, 3)) for event in tracker_input])

tracker_input_std = tracker_events_std.apply(concatenate_event, axis=1)
tracker_input_std = np.array([np.reshape([list(track) for track in event], (-1, 3)) for event in tracker_input_std])

#############################################################################################################
############ Check tracks
#############################################################################################################

# Number --- Check no longer valid, because some tracks are out of bounds and are rejected
# assert np.all(tracker_events["x_projections"].apply(len) == tracker_events["n_Particles"]), "Wrong track length for at least one event."
# assert np.all(np.array([len(event) for event in tracker_input]) == tracker_events["n_Particles"]), "Wrong track length for at least one event."

# K, pi, pi, mu
idx = 0
ref = (tracker_events["K_x_projection"][idx], tracker_events["K_y_projection"][idx], tracker_events["K_real_ET"][idx])
min_dist = np.Inf
for i, ev in enumerate(tracker_input[idx]):
    dist = np.linalg.norm(np.array(ref)-np.array(ev))
    if dist < min_dist:
        min_dist = dist
        min_id = i

print(min_dist)
print(tuple(np.round(ref, 4)))
print(tracker_input[idx][min_id])

def signal_out_of_event(event, signal):
    # Certain particle information of own leaf in root tree
    p = np.array([event["{}_x_projection".format(signal)], event["{}_y_projection".format(signal)], event["{}_real_ET".format(signal)]])
    event = concatenate_event(event)

    # Compare certain leaf to value in all particles
    best_idx = np.argmin([np.linalg.norm(np.array(track)-p) for track in event])
    return (best_idx, event[best_idx], event[best_idx]-p)

diff_df = pd.DataFrame()
diff_df["K"] = tracker_events.apply(signal_out_of_event, signal="K", axis=1).values
diff_df["Pi1"] = tracker_events.apply(signal_out_of_event, signal="Pi1", axis=1).values
diff_df["Pi2"] = tracker_events.apply(signal_out_of_event, signal="Pi2", axis=1).values
diff_df["Mu"] = tracker_events.apply(signal_out_of_event, signal="Mu", axis=1).values

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(16, 9), facecolor='w', edgecolor='k')
axs = np.ravel(axs)

counter = 0
for signal in ["K", "Pi1", "Pi2", "Mu"]:
    for var in range(3):
        values = diff_df.apply(lambda x: x[signal][2][var], axis=1)
        axs[counter].hist(values, bins=30)

        if signal == "K":
            axs[counter].set_title(["X projection", "Y projection", "Energy"][var])
        if var == 0:
            axs[counter].set_ylabel(signal)
        counter += 1

fig.suptitle("(Value from L0Calo_HCAL_xProjections) - (Value from L0Calo_HCAL_xProjection)")
plt.savefig(data_load_path+"/Consistency.png")



#############################################################################################################
############ Save
#############################################################################################################
with open(data_load_path+"/tracker_events.pickle", "wb") as f:
    pickle.dump(tracker_events, f)
with open(data_load_path+"/tracker_images.pickle", "wb") as f:
    pickle.dump(tracker_images, f)
with open(data_load_path+"/calo_events.pickle", "wb") as f:
    pickle.dump(calo_events, f)
with open(data_load_path+"/calo_images.pickle", "wb") as f:
    pickle.dump(calo_images, f)
with open(data_load_path+"/tracker_input.pickle", "wb") as f:
    pickle.dump(tracker_input_std, f)