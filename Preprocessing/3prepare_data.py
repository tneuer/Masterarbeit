#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-23 14:45:06
    # Description :
####################################################################################
"""
import pickle

import numpy as np
import pandas as pd
import initialization as init

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#############################################################################################################
############ Global variables
#############################################################################################################
erange = "PiplusMixedP"
path_loading = "../../Data/{}/TestingPurpose".format(erange)

np_seed = 20191223

path_saving = path_loading
image_scaling = True
image_shape = [52, 64]

if "Debug" in path_loading:
    validation_size = 500
    test_size = 500
elif "LargeSample" in path_loading:
    validation_size = 10000
    test_size = 8000
elif "Sample" in path_loading:
    validation_size = 5000
    test_size = 3000
elif "TestingPurpose" in path_loading:
    validation_size = 0
    test_size = 0
else:
    raise NotImplementedError("Wrong data sample selected.")


#############################################################################################################
############ Load data
#############################################################################################################
data = init.load_data(data_path=path_loading, mode="train")
tracker_events = data["tracker_events"]
calo_images = data["calo_images"]


#############################################################################################################
############ Split data
#############################################################################################################
available_indices = np.arange(len(tracker_events))
np.random.seed(np_seed)
np.random.shuffle(available_indices)

validation_idx = available_indices[:validation_size]
available_indices = available_indices[validation_size:]
assert len(validation_idx) == validation_size, "Wrong validation size."

test_idx = available_indices[:test_size]
available_indices = available_indices[test_size:]
assert len(test_idx) == test_size, "Wrong test size."

train_idx = available_indices

train_tracker, train_calo = tracker_events.iloc[train_idx], calo_images[train_idx]
val_tracker, val_calo = tracker_events.iloc[validation_idx], calo_images[validation_idx]
test_tracker, test_calo = tracker_events.iloc[test_idx], calo_images[test_idx]


#############################################################################################################
############ Preprocess data
#############################################################################################################


### Image scaling
calo_scaling = 6120
train_calo /= calo_scaling
val_calo /= calo_scaling
test_calo /= calo_scaling

### Tracker scaling

def standardize_data(data, scaler, exclude=None):
    standardized_data = data.drop(exclude, axis=1, inplace=False)
    colnames = standardized_data.columns.values
    standardized_data = pd.DataFrame(data=scaler.transform(standardized_data), columns=colnames, index=data.index)
    data = pd.concat([standardized_data, data[exclude]], axis=1, sort=False)
    return data

exclude = ["theta", "phi", "region"]
std_cols = train_tracker.columns.drop(exclude)
if "TestingPurpose" in path_loading:
    with open("../../Data/{}/LargeSample/ProcessedScaler.pickle".format(erange), "rb") as f:
        tracker_scaling = pickle.load(f)["Tracker"]
    train_tracker = standardize_data(data=train_tracker, scaler=tracker_scaling, exclude=exclude)
    is_not_outlier = (train_tracker["real_ET"]<4) | (train_tracker["momentum_p"]<4)
    train_tracker = train_tracker.loc[is_not_outlier, ]
    train_calo = train_calo[is_not_outlier, ]

    with open(path_saving+"/ProcessedTest.pickle", "wb") as f:
        pickle.dump({"Tracker": train_tracker, "Calo": train_calo, "Idx": train_idx}, f)
    with open(path_saving+"/ProcessedScaler.pickle", "wb") as f:
        pickle.dump({"Tracker": tracker_scaling, "Calo": calo_scaling, "Names": std_cols}, f)
    print(train_tracker.shape, train_calo.shape)

else:
    tracker_scaling = StandardScaler()
    tracker_scaling.fit(train_tracker.drop(exclude, axis=1, inplace=False))
    train_tracker = standardize_data(data=train_tracker, scaler=tracker_scaling, exclude=exclude)
    val_tracker = standardize_data(data=val_tracker, scaler=tracker_scaling, exclude=exclude)
    test_tracker = standardize_data(data=test_tracker, scaler=tracker_scaling, exclude=exclude)

    with open(path_saving+"/ProcessedTrain.pickle", "wb") as f:
        pickle.dump({"Tracker": train_tracker, "Calo": train_calo, "Idx": train_idx}, f, protocol=4)
    with open(path_saving+"/ProcessedValidation.pickle", "wb") as f:
        pickle.dump({"Tracker": val_tracker, "Calo": val_calo, "Idx": validation_idx}, f, protocol=4)
    with open(path_saving+"/ProcessedTest.pickle", "wb") as f:
        pickle.dump({"Tracker": test_tracker, "Calo": test_calo, "Idx": test_idx}, f, protocol=4)
    with open(path_saving+"/ProcessedScaler.pickle", "wb") as f:
        pickle.dump({"Tracker": tracker_scaling, "Calo": calo_scaling, "Names": std_cols}, f, protocol=4)
    print(train_tracker.shape, train_calo.shape)
    print(val_tracker.shape, val_calo.shape)
    print(test_tracker.shape, test_calo.shape)
