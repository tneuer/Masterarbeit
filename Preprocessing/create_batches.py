#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-19 16:57:42
    # Description :
####################################################################################
"""
import os
import json
import pickle

import numpy as np
import initialization as init


############################################################################################################
# Parameter definiton
############################################################################################################
path_loading = "../../Data/Piplus/Debug"
path_saving = path_loading + "/Batches"
particle = "piplus"


padding = {"top": 6, "bottom": 6, "left":0, "right":0}
rotate = False
test_size = 500
test_seed = 42
logging_size = 10
logging_seed = 42
keep_cols = ["x_projections", "y_projections", "momentum_p", "momentum_px", "momentum_py", "momentum_pz"]
image_flatten = False
image_scaling = True

batch_size = 1000


############################################################################################################
# Data loading and initilization
############################################################################################################
if not os.path.exists(path_saving):
    os.mkdir(path_saving)
    os.mkdir(path_saving+"/BatchesX")
    os.mkdir(path_saving+"/BatchesY")

train_calo, train_tracker, test_calo, test_tracker, logging_calo, logging_tracker = (
        init.load_train_test_log(path_loading, path_saving, particle, padding, rotate,
                                 test_size, test_seed, logging_size, logging_seed, keep_cols, image_flatten, image_scaling)
)

train_test_log_max = [np.max(train_calo), np.max(test_calo), np.max(logging_calo)]
train_test_log_shape = [train_calo.shape, test_calo.shape, logging_calo.shape]
nr_batches = int(np.ceil(train_calo.shape[0]/batch_size))



############################################################################################################
# Create batches
############################################################################################################

calo_sizes = []
tracker_sizes = []
for batch in range(nr_batches):
    batch_calo = train_calo[batch*batch_size:(batch+1)*batch_size]
    batch_tracker = train_tracker[batch*batch_size:(batch+1)*batch_size]

    with open(path_saving+"/BatchesX/BatchX{}.pickle".format(batch), "wb") as f:
        pickle.dump(batch_calo, f)
    with open(path_saving+"/BatchesY/BatchY{}.pickle".format(batch), "wb") as f:
        pickle.dump(batch_tracker, f)

    calo_sizes.append([batch, batch_calo.shape[0]])
    tracker_sizes.append([batch, batch_tracker.shape[0]])


with open(path_saving+"/BatchX_Test.pickle", "wb") as f:
    pickle.dump(test_calo, f)
with open(path_saving+"/BatchY_Test.pickle", "wb") as f:
    pickle.dump(test_tracker, f)
with open(path_saving+"/BatchX_Logging.pickle", "wb") as f:
    pickle.dump(logging_calo, f)
with open(path_saving+"/BatchY_Logging.pickle", "wb") as f:
    pickle.dump(logging_tracker, f)


config_data = init.get_config_dict(globals())
with open(path_saving+"/config.json", "w") as f:
    json.dump(config_data, f, indent=4)