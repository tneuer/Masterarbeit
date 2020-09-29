#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-05 13:16:51
    # Description :
####################################################################################
"""
import numpy as np

def get_parameter_grid(param_dict, n, allow_repetition=False):
    nr_possibilities = np.prod([len(opt) for opt in param_dict.values()])
    if not allow_repetition and n > nr_possibilities:
        raise ValueError("n ({}) must be smaller than the number of possibilities ({}) if allow_repetition is False.".format(n, nr_possibilities))

    grid = []
    rep = 0
    while len(grid) != n:
        gridpoint = {key: np.random.choice(param_dict[key]) for key in param_dict.keys()}
        if gridpoint in grid:
            if not allow_repetition:
                continue
            else:
                rep += 1
        grid.append(gridpoint)

    print("{} / {} unique gridpoints ({} total).".format(n-rep, nr_possibilities, n))
    return grid

if __name__ == "__main__":
    param_dict = {
                "z_dim": [32, 64, 128, 256],
                "activation": ["tf.nn.leaky_relu", "tf.nn.relu"],
                "append_y": [True, False],
                "keep_cols": [["x_projections", "y_projections", "momentum_p", "momentum_px", "momentum_py", "momentum_pz"],
                                ["x_projections", "y_projections", "momentum_p"]],
                "penalize_cells": [True, False],
                "add_last_conv": [True, False]
    }
    get_parameter_grid(param_dict=param_dict, n=100, allow_repetition=False)