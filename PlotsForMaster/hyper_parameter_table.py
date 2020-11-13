#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-11-06 18:19:18
    # Description :
####################################################################################
"""

import os

params = {
    "Dim_z": [4, 8, 16, 32, 64, 128], "PatchGAN": [True, False], "Loss": ["JS", "KL", "Wasserstein"],
    "LR Generator": [0.001, 0.0005, 0.0001, 0.00005, 0.00001], "LR Discriminator": [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000001],
    "Label smoothing": [0.8, 0.9, 0.95], "Optimizer": ["RMSProp", "AdamOptimizer"], "Feature matching": [True, False],
    "lambda X": [0, 0.5, 1, 5, 10], "lambda Z": [0, 0.5, 1, 5, 10], "lambda VAE": [0, 0.5, 1, 5, 10],
    "Random labels": [0, 0.05, 0.1], "Batch size": [1, 4, 8, 16, 32, 64], "Adversarial steps": [1, 5, 10],
}

keys = sorted(params.keys())
max_key_length = max([len(key) for key in keys]) + 1
max_options_length = max([len(str(options)) for _, options in params.items()]) + 1

result_table = "|" + "Parameter".ljust(max_key_length) + "|" + "Options".ljust(max_options_length) + "|" + "#Options".ljust(9) + "|" + "Product".ljust(11) + "|"
total_params = 1
for key in keys:
    options = params[key]
    total_params *= len(options)
    new_line = (
        "\n"+"-"*88+"\n|"+ key.ljust(max_key_length) + "|" + str(options).ljust(max_options_length) +
        "|" + str(len(options)).ljust(9) + "|" + str(total_params).ljust(11) + "|"
    )
    result_table += new_line
result_table += "\n"+"-"*88

with open("../../Thesis/figures/Results/hyperparameters.txt", "w") as f:
    f.write(result_table)


