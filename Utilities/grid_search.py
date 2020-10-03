#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-05 13:16:51
    # Description :
####################################################################################
"""
import json
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


def run_agorithm(algorithm, init_params, compile_params, train_params, path_saving, config_data=None):
    try:
        network = algorithm(**init_params)
        print(network.show_architecture())
        for architecture in network._nets:
            print(architecture._name, architecture.get_number_params())
        network.log_architecture()
        network.compile(**compile_params)

        if config_data is not None:
            config_data["nr_params"] = network.get_number_params()
            config_data["nr_gen_params"] = network._generator.get_number_params()
            try:
                config_data["nr_adv_params"] = network._discriminator.get_number_params()
            except AttributeError:
                pass
            try:
                config_data["nr_adv_params"] = network._critic.get_number_params()
            except AttributeError:
                pass
            try:
                config_data["nr_adv_params"] = network._adversarial.get_number_params()
            except AttributeError:
                pass
            try:
                config_data["nr_enc_params"] = network._encoder.get_number_params()
            except AttributeError:
                pass
            config_data["sampler"] = network.get_sampling_distribution()
            if algorithm.__name__ == "CycleGAN":
                config_data.update({"generator_out": network._generator_xy._output_layer.name,
                                   "generator_out_yx": network._generator_yx._output_layer.name
                                   })
            elif algorithm.__name__ == "Pix2PixGAN":
                config_data.update({"generator_out": network._generator._output_layer.name})
            elif algorithm.__name__ == "CVAEGAN":
                config_data.update({"generator_out": network._generator._output_layer.name})
            elif algorithm.__name__ == "BiCycleGAN":
                config_data.update({"generator_out": network._generator._output_layer.name})
            with open(path_saving+"/config.json", "w") as f:
                json.dump(config_data, f, indent=4)
        network.train(**train_params)
        with open(path_saving+"/EXIT_FLAG0.txt", "w") as f:
            f.write("EXIT STATUS: 0. No errors or warnings.")
    except GeneratorExit as e:
        with open(path_saving+"/EXIT_FLAG1.txt", "w") as f:
            f.write("EXIT STATUS: 1. {}.".format(e))



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