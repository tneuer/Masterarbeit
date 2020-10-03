#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-26 18:19:38
    # Description :
####################################################################################
"""
import os
import re
import sys
import json
import shutil
import imageio
import natsort

import numpy as np
import pandas as pd

from tabulate import tabulate
from collections import OrderedDict

# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

from plotnine import *
from PyPDF2 import PdfFileMerger
from functionsOnImages import savefigs, get_layout


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_index(paths_to_outputs, variables, save_path, sort_by="Log", generell_info="", ignore=None):
    if isinstance(paths_to_outputs, str):
        paths_to_outputs = [paths_to_outputs]
    if ignore is None:
        ignore = []
    subfolders = [f.path for fpath in paths_to_outputs for f in os.scandir(fpath) if f.is_dir() ]
    info = []
    used_variables = ["Log", "Layers", "Epochs", "Type"]
    used_variables.extend(variables)

    for i, folder in enumerate(subfolders):
        if i % 10 == 0:
            print(i+1, "/", len(subfolders), ":", folder)
        folder_config = []

        try:
            with open(folder+"/config.json", "r") as conf_f, open(folder+"/architecture.json", "r") as arch_f:
                config_dict = json.load(conf_f)
                architecture = json.load(arch_f)

                try:
                    exitflag_name = [f.path.split("/")[-1] for f in os.scandir(folder) if not os.path.isdir(f) and "EXIT" in f.path][0]
                except IndexError:
                    exitflag_name = "1"

                if "1" in exitflag_name:
                    config_dict["Exit"] = "1"
                else:
                    config_dict["Exit"] = "0"

                f_path = folder.rsplit("/")[-1]
                origin = folder.rsplit("/")[-2]
                f_path = origin+"/"+f_path

                graph_index = [f for f in os.listdir("{}/TFGraphs".format(folder)) if "index" in f]
                # epoch = max([int(re.findall(r'\d+', idx)[0]) for idx in graph_index])

                try:
                    act = config_dict["activation"].split(".")[-1]
                except KeyError:
                    act = "Not Implemented"

                layers_per_net = ""
                type_per_net = ""
                for name, network in architecture.items():
                    nr_layers = len(network)
                    layers_per_net += "{}: {}\n".format(name, nr_layers)

                    network_type = "Dense"
                    dropout = "False"
                    batch_norm = "False"

                    try:
                        reshape_z = config_dict["reshape_z"]
                    except KeyError:
                        reshape_z = "none"

                    try:
                        rotate = config_dict["rotate"]
                    except KeyError:
                        rotate = "false"

                    for layer in network:
                        if "conv" in layer[0]:
                            network_type = "Conv"
                        if "drop" in layer[0]:
                            dropout = "True"
                        if "norm" in layer[0]:
                            batch_norm = "True"
                    type_per_net += "{}: {}\n".format(name, network_type)

                    config_dict["Log"] = f_path
                    config_dict["Layers"] = layers_per_net
                    # config_dict["Epochs"] = epoch
                    config_dict["Type"] = network_type

                this_dict = {}
                for var in used_variables:
                    try:
                        this_dict[var] = config_dict[var]
                        if var == "keep_cols":
                            this_dict[var] = "\n".join(config_dict[var])
                    except KeyError:
                        this_dict[var] = "Not Implemented"
                info.append(OrderedDict(this_dict))

        except FileNotFoundError:
            print("File not found for {}.".format(folder))

    info_df = pd.DataFrame(info)
    if sort_by == "Log":
        idx = natsort.index_natsorted(info_df["Log"].values)
        info_df = info_df.iloc[idx]
    else:
        info_df = info_df.sort_values(by=[sort_by])
    info_df.reset_index(drop=True, inplace=True)
    to_append = ""
    for col in info_df:
        nr_unique = info_df[col].astype(str).nunique()
        if nr_unique == 1:
            to_append += "\n{}: {}".format(col, info_df.loc[0, col])
            info_df.drop(col, axis=1, inplace=True)
    to_append += "\n" + generell_info
    tabulation = tabulate(info_df, tablefmt="grid", headers="keys")
    if save_path is not None:
        with open(save_path+"/index.table", "w") as f:
            f.write(tabulation)
            f.write("\n")
            f.write(to_append)
    return info_df


def create_image_summary(paths_to_outputs, image_folder, nr_images, save_path, ignore=None):
    if isinstance(paths_to_outputs, str):
        paths_to_outputs = [paths_to_outputs]
    if ignore is None:
        ignore = "__//##//"
    subfolders = [f.path+"/"+image_folder for fpath in paths_to_outputs for f in os.scandir(fpath) if f.is_dir() ]
    subfolders.sort(key=natural_keys)
    figs = []
    nrows, ncols = get_layout(n=nr_images)
    for i, folder in enumerate(subfolders):
        if i % 10 == 0:
            print(i+1, "/", len(subfolders), ":", folder)
        if ignore in folder:
            continue
        image_paths = [f.path for f in os.scandir(folder)]
        image_paths.sort(key=natural_keys)
        im_ids = (np.linspace(0, 1, nr_images) * (len(image_paths) - 1)).astype(int)

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
        fig.suptitle(folder)
        axs = np.ravel(axs)
        for ax_idx, ax in enumerate(axs):
            if im_ids[ax_idx] >= len(image_paths):
                break
            im_id = im_ids[ax_idx] if nr_images != 1 else -1
            current_image_path = image_paths[im_id]
            image = imageio.imread(current_image_path)
            im_name = current_image_path.partition(image_folder+"/")[2]
            ax.set_title("Image: {}".format(im_name))
            ax.imshow(image)
        figs.append(fig)
        plt.close("all")
        if (i+1) % 10 == 0:
            savefigs(figures=figs, save_path=save_path+"/samples_{}.pdf".format(i+1))
            plt.cla()
            plt.clf()
            for fig in figs:
                del fig
            figs = []
        elif (i+1) == len(subfolders):
            savefigs(figures=figs, save_path=save_path+"/samples_{}.pdf".format(i+1))

    merger = PdfFileMerger()
    pdfs = [f.path for f in os.scandir(save_path) if ".pdf" in f.path and "samples_" in f.path]
    pdfs.sort(key=natural_keys)
    for pdf in pdfs:
        merger.append(pdf)
        os.remove(pdf)
    merger.write(save_path+"/samples.pdf")
    merger.close()


def concatenate_images(paths_to_outputs, image_name, save_path):
    if isinstance(paths_to_outputs, str):
        paths_to_outputs = [paths_to_outputs]
    subfolders = [f.path for fpath in paths_to_outputs for f in os.scandir(fpath) if f.is_dir() ]
    subfolders.sort(key=natural_keys)
    figs = []
    for i, folder in enumerate(subfolders):
        if i % 10 == 0:
            print(i+1, "/", len(subfolders), ":", folder)
        image_path = [f.path for f in os.scandir(folder) if image_name in f.path]
        if len(image_path) == 0:
            print("{} does not contain image.".format(folder))
            continue
        else:
            assert len(image_path) == 1, "Ambiguous files in {}.".format(folder)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 16))
        image = imageio.imread(image_path[0])
        ax.imshow(image)
        ax.set_title(folder)
        figs.append(fig)

    savefigs(figures=figs, save_path=save_path+"{}.pdf".format(image_name))


def create_loss_figure(paths_to_outputs, save_path, ignore=None):
    if isinstance(paths_to_outputs, str):
        paths_to_outputs = [paths_to_outputs]
    subfolders = [f.path+"/TensorboardData" for fpath in paths_to_outputs for f in os.scandir(fpath) if f.is_dir() ]
    subfolders.sort(key=natural_keys)
    result_dict = {"Network": [], "Type": [], "Name": [], "Value": []}

    for i, folder in enumerate(subfolders):
        print(i+1, "/", len(subfolders), ":", folder)
        tf_paths = [f.path for f in os.scandir(folder)]
        tf_paths.sort(key=natural_keys)
        for tb_folder in tf_paths:
            tb_summary = [f.path for f in os.scandir(tb_folder)][-1]
            for summary in tf.train.summary_iterator(tb_summary):
                for statistic in summary.summary.value:
                    if "loss" in statistic.tag:
                        result_dict["Network"].append(re.findall("CycleGAN[0-9]+", folder)[0])
                        if "train" in tb_summary.lower():
                            result_dict["Type"].append("Train")
                        elif "test" in tb_summary.lower():
                            result_dict["Type"].append("Test")

                        result_dict["Name"].append(statistic.tag)
                        result_dict["Value"].append(statistic.simple_value)

    result_df = pd.DataFrame(result_dict)
    result_df["Step"] = 1
    id_list = []
    for i, row in result_df.iterrows():
        le_id = row["Network"]+row["Type"]+row["Name"]
        id_list.append(le_id)
        result_df.loc[i, "Step"] = id_list.count(le_id)
    plt1 = (
            ggplot(result_df, aes(x="Step", y="Value", color="Network", linetype="Type")) +
                geom_line() +
                facet_wrap(["Name"])
    )
    ggsave(plot=plt1, filename=save_path+"/Losses.png", device="png", height=12, width=12)


def create_statistical_summary(paths_to_outputs, subfolders, variables, save_path, pairwise=None):
    paths_to_outputs = [paths_to_outputs+"/"+s  for s in subfolders]
    summary_df = create_index(paths_to_outputs=paths_to_outputs, variables=variables, save_path=None)
    summary_df["Type"] = [re.findall("[a-zA-Z0-9_]+/", log)[0][:-1] for log in summary_df["Log"]]
    figs = []
    for col in summary_df:
        print("Processing: ", col)
        if col not in ["Log", "Type"]:
            current_df = summary_df[[col, "Type"]]
            aggregate_df = current_df.groupby([col, "Type"]).size().reset_index()
            aggregate_df.columns = [col, "Type", "Count"]

            wide_aggregate_df = aggregate_df.pivot(index='Type', columns=col, values='Count')
            wide_aggregate_df = wide_aggregate_df.reindex(index=subfolders)
            ax = wide_aggregate_df.plot.bar(rot=0)
            fig = plt.gcf()
            figs.append(fig)

    for pair in pairwise:
        print("Processing: ", pair)
        try:
            current_df = summary_df[[pair[0], pair[1], "Type"]]
        except KeyError as e:
            print("KeyError: {}".format(e))
            continue
        aggregate_df = current_df.groupby([pair[0], pair[1], "Type"]).size().reset_index()
        aggregate_df["Pair"] = aggregate_df[pair[0]].astype(str) + "-" + aggregate_df[pair[1]].astype(str)
        aggregate_df.drop(pair, inplace=True, axis=1)
        aggregate_df.columns = ["Type", "Count", "Pair"]
        wide_aggregate_df = aggregate_df.pivot(index='Type', columns="Pair", values='Count')
        wide_aggregate_df = wide_aggregate_df.reindex(index=subfolders)
        ax = wide_aggregate_df.plot.bar(rot=0)
        fig = plt.gcf()
        figs.append(fig)

    savefigs(figures=figs, save_path=save_path+"/statistics.pdf")
    return(figs)


def move_to_exit(paths_to_outputs, target_folder="4Exit"):
    if not paths_to_outputs.endswith("/"):
        paths_to_outputs += "/"
    if not target_folder.endswith("/"):
        target_folder += "/"
    target_folder = paths_to_outputs + target_folder
    summary_df = create_index(paths_to_outputs=paths_to_outputs, variables=["Exit"], save_path=None)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    nr_deleted = 0
    for idx in range(len(summary_df.index)):
        exitflag = summary_df.loc[idx, "Exit"]
        if "1" == exitflag:
            log_name = summary_df.loc[idx, "Log"].rsplit("/", 1)[-1]
            file_to_move = paths_to_outputs + log_name
            target_file = target_folder + log_name
            nr_deleted += 1
            shutil.move(file_to_move, target_file)

    print("{} / {} networks did not converge and were moved to {}.".format(nr_deleted, len(summary_df.index), target_folder))


if __name__ == "__main__":
    results_folder = "../../Results/ServerTemp/PiplusLowerPSummary/PiplusLowerP4/"
    results_folder = "../../Results/Test/B2Dmunu/"
    include_folders = [results_folder]
    subfolders = ["1Good", "2Okey", "3Bad", "4Exit"]

    # include_folders = [results_folder+subfolder for subfolder in subfolders]

    use_vars = ["Exit", "y_dim", "z_dim", "keep_cols", "architecture", "nr_params", "nr_gen_params", "nr_disc_params",
                "is_patchgan", "loss", "optimizer", "batch_size", "nr_train", "shuffle",
                "gen_steps", "adv_steps", "dataset", "algorithm", "dropout", "batchnorm", "label_smoothing", "invert_images",
                "feature_matching"]
    pairwise = [["feature_matching", "loss"], ["optimizer", "learning_rate"]]

    create_index(include_folders, variables=use_vars, save_path=results_folder, sort_by="Log")
    # create_image_summary(include_folders, image_folder="GeneratedSamples", nr_images=9, save_path=results_folder, ignore="4Exit")
    # concatenate_images(include_folders, image_name="TrainStatistics", save_path=results_folder)
    # move_to_exit(paths_to_outputs=results_folder)
    # create_statistical_summary(results_folder, subfolders, variables=use_vars, save_path=results_folder,
    #                            pairwise=pairwise)

