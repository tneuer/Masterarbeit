#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-05 10:58:11
    # Description :
####################################################################################
"""
import os
import re

from shutil import rmtree
from distutils.dir_util import copy_tree

def combine_folders(sources, target, delete_old):
    if os.path.exists(target):
        raise FileExistsError("{} already exists.".format(target))
    os.mkdir(target)

    folders = [f.path for src in sources for f in os.scandir(src) if os.path.isdir(f.path)]
    path_translation = {}
    for folder_idx, folder in enumerate(folders):
        subfolder = folder.split("/")[-1]

        while subfolder in path_translation:
            subfolder_nr = re.findall(r'\d+', subfolder)[0]
            subfolder = subfolder.replace(subfolder_nr, "") + str(int(subfolder_nr)+1)
        path_translation[subfolder] = folder

    log = ""
    for idx, (new_name, old_path) in enumerate(path_translation.items()):
        new_path = target+"/"+new_name
        log_idx = "Copying {}/{}:   {} --->> {}".format(idx+1, len(path_translation), old_path, new_path)
        print(log_idx)
        log = "{}\n{}".format(log, log_idx)
        if os.path.isdir(old_path):
            copy_tree(old_path, new_path)
            if delete_old:
                print("Deleted.")
                rmtree(old_path)

    with open(target+"/Log.txt", "w") as f:
        f.write(log)

    if delete_old:
        [rmtree(src) for src in sources]
        print("Deleted sources: {}.".format(sources))


def clean_folders(target_folders, keep):
    rm_folders = [f.path for target_folder in target_folders for f in os.scandir(target_folder) if os.path.isdir(f.path)]
    for rm_folder in rm_folders:
        files_in_folder = [f.path for f in os.scandir(rm_folder)]
        for file_in_folder in files_in_folder:
            keep_it = False
            for potential_keep in keep:
                if potential_keep in file_in_folder:
                    keep_it = True
            if not keep_it:
                try:
                    rmtree(file_in_folder)
                except NotADirectoryError:
                    os.remove(file_in_folder)


if __name__ == "__main__":
    # for subfolder in ["1Good", "2Okey", "3Bad", "4Exit"]:
    #     source = "../../Results/ServerTemp/B2Dmunu/{}".format(subfolder)
    #     keep = ["config", "architecture", "architecture_details", "EXIT_FLAG", "TrainStatistics"]
    #     clean_folders(target_folders=[source], keep=keep)

    # for subfolder in ["1Good", "2Okey", "3Bad", "4Exit"]:
    #     source = "../../Results/ServerTemp/B2Dmunu/{}".format(subfolder)
    #     sources = [source, "../../Results/ServerTemp/B2DmunuHistory/{}".format(subfolder)]
    #     target = "../../Results/ServerTemp/B2DmunuHistory2/{}".format(subfolder)
    #     combine_folders(sources=sources, target=target, delete_old=False)





