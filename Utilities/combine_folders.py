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

    folders = [src+"/"+f for src in sources for f in os.listdir(src)]
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

if __name__ == "__main__":
    pass
    # source_folder = "../../Results"
    # to_combine = [source_folder+loc for loc in ["/Server", "/ServerTemp"]]
    # target = source_folder+"/Server2"
    # combine_folders(sources=to_combine, target=target, delete_old=True)