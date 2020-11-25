#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-08-20 22:14:52
    # Description :
####################################################################################
"""

import re
import os
import cv2

path = "/home/tneuer/Backup/Uni/Masterarbeit/Programs/PlotsForMaster/Results/wasserstein_patchTrue_fmFalse1/GeneratedSamples"

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text) ]

def make_video_of(folder, outname):
    image_folder = folder
    video_name = outname

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=natural_keys)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 10, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # cv2.destroyAllWindows()
    video.release()

make_video_of(path, path+"/process.avi")
