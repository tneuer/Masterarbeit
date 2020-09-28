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

method = "CVAE_log"

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

    cv2.destroyAllWindows()
    video.release()

if "VAE" in method:
    make_video_of("../Autoencoders/Tensorflow/{}/Spectrum".format(method), '../Autoencoders/Tensorflow/{}/spectrum.avi'.format(method))
    make_video_of("../Autoencoders/Tensorflow/{}/Cluster".format(method), '../Autoencoders/Tensorflow/{}/cluster.avi'.format(method))
elif "GAN" in method:
    make_video_of("../GANs/Tensorflow/{}/Spectrum".format(method), '../GANs/Tensorflow/{}/spectrum.avi'.format(method))
    make_video_of("../GANs/Tensorflow/{}/Cluster".format(method), '../GANs/Tensorflow/{}/cluster.avi'.format(method))
else:
    print("No valid log")
# make_video_of("../GANs/{}/Cluster".format(method), './Tensorflow/{}/cluster.avi'.format(method))