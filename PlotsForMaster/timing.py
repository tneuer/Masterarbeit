#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-11-06 10:50:55
    # Description :
####################################################################################
"""
import os
import sys
import time
sys.path.insert(1, "../Utilities")
import pickle
# import GPUtil

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import defaultdict
from functionsOnImages import padding_zeros

from TrainedGenerator import TrainedGenerator
from TrainedIm2Im import TrainedIm2Im

def time_it(data, generator, batch_size, mode):
    gpu_name = "GeForce GTX 950M" if prefix == "Local" else "Tesla P100 PCIe"
    memory_used = tf.contrib.memory_stats.MaxBytesInUse()
    def get_empty_list():
        return([0 for i in range(nr_batches)])
    time_stats = defaultdict(get_empty_list)
    nr_batches = len(data) // batch_size

    total_time = 0
    total_examples = 0
    for i in range(nr_batches):
        print(i, "/", nr_batches)
        if i == 2:
            gpu_mem = generator._sess.run(memory_used) / 1024 / 1024
        batch_start = time.clock()
        start = i*batch_size
        stop = (i+1)*batch_size
        generated_images_cgan = generator.generate_batches(list_of_inputs=data[start:stop], batch_size=batch_size)
        batch_end = time.clock()
        batch_time = batch_end - batch_start
        total_time += batch_time
        total_examples += batch_size

        time_stats["batch"][i] = i
        time_stats["cum_batch"][i] = total_examples
        time_stats["batch_size"][i] = batch_size
        time_stats["batch_time"][i] = batch_time
        time_stats["cum_time"][i] = total_time

    savepath = "../../Results/Timing/Mode_{}".format(mode)
    for i in range(100):
        savepath_i = savepath + "_{}".format(i) + ".csv"
        if not os.path.exists(savepath_i):
            the_stats = pd.DataFrame(time_stats)
            the_stats["GPU"] = gpu_name
            the_stats["GPU_mem"] = gpu_mem
            the_stats["GPU_nr"] = nr_gpus
            the_stats.to_csv(savepath_i, index=False)
            break
    return(the_stats)

datasize = "LargeSample"
model_vec = "CWGANGP8"
model_im = "BiCycleGAN1"
data_path = "../../Data/B2Dmunu/{}".format(datasize)
batch_size = 50
if "lhcb_data" in os.getcwd():
    prefix = "Server"
else:
    prefix = "Local"

time_id = 0
nr_gpus = 1
gpu_fraction = 0.5
os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
if time_id == 0:
    model_path_vec = "../../Results/PiplusLowerP/{}/".format(model_vec)
    meta_path_vec = model_path_vec + "TFGraphs/"
    config_path_vec = model_path_vec + "config.json"
    with open(data_path+"/tracker_input.pickle", "rb") as f:
        tracker_input = pickle.load(f)
    Generator_vec = TrainedGenerator(path_to_meta=meta_path_vec, path_to_config=config_path_vec, gpu_options=gpu_options)
    time_it(data=tracker_input, generator=Generator_vec, batch_size=batch_size, mode=prefix+"Vector")
elif time_id == 1:
    model_path_im = "../../Results/B2Dmunu/{}/".format(model_im)
    meta_path_im = model_path_im + "TFGraphs/"
    config_path_im = model_path_im + "config.json"
    with open(data_path+"/Trained/PiplusLowerP_CWGANGP8_out_1.pickle", "rb") as f:
        cgan_images = np.clip(padding_zeros(pickle.load(f), top=4, bottom=4), a_min=0, a_max=6120).reshape([-1, 64, 64, 1]) / 6120
    Generator_im  = TrainedIm2Im(path_to_meta=meta_path_im, path_to_config=config_path_im, gpu_options=gpu_options)
    time_it(data=cgan_images, generator=Generator_im, batch_size=batch_size, mode=prefix+"Image")
elif time_id == 2:
    model_path_direct = "../../Results/ServerTemp/B2DmunuTracker/1Good/BiCycleGANTracker13/"
    meta_path_direct = model_path_direct + "TFGraphs/"
    config_path_direct = model_path_direct + "config.json"
    with open(data_path+"/tracker_images.pickle", "rb") as f:
        tracker_images = padding_zeros(pickle.load(f), top=6, bottom=6).reshape([-1, 64, 64, 1]) / 6120
    Generator_im_direct  = TrainedIm2Im(path_to_meta=meta_path_direct, path_to_config=config_path_direct, gpu_options=gpu_options)
    time_it(data=tracker_images, generator=Generator_im_direct, batch_size=batch_size, mode=prefix+"Direct")
else:
    raise ValueError("Wrong time_it value.")
