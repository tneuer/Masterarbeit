#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-12-08 22:45:25
    # Description :
####################################################################################
"""
import os
import re
import sys
sys.path.insert(1, "./building_blocks")
sys.path.insert(1, "../Utilities")
import time
import pickle

import numpy as np
import tensorflow as tf

from building_blocks.layers import reshape_layer, image_condition_concat
from building_blocks.networks import Critic, ConditionalGenerator
from building_blocks.generativeModels import ConditionalGenerativeModel
from functionsOnImages import padding_zeros

from CWGANGP_OO import CWGANGP


def natural_keys(text):
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


class BCWGANGP(CWGANGP):
    def train(self, batch_x_path, batch_y_path, x_test=None, y_test=None, epochs=100, batch_size=64,
              gen_steps=1, critic_steps=5, log_step=3, steps=None, gpu_options=None,
              preprocess_func=None, **kwargs):
        if steps is not None:
            gen_steps = 1
            critic_steps = steps
        self._set_up_training(log_step=log_step, gpu_options=gpu_options)
        batches_x = [batch_x_path+"/"+batchfile for batchfile in os.listdir(batch_x_path)]
        batches_x.sort(key=natural_keys)
        batches_y = [batch_y_path+"/"+batchfile for batchfile in os.listdir(batch_y_path)]
        batches_y.sort(key=natural_keys)

        assert len(batches_x)==len(batches_y), "BatchX and BatchY folder must have same amount of files. X: {}, Y: {}.".format(len(batches_x), len(batches_y))
        assert all([".pickle" in f for f in batches_x+batches_y]), "All batch files must be .pickle files."

        for epoch in range(epochs):
            for bn, (batch_x, batch_y) in enumerate(zip(batches_x, batches_y)):
                with open(batch_x, "rb") as f:
                    x_train = pickle.load(f)
                with open(batch_y, "rb") as f:
                    y_train = pickle.load(f)
                if preprocess_func is not None:
                    x_train, y_train = preprocess_func(x_train, y_train, **kwargs)

                self._set_up_test_train_sample(x_train, y_train, x_test, y_test)
                if epoch == 0 and bn == 0:
                    self._log_results(epoch=0, epoch_time=0)
                batch_nr = 0
                critic_loss_epoch = 0
                gen_loss_epoch = 0
                start = time.clock()
                trained_examples = 0
                print("Working on batch {} / {}...".format(bn+1, len(batches_x)))
                while trained_examples < len(x_train):
                    critic_loss_batch, gen_loss_batch = self._optimize(self._trainset, batch_size, critic_steps, gen_steps)
                    trained_examples += batch_size

                    critic_loss_epoch += critic_loss_batch
                    gen_loss_epoch += gen_loss_batch


            epoch_train_time = (time.clock() - start)/60
            critic_loss_epoch = np.round(critic_loss_epoch, 2)
            gen_loss_epoch = np.round(gen_loss_epoch, 2)

            print("Epoch {}: Critic: {} \n\t\t\tGenerator: {}.".format(epoch+1, critic_loss_epoch, gen_loss_epoch))

            if self._log_step is not None:
                self._log(epoch+1, epoch_train_time)




if __name__ == '__main__':
    path_loading = "../../Data/MNIST/Batches"

    with open(path_loading+"/BatchX_Logging.pickle", "rb") as f:
        x_log = pickle.load(f)
    with open(path_loading+"/BatchY_Logging.pickle", "rb") as f:
        y_log = pickle.load(f)
    with open(path_loading+"/BatchX_Test.pickle", "rb") as f:
        x_test = pickle.load(f)[:400]
    with open(path_loading+"/BatchY_Test.pickle", "rb") as f:
        y_test = pickle.load(f)[:400]


    gen_architecture = [
                        [tf.layers.dense, {"units": 4*4*512, "activation": tf.nn.relu}],
                        [reshape_layer, {"shape": [4, 4, 512]}],

                        [tf.layers.conv2d_transpose, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        # [tf.layers.batch_normalization, {}],
                        # [tf.layers.dropout, {}],

                        [tf.layers.conv2d_transpose, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

                        [tf.layers.conv2d_transpose, {"filters": 1, "kernel_size": 2, "strides": 2, "activation": tf.nn.sigmoid}]
                        ]
    critic_architecture = [
                        [tf.layers.conv2d, {"filters": 64, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        # [tf.layers.batch_normalization, {}],
                        # [tf.layers.dropout, {}],

                        [tf.layers.conv2d, {"filters": 128, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],

                        [tf.layers.conv2d, {"filters": 256, "kernel_size": 2, "strides": 2, "activation": tf.nn.relu}],
                        ]
    inpt_dim = [32, 32, 1]
    image_shape = inpt_dim
    batch_x_path = path_loading+"/BatchesX"
    batch_y_path = path_loading+"/BatchesY"
    gpu_options = None

    z_dim = 128
    label_dim = 10
    bcwgangp = BCWGANGP(x_dim=inpt_dim, y_dim=label_dim, z_dim=z_dim, last_layer_activation=tf.nn.sigmoid,
                  gen_architecture=gen_architecture, critic_architecture=critic_architecture,
                  folder="../../Results/Test/BCWGANGP", image_shape=image_shape, append_y_at_every_layer=False)
    print(bcwgangp.show_architecture())
    bcwgangp.log_architecture()
    bcwgangp.compile(logged_images=x_log, logged_labels=y_log)
    bcwgangp.train(batch_x_path, batch_y_path, x_test, y_test, epochs=100, critic_steps=5, gen_steps=1, log_step=3,
                  gpu_options=gpu_options)