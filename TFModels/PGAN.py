#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-14 17:18:08
    # Description :
####################################################################################
"""
import os
import sys
sys.path.insert(1, "../Utilities")
sys.path.insert(1, "./building_blocks")
sys.path.insert(1, "./CGAN")
sys.path.insert(1, "./GAN")
sys.path.insert(1, "./CGAN/OLD")
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from CWGAN_OO import CWGAN
from CWGANGP_OO import CWGANGP
from CVAE_OO import CVAE
from CC_CWGAN1_OO import CC_CWGAN1
from CC_CWGAN2_OO import CC_CWGAN2
from InfoGAN_OO import InfoGAN
from CBiGAN_OO import CBiGAN
from CLSGAN_OO import CLSGAN
from CGAN import CGAN

from BCWGANGP_OO import BCWGANGP

from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, crop_images


def create_algorithm(algorithm, penalize_cells=False, **kwargs):

    def prepare_kwargs(keywords, possible_keywords, translation):
        delete_keys = get_pop_keywords(keywords, possible_keywords, translation)
        pop_keywords(delete_keys, kwargs)
        return keywords

    def get_pop_keywords(keywords, possible_keywords, translation):
        del_keys = []
        keys = list(keywords.keys())
        print("\nRenamed keys:")
        for key in keys:
            if key in translation:
                keywords[translation[key]] = keywords.pop(key)
                print("\t{} -> {}.".format(key, translation[key]))
            elif key not in possible_keywords:
                del_keys.append(key)

        return del_keys

    def pop_keywords(delete_keys, keywords):
        print("Deleted keys:")
        for key in delete_keys:
            keywords.pop(key)
            print("\t{}".format(key))


    necessary_keywords = ["x_dim", "y_dim", "z_dim", "last_layer_activation"]
    allowed_keywords = ["folder", "image_shape", "append_y_at_every_layer"]

    if algorithm in ["CC_CWGAN1", "CC_CWGAN2", "CC_CWGANGP"]:
        algorithm_specific_keywords = ["critic_architecture", "gen_architecture", "aux_architecture"]
        translation = {"adv_architecture": "critic_architecture"}

    elif algorithm in ["CWGANGP"]:
        algorithm_specific_keywords = ["critic_architecture", "gen_architecture", "PatchGAN"]
        translation = {"adv_architecture": "critic_architecture"}

    elif algorithm in ["CWGAN", "BCWGANGP", "CLSGAN"]:
        algorithm_specific_keywords = ["critic_architecture", "gen_architecture"]
        translation = {"adv_architecture": "critic_architecture"}

    elif algorithm in ["CVAE"]:
        algorithm_specific_keywords = ["dec_architecture", "enc_architecture"]
        translation = {"adv_architecture": "enc_architecture", "gen_architecture": "dec_architecture"}

    elif algorithm in ["CBiGAN"]:
        algorithm_specific_keywords = ["gen_architecture", "disc_architecture", "enc_architecture"]
        translation = {"adv_architecture": "disc_architecture", "aux_architecture": "enc_architecture"}

    elif algorithm in ["CGAN"]:
        algorithm_specific_keywords = ["adversarial_architecture", "gen_architecture", "is_patchgan", "is_wasserstein", "aux_architecture"]
        translation = {"adv_architecture": "adversarial_architecture"}

    # elif algorithm in ["InfoGAN"]:
    #     algorithm_specific_keywords = ["gen_architecture", "disc_architecture", "aux_architecture"]
    #     translation = {"adv_architecture": "disc_architecture"}
    else:
        raise NotImplementedError("Algorithm not available.")

    possible_keywords = necessary_keywords + allowed_keywords + algorithm_specific_keywords
    kwargs = prepare_kwargs(kwargs, possible_keywords, translation)

    class PGAN(eval(algorithm)):
        def __init__(self, **kwargs):
            tf.reset_default_graph()
            super(PGAN, self).__init__(**kwargs)
            self._maxEnergy = 6120
            self._best_loss = np.infty
            self._custom_saver = tf.train.Saver(max_to_keep=1)

            os.mkdir(self._folder+"/Evaluation/Cells")
            os.mkdir(self._folder+"/Evaluation/CenterOfMassX")
            os.mkdir(self._folder+"/Evaluation/CenterOfMassY")
            os.mkdir(self._folder+"/Evaluation/ConditionalSpace")
            os.mkdir(self._folder+"/Evaluation/Energy")
            os.mkdir(self._folder+"/Evaluation/MaxEnergy")
            os.mkdir(self._folder+"/Evaluation/StdEnergy")

            self._penalize_cells = penalize_cells


        def evaluate(self, true, condition, epoch):
            if self._penalize_cells:
                print(self._sess.run(self._lit_cells, feed_dict={self._Z_input: self._z_test, self._Y_input: self._y_test, self._is_training: True}))
            fake = self.generate_samples(inpt=condition)
            self.scan_conditional_space(condition, epoch)
            true = true.reshape([-1, self._image_shape[0], self._image_shape[1]])
            fake = fake.reshape([-1, self._image_shape[0], self._image_shape[1]])
            build_histogram(true=true, fake=fake, function=get_energies, name="Energy", epoch=epoch,
                            folder=self._folder, energy_scaler=self._maxEnergy)
            build_histogram(true=true, fake=fake, function=get_number_of_activated_cells, name="Cells", epoch=epoch,
                            folder=self._folder, threshold=5/self._maxEnergy)
            build_histogram(true=true, fake=fake, function=get_max_energy, name="MaxEnergy", epoch=epoch,
                            folder=self._folder, energy_scaler=self._maxEnergy)
            build_histogram(true=true, fake=fake, function=get_center_of_mass_x, name="CenterOfMassX", epoch=epoch,
                            folder=self._folder, image_shape=self._image_shape)
            build_histogram(true=true, fake=fake, function=get_center_of_mass_y, name="CenterOfMassY", epoch=epoch,
                            folder=self._folder, image_shape=self._image_shape)
            build_histogram(true=true, fake=fake, function=get_std_energy, name="StdEnergy", epoch=epoch,
                            folder=self._folder, energy_scaler=self._maxEnergy)
            # self.save_if_best(true=true, fake=fake, epoch=epoch)
            plt.close("all")


        def scan_conditional_space(self, condition, epoch):
            mins = np.min(condition, axis=0)
            maxs = np.max(condition, axis=0)
            means = np.mean(condition, axis=0)
            nr_attributes = len(mins)

            idx_xProjection = self.attributes=="x_projection"

            cond_space = []
            for i, attribute in enumerate(self.attributes):
                scanned_conditions = means.copy()
                if attribute not in ["x_projection", "y_projection"]:
                    scanned_conditions[idx_xProjection] = maxs[idx_xProjection] / 1.25
                scan_space = np.linspace(mins[i], maxs[i], 10)
                for scan in scan_space:
                    scanned_conditions[i] = scan
                    cond_space.append([*scanned_conditions.tolist()])

            cond_space = np.stack(cond_space, axis=0)
            self._generator.plot_samples_from_labels(inpt_y=cond_space, sess=self._sess, image_shape=self._image_shape, row_titles=self.attributes,
                                                     nrows=nr_attributes, epoch=epoch, path=self._folder+"/Evaluation/ConditionalSpace/Scan{}.png".format(epoch))


        def save_if_best(self, true, fake, epoch):
            l1 = np.mean(np.abs(get_energies(true) - get_energies(fake)))
            l2 = np.mean(np.abs(get_number_of_activated_cells(true, threshold=5/self._maxEnergy) -
                               get_number_of_activated_cells(fake, threshold=5/self._maxEnergy)))

            print(l1, l2)
            if l1+l2 < self._best_loss:
                self._custom_saver.save(self._sess, "{}/TFGraphs/Best".format(self._folder), global_step=epoch)
                print("Saved best! Loss: {} --> {}.".format(self._best_loss, l1+l2))
                self._best_loss = l1+l2


        def set_attributes(self, attributes):
            self.attributes = attributes


        if penalize_cells:
            def _define_loss(self):
                super(PGAN, self)._define_loss()
                # self._lit_cells = tf.reduce_sum(tf.cast(tf.greater(self._output_gen, 5/self._maxEnergy), tf.float32))/tf.cast(tf.shape(self._output_gen)[0], tf.float32)
                self._lit_cells = tf.reduce_mean(tf.cast(tf.greater(self._output_gen, 5/self._maxEnergy), tf.float32))
                self._gen_loss += self._lit_cells


    return PGAN(**kwargs)