#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2020-08-21 18:33:55
    # Description : Code more or less directly from:
        https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/
####################################################################################
"""
import os
if "lhcb_data2" in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.insert(1, "Preprocessing")
sys.path.insert(1, "Utilities")
import json
import pickle

import numpy as np
import tensorflow as tf
import Preprocessing.initialization as init
import json_to_keras as jtk
if "lhcb_data2" in os.getcwd():
    gpu_frac = 0.3
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
    set_session(tf.Session(config=config))
    print("1 GPU limited to {}% memory.".format(np.round(gpu_frac*100)))
else:
    gpu_options = None

from functionsOnImages import padding_zeros, build_images
from functionsOnImages import build_histogram, get_energies, get_max_energy, get_number_of_activated_cells
from functionsOnImages import get_center_of_mass_x, get_center_of_mass_y, get_std_energy, crop_images

import matplotlib.pyplot as plt

# example of defining a composite model for training the generator model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D, Conv2DTranspose, Concatenate
from keras.layers import Activation, LeakyReLU
from keras.layers import Dropout, BatchNormalization
from keras.utils.vis_utils import plot_model


def define_discriminator_from_json(image_shape, path_to_json):
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)
    merged = Concatenate()([in_src_image, in_target_image])
    patch_out = jtk.json_to_keras(inpt=merged, path_to_json=path_to_json, network="Discriminator")
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


def define_generator_from_json(image_shape, path_to_json):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    out_image = jtk.json_to_keras(inpt=in_image, path_to_json=path_to_json, network="Generator")
    model = Model(in_image, out_image)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    return model


def generate_real_samples(dataset, n_samples, patch_shape):
    """select a batch of random samples, returns images and target
    """
    trainA, trainB = dataset
    ix = np.random.randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    """generate a batch of images, returns images and targets
    """
    X = g_model.predict(samples)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    trainA, trainB = dataset
    batches_per_epo = int(len(trainA) / n_batch)
    n_steps = batches_per_epo * n_epochs
    for i in range(n_steps):
        n_patch = d_model.layers[-1].output_shape[1]
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

        if i % 5000 == 0:
            print('>%d / %d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, n_steps, d_loss1, d_loss2, g_loss))
            nr_eval = 300
            im_shape = [64, 64]
            samples_y = g_model.predict(test_x[:nr_eval]).reshape([-1, *im_shape])
            samples_x = test_x[:nr_eval].reshape([-1, *im_shape])

            array_of_images = np.array([
                [
                    tx.reshape(image_shape[0:2]),
                    ty.reshape(image_shape[0:2]),
                    gx.reshape(image_shape[0:2])
                ] for tx, ty, gx in zip(test_x[:10], test_y[:10], samples_y[:10])
            ])
            fig, ax = build_images(array_of_images, column_titles=["X", "Y", "G(X)"], fs_x=45, fs_y=25)
            plt.savefig("{}/GeneratedSamples/Samples_{}.png".format(path_saving, i))
            build_histogram(true=samples_x, fake=samples_y, function=get_energies, name="Energy", epoch=i,
                            folder=path_saving, energy_scaler=energy_scaler)
            build_histogram(true=samples_x, fake=samples_y, function=get_number_of_activated_cells, name="Cells", epoch=i,
                            folder=path_saving, threshold=5/energy_scaler)
            build_histogram(true=samples_x, fake=samples_y, function=get_max_energy, name="MaxEnergy", epoch=i,
                            folder=path_saving, energy_scaler=energy_scaler)
            build_histogram(true=samples_x, fake=samples_y, function=get_center_of_mass_x, name="CenterOfMassX", epoch=i,
                            folder=path_saving, image_shape=im_shape)
            build_histogram(true=samples_x, fake=samples_y, function=get_center_of_mass_y, name="CenterOfMassY", epoch=i,
                            folder=path_saving, image_shape=im_shape)
            build_histogram(true=samples_x, fake=samples_y, function=get_std_energy, name="StdEnergy", epoch=i,
                            folder=path_saving, energy_scaler=energy_scaler)

            save_files = os.listdir(path_saving+"/ModelSave")
            for f in save_files:
                os.remove(path_saving+"/ModelSave/"+f)
            with open(path_saving+"/ModelSave/Generator_{}.pickle".format(i), "wb") as f:
                pickle.dump(g_model, f)
            with open(path_saving+"/ModelSave/Discriminator_{}.pickle".format(i), "wb") as f:
                pickle.dump(d_model, f)
            with open(path_saving+"/ModelSave/Model_{}.pickle".format(i), "wb") as f:
                pickle.dump(gan_model, f)



if __name__ == "__main__":
    ############################################################################################################
    # Parameter definiton
    ############################################################################################################

    if "lhcb_data2" in os.getcwd():
        path_loading = "../../Data/B2Dmunu/LargeSample"
        path_results = "../../Results/B2Dmunu"
    else:
        path_loading = "../../Data/B2Dmunu/Debug"
        path_results = "../../Results/Test/B2Dmunu"
    algorithm = "Pix2Pix"
    padding = {"top": 4, "bottom": 4, "left":0, "right":0}

    if not os.path.exists(path_results):
        os.mkdir(path_results)

    path_saving = init.initialize_folder(algorithm=algorithm+"Keras", base_folder=path_results)
    os.mkdir(path_saving+"/Evaluation/Cells")
    os.mkdir(path_saving+"/Evaluation/CenterOfMassX")
    os.mkdir(path_saving+"/Evaluation/CenterOfMassY")
    os.mkdir(path_saving+"/Evaluation/Energy")
    os.mkdir(path_saving+"/Evaluation/MaxEnergy")
    os.mkdir(path_saving+"/Evaluation/StdEnergy")
    os.mkdir(path_saving+"/ModelSave")

    ############################################################################################################
    # Data loading
    ############################################################################################################
    with open("{}/Trained/PiplusLowerP_CWGANGP8_out_1.pickle".format(path_loading), "rb") as f:
        train_x = pickle.load(f)
        train_x = padding_zeros(train_x, **padding)

    with open("{}/calo_images.pickle".format(path_loading), "rb") as f:
        train_y = pickle.load(f)
        train_y = padding_zeros(train_y, top=6, bottom=6, left=0, right=0).reshape([-1, 64, 64, 1])

    energy_scaler = np.max(train_y)
    train_x = np.clip(train_x, a_min=0, a_max=energy_scaler)
    train_x /= energy_scaler
    train_y /= energy_scaler

    nr_test = int(min(0.1*len(train_x), 500))
    test_x = train_x[-nr_test:]
    train_x = train_x[:-nr_test]
    test_y = train_y[-nr_test:]
    train_y = train_y[:-nr_test]

    nr_train = train_x.shape[0]

    print(train_x.shape)
    print(train_y.shape)
    print(np.max(train_x))
    print(np.max(train_y))
    print(energy_scaler)

    image_shape = train_x.shape[1:]

    ############################################################################################################
    # Model defintion
    ############################################################################################################
    # define the models
    path_to_json = "../../Architectures/Pix2Pix/dense.json"
    d_model = define_discriminator_from_json(image_shape, path_to_json=path_to_json)
    g_model = define_generator_from_json(image_shape, path_to_json=path_to_json)

    # define the composite model
    gan_model = define_gan(g_model, d_model, image_shape)
    # summarize the model
    sys_console = sys.stdout
    sys.stdout = open('{}/d_model.txt'.format(path_saving), 'w')

    d_model.summary()
    print("\n\n\n")
    g_model.summary()
    print("\n\n\n")
    gan_model.summary()

    sys.stdout = sys_console

    # plot the model
    plot_model(d_model, to_file='{}/d_model_plot.png'.format(path_saving), show_shapes=True, show_layer_names=True)
    plot_model(g_model, to_file='{}/g_model_plot.png'.format(path_saving), show_shapes=True, show_layer_names=True)
    plot_model(gan_model, to_file='{}/gan_model_plot.png'.format(path_saving), show_shapes=True, show_layer_names=True)
    dataset = (train_x, train_y)
    train(d_model=d_model, g_model=g_model, gan_model=gan_model, dataset=dataset,
          n_batch=1, n_epochs=40)