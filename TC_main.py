import glob2
import os
import numpy as np
import matplotlib.pyplot as plt
import random

from keras.utils import to_categorical
import keras

from TC_data import *
from TC_model import *
from TC_visualization import *

# global settings
trainnetwork = True
trainingsetsize = 0.8 # part of data to be used for training
validationsetsize = 0.2 # part of train set to be used for validation
dimensions = 2
cropdims = [144, 144]

nr_epochs = 2
minibatchsize = 1
networkpath_2D = r'Networks/network_2D_test.h5'

def main():
    paths = glob2.glob(r'Data/patient*/patient*_frame*.nii.gz')

    # figure out number of patients
    datasetsize = sum(os.path.isdir(os.path.join('Data', i)) for i in os.listdir('Data'))

    # split data paths, for training and testing
    trainsamples = int(np.ceil(datasetsize*trainingsetsize))
    train_paths = paths[:trainsamples*4] # now 320 paths
    test_paths = paths[trainsamples*4:] # now 80 paths

    # now load images for training set
    # loads as a tuple [ED_ims, ES_ims, gt_ED_ims, gt_ES_ims, spacings]
    ED_ims, ES_ims, gt_ED_ims, gt_ES_ims, spacings = loadImages(train_paths)

    # for training, the ES and ED images can be joined together
    train_ims = np.concatenate((ED_ims, ES_ims), axis=0) # (160,)
    gt_ims = np.concatenate((gt_ED_ims, gt_ES_ims), axis=0) # (160,)

    # resample to the same voxel spacings
    # first get the average x (and y) spacing of all images rounded to 1 decimal
    avgspacing = np.round(np.mean([spacings[i][0] for i in range(len(spacings))]), 1)

    # for i in range(10):
    #     plt.figure()
    #     plt.imshow(train_ims[i][5,:,:], cmap='gray')
    # plt.show()

    ########## needs to be done
    train_ims = resample(train_ims)

    # remove outliers from the images
    train_ims = removeOutliers(train_ims)

    # normalize images to [0,1] range
    train_ims = normalize(train_ims)

    if dimensions == 2:

        # crop images to same size (e.g. [144, 144], smallest image dimensions are x=154 and y=176)
        # first get centers of mass from the ground truth images
        CoM_ims = center_of_mass(gt_ims)

        # then reduce the x and y dimensions around the centers of mass
        train_ims = reduceDimensions(train_ims, cropdims, CoM_ims)
        gt_ims = reduceDimensions(gt_ims, cropdims, CoM_ims)

        # now dataset can be split into validation and training set
        # use vstack to get an array of the 2D slices
        valsamples = int(np.ceil(train_ims.shape[0]*validationsetsize))
        val_ims = np.vstack(train_ims[:valsamples])
        gt_val_ims = np.vstack(gt_ims[:valsamples])
        train_ims = np.vstack(train_ims[valsamples:])
        gt_train_ims = np.vstack(gt_ims[valsamples:])

        # need extra dimension for feature channels
        train_ims = np.expand_dims(train_ims, axis=3)
        val_ims = np.expand_dims(val_ims, axis=3)

        # one-hot encode the labels
        gt_train_ims = to_categorical(gt_train_ims)
        gt_val_ims = to_categorical(gt_val_ims)

        # initialize network
        unet_2D = get_unet_2D(cropdims[0], cropdims[1])
        unet_2D.summary()

        if trainnetwork:
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
            hist = unet_2D.fit(train_ims, gt_train_ims, batch_size=minibatchsize, epochs=nr_epochs, validation_data=(val_ims, gt_val_ims), callbacks=[early_stopping], verbose=1, shuffle=True)
            print(hist.history.keys())
            unet_2D.save(networkpath_2D)

        else:
            unet_2D = keras.models.load_model(networkpath_2D)

        visualizeTraining(hist)

        return

    return

main()
