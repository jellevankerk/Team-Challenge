import glob2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from contextlib import redirect_stdout

from keras.utils import to_categorical
import keras

from TC_data import *
from TC_model import *
from TC_visualization import *
from TC_test import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # global settings
    trainnetwork = False
    evaluatenetwork = True

    # networkpath only used when trainnetwork = False but evaluatenetwork = True
    networkpath = r'Networks/network_2D_epochs=50_bs=1_channels=64-512'

    trainingsetsize = 0.8 # part of data to be used for training
    validationsetsize = 0.2 # part of train set to be used for validation
    dimensions = 2
    cropdims = [144, 144]

    num_epochs = 1000
    batchsize = 1

    # make list of all patients
    patients = os.listdir('Data')

    # split data for training and testing
    trainsamples = int(np.ceil(len(patients)*trainingsetsize))
    train_patients = patients[:trainsamples]
    test_patients = patients[trainsamples:]

    # now load images for all patients of the training set
    ED_ims, ES_ims, gt_ED_ims, gt_ES_ims, spacings = [], [], [], [], []
    for i in range(len(train_patients)):
        patient = train_patients[i]
        ED_im, ES_im, gt_ED_im, gt_ES_im, spacing = loadImages(patient)

        ED_ims.append(ED_im)
        ES_ims.append(ES_im)
        gt_ED_ims.append(gt_ED_im)
        gt_ES_ims.append(gt_ES_im)
        spacings.append(spacing)

    # make np array of the lists of arrays
    ED_ims = np.asarray(ED_ims)
    ES_ims = np.asarray(ES_ims)
    gt_ED_ims = np.asarray(gt_ED_ims)
    gt_ES_ims = np.asarray(gt_ES_ims)
    spacings = np.asarray(spacings)

    # resample to the same voxel spacings
    ED_ims = resample(ED_ims, spacings)
    ES_ims = resample(ES_ims, spacings)

    # for training, the ES and ED images can be joined together
    train_ims = np.concatenate((ED_ims, ES_ims), axis=0) # (160,)
    gt_ims = np.concatenate((gt_ED_ims, gt_ES_ims), axis=0) # (160,)

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

        # these are now arrays of 3D images
        val_ims = train_ims[:valsamples]
        gt_val_ims = gt_ims[:valsamples]
        train_ims = train_ims[valsamples:]
        gt_train_ims = gt_ims[valsamples:]

        # make arrays of 2D images for training on 2D images
        train_ims = create2DArray(train_ims)
        val_ims = create2DArray(val_ims)
        gt_train_ims = create2DArray(gt_train_ims)
        gt_val_ims = create2DArray(gt_val_ims)

        # need extra dimension for feature channels
        train_ims = np.expand_dims(train_ims, axis=3)
        val_ims = np.expand_dims(val_ims, axis=3)

        # one-hot encode the labels
        gt_train_ims = to_categorical(gt_train_ims, num_classes=4)
        gt_val_ims = to_categorical(gt_val_ims, num_classes=4)

        # initialize network
        unet_2D = get_unet_2D(cropdims[0], cropdims[1])
        unet_2D.summary()

        if trainnetwork:
            # get nr of channels of model to put in network filename
            nr_channels_start = unet_2D.layers[1].output_shape[3]
            nr_channels_end = unet_2D.layers[12].output_shape[3] # DIT AANPASSEN
            networkpath = r'Networks/network_{}D_epochs={}_bs={}_channels={}-{}'.format(dimensions, num_epochs, batchsize, nr_channels_start, nr_channels_end)

            # write model summary to file
            with open(r'{}_model.txt'.format(networkpath), 'w') as f:
                with redirect_stdout(f):
                    unet_2D.summary()

            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
            csv_logger = keras.callbacks.CSVLogger(r'{}_log.csv'.format(networkpath), separator='|', append=False)

            # train and save model
            hist = unet_2D.fit(train_ims, gt_train_ims, batch_size=batchsize, epochs=num_epochs, validation_data=(val_ims, gt_val_ims), callbacks=[csv_logger], verbose=1, shuffle=True)
            unet_2D.save(r'{}.h5'.format(networkpath))

            # plot loss and multilabel software of the training
            visualizeTraining(hist, dimensions, num_epochs, batchsize, nr_channels_start, nr_channels_end)

        # evaluate the network on test data
        if evaluatenetwork:

            #load network
            keras.backend.clear_session()
            unet_2D = keras.models.load_model(r'{}.h5'.format(networkpath), custom_objects={'softdice_coef_multilabel': softdice_coef_multilabel, 'softdice_multilabel_loss': softdice_multilabel_loss, 'tversky_loss': tversky_loss})

            print("Preparing test data...")
            test_ED_ims, test_ES_ims, test_gt_ED_ims, test_gt_ES_ims, test_spacings = [], [], [], [], []
            # do for every patient
            for i in range(len(test_patients)):
                patient = test_patients[i]
                test_ED_im, test_ES_im, test_gt_ED_im, test_gt_ES_im, test_spacing = loadImages(patient)

                test_ED_ims.append(test_ED_im)
                test_ES_ims.append(test_ES_im)
                test_gt_ED_ims.append(test_gt_ED_im)
                test_gt_ES_ims.append(test_gt_ES_im)
                test_spacings.append(test_spacing)

            # make np array of the lists of arrays
            test_ED_ims = np.asarray(test_ED_ims)
            test_ES_ims = np.asarray(test_ES_ims)
            test_gt_ED_ims = np.asarray(test_gt_ED_ims)
            test_gt_ES_ims = np.asarray(test_gt_ES_ims)
            test_spacings = np.asarray(test_spacings)

            # normalize the images
            test_ED_ims = normalize(test_ED_ims)
            test_ES_ims = normalize(test_ES_ims)

            CoM_test_ED_ims = center_of_mass(test_gt_ED_ims)
            test_ED_ims = reduceDimensions(test_ED_ims, cropdims, CoM_test_ED_ims)
            test_gt_ED_ims = reduceDimensions(test_gt_ED_ims, cropdims, CoM_test_ED_ims)

            CoM_test_ES_ims = center_of_mass(test_gt_ES_ims)
            test_ES_ims = reduceDimensions(test_ES_ims, cropdims, CoM_test_ES_ims)
            test_gt_ES_ims = reduceDimensions(test_gt_ES_ims, cropdims, CoM_test_ES_ims)

            # save shape for reconstruction of 3D images later
            test_ED_shape = copy.deepcopy(test_ED_ims)
            test_ES_shape = copy.deepcopy(test_ES_ims)
            test_gt_ED_shape = copy.deepcopy(test_gt_ED_ims)
            test_gt_ES_shape = copy.deepcopy(test_gt_ES_ims)

            test_ED_ims = create2DArray(test_ED_ims)
            test_ES_ims = create2DArray(test_ES_ims)
            test_gt_ED_ims = create2DArray(test_gt_ED_ims)
            test_gt_ES_ims = create2DArray(test_gt_ES_ims)

            test_ED_ims = np.expand_dims(test_ED_ims, axis=3)
            test_ES_ims = np.expand_dims(test_ES_ims, axis=3)

            test_gt_ED_ims = to_categorical(test_gt_ED_ims, num_classes=4)
            test_gt_ES_ims = to_categorical(test_gt_ES_ims, num_classes=4)
            print("Test data prepared")

            # make predictions for the test images
            print("Making segmentations...")
            pred_ED = unet_2D.predict(test_ED_ims, batch_size=batchsize, verbose=0)
            pred_ES = unet_2D.predict(test_ES_ims, batch_size=batchsize, verbose=0)

            # make the segmentations according to the probabilities
            # every class label with probability >= 0.5 gets value of class label (0, 1, 2 or 3)
            for i in range(pred_ED.shape[0]):
                image = pred_ED[i,:,:,:]
                for label in range(image.shape[2]):
                    im = image[:,:,label]
                    im = np.where(im >= 0.5, 1, 0)
                    image[:,:,label] = im
                    pred_ED[i,:,:,:] = image

            for i in range(pred_ES.shape[0]):
                image = pred_ES[i,:,:,:]
                for label in range(image.shape[2]):
                    im = image[:,:,label]
                    im = np.where(im >= 0.5, 1, 0)
                    image[:,:,label] = im
                    pred_ES[i,:,:,:] = image

            # reconstruct to 3D images in order to be able to calculate the EF
            ED_images_3D = np.empty_like(test_ED_shape)
            gt_ED_images_3D = np.empty_like(test_gt_ED_shape)
            for i in range(len(test_ED_shape)):
                # use saved shape to determine how many slices each 3D image should have
                slices = int(test_ED_shape[i].shape[0])
                image_3D = pred_ED[i:(i+slices),:,:,:]
                gt_image_3D = test_gt_ED_ims[i:(i+slices),:,:,:]
                ED_images_3D[i] = image_3D
                gt_ED_images_3D[i] = gt_image_3D

            ES_images_3D = np.empty_like(test_ES_shape)
            gt_ES_images_3D = np.empty_like(test_gt_ES_shape)
            for i in range(len(test_ES_shape)):
                slices = int(test_ES_shape[i].shape[0])
                image_3D = pred_ES[i:(i+slices),:,:,:]
                gt_image_3D = test_gt_ES_ims[i:(i+slices),:,:,:]
                ES_images_3D[i] = image_3D
                gt_ES_images_3D[i] = gt_image_3D
            print("Segmentations made")

            # calculate various results and save to text file
            # only do this if this hasn't been done yet
            if not os.path.isfile(r'{}_results.txt'.format(networkpath)):
                with open(r'{}_results.txt'.format(networkpath), 'w') as text_file:
                    # calculate various softdices and multiclass softdices and write to text file
                    print("Saving results...")
                    saveResults(ED_images_3D, gt_ED_images_3D, text_file, type="ED")
                    saveResults(ES_images_3D, gt_ES_images_3D, text_file, type="ES")

                    # also write resulting EF for both the obtained segmentation as the ground truth to same file
                    for patient in range(len(ED_images_3D)):
                        strokevolume_gt, LVEF_gt = calculateEF(gt_ED_images_3D, gt_ES_images_3D, test_spacings, patient)
                        strokevolume, LVEF = calculateEF(ED_images_3D, ES_images_3D, test_spacings, patient)
                        print('LV stroke volume is {:.2f} ml and ejection fraction is {:.2f}% for patient {}'.format(strokevolume_gt*0.001, LVEF_gt, patient), file=text_file)
                        print('LV stroke volume for ground truth is {:.2f} ml and ejection fraction is {:.2f}% for patient {} \n'.format(strokevolume*0.001, LVEF, patient), file=text_file)
                    print("Results saved")
                text_file.close()

            # see if folder to save plots exist, else make it
            if not os.path.isdir(networkpath):
                os.mkdir(networkpath)

            # plot and save 3D images for all patients in test set, including overlay of segmentation
            print("Saving segmentation images...")
            for i in range(len(ED_images_3D)):
                savepath = r'{}/test_ED_im_{}.png'.format(networkpath, i)
                visualize3Dimage(ED_images_3D[i], test_ED_shape[i], savepath)
            for i in range(len(ES_images_3D)):
                savepath = r'{}/test_ES_im_{}.png'.format(networkpath, i)
                visualize3Dimage(ES_images_3D[i], test_ES_shape[i], savepath)
            print("Segmentation images saved")

        return
    return

main()
