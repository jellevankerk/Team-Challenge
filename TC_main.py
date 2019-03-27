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
from TC_test import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # global settings
    trainnetwork = False
    evaluatenetwork = True

    # networkpath only used when trainnetwork = False but evaluatenetwork = True
    networkpath = r'Networks/network_2D_epochs=1000_bs=1_channels=64-512.h5'

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

        # get weightings according to occurrence of classes
        #weightings = get_weightings(gt_ims)

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

        # print(train_ims.shape) #(160,)

        # shuffle the dataset for training
        # same seed so ground truth is shuffled the same way
        # seed=1
        # train_ims = shuffleArray(train_ims, seed)
        # gt_train_ims = shuffleArray(gt_train_ims, seed)
        # val_ims = shuffleArray(val_ims, seed)
        # gt_val_ims = shuffleArray(gt_val_ims, seed)

        # only take certain class for training
        # label = 3
        # gt_train_ims = np.where(gt_train_ims == label, 1, 0)

        # print(np.unique(gt_train_ims))
        # print(gt_train_ims.shape) #(1224,144,144)
        #
        # plt.imshow(gt_train_ims[6][:,:])
        # plt.show()

        # need extra dimension for feature channels
        train_ims = np.expand_dims(train_ims, axis=3)
        val_ims = np.expand_dims(val_ims, axis=3)

        # one-hot encode the labels
        gt_train_ims = to_categorical(gt_train_ims, num_classes=4)
        gt_val_ims = to_categorical(gt_val_ims, num_classes=4)

        # initialize network
        unet_2D = get_unet_2D(cropdims[0], cropdims[1])
        # unet_2D = get_unet_2D(None, None)
        unet_2D.summary()

        # num_train_images = len(train_ims)
        # num_val_images = len(val_ims)

        if trainnetwork:
            # get nr of channels of model to put in network filename
            nr_channels_start = unet_2D.layers[1].output_shape[3]
            nr_channels_end = unet_2D.layers[12].output_shape[3]
            networkpath = r'Networks/network_{}D_epochs={}_bs={}_channels={}-{}.h5'.format(dimensions, num_epochs, batchsize, nr_channels_start, nr_channels_end)
            print(networkpath)

            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10000, verbose=0, mode='min')
            hist = unet_2D.fit(train_ims, gt_train_ims, batch_size=batchsize, epochs=num_epochs, validation_data=(val_ims, gt_val_ims), verbose=1, shuffle=True)

            # training_gen = image_generator(train_ims, gt_train_ims, batchsize, mode="train", aug=None)
            # valid_gen = image_generator(val_ims, gt_val_ims, batchsize, mode="train", aug=None)
            #
            # hist = unet_2D.fit_generator(
            #     training_gen,
            #     steps_per_epoch=num_train_images//batchsize,
            #     validation_data=valid_gen,
            #     validation_steps=num_val_images//batchsize,
            #     epochs=num_epochs)


            print(hist.history.keys())
            unet_2D.save(networkpath)

            visualizeTraining(hist, dimensions, num_epochs, batchsize, nr_channels_start, nr_channels_end)

        # evaluate the network on test data
        if evaluatenetwork:

            #load network
            keras.backend.clear_session()
            unet_2D = keras.models.load_model(networkpath, custom_objects={'softdice_coef_multilabel': softdice_coef_multilabel, 'softdice_multilabel_loss': softdice_multilabel_loss, 'tversky_loss': tversky_loss})

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

            # # save shape for reconstruction of 3D images later
            # test_ED_shape = copy.deepcopy(test_ED_ims)
            # test_ES_shape = copy.deepcopy(test_ES_ims)
            # test_gt_ED_shape = copy.deepcopy(test_gt_ED_ims)
            # test_gt_ES_shape = copy.deepcopy(test_gt_ES_ims)

            # test_ED_ims, test_ES_ims, test_gt_ED_ims, test_gt_ES_ims, test_spacings = prepareTestData(test_ED_ims, test_ES_ims, test_gt_ED_ims, test_gt_ES_ims, test_spacings, cropdims)


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

            print(test_gt_ES_ims.shape)
            for i in range(5):
                im = test_gt_ES_ims[i,:,:,:]
                for label in range(im.shape[2]):
                    image = im[:,:,label]
                    print(np.unique(image))
                    plt.figure()
                    plt.imshow(image[:,:])

            plt.show()

            # make predictions for the test images
            pred_ED = unet_2D.predict(test_ED_ims, batch_size=batchsize, verbose=1)
            pred_ES = unet_2D.predict(test_ES_ims, batch_size=batchsize, verbose=1)

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

            print(test_ED_shape.shape)
            print(test_gt_ED_shape.shape)
            print(test_ES_shape.shape)
            print(test_gt_ES_shape.shape)

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

            print(ED_images_3D.shape) #(20,) met elk (slices, 144, 144, 4)
            print(ES_images_3D.shape)
            print(gt_ED_images_3D.shape)
            print(gt_ES_images_3D.shape)

            asd

            # for i in range(5):
            #     im = gt_ES_images_3D[i]
            #     for label in range(im.shape[3]):
            #         image = im[:,:,:,label]
            #         print(np.unique(image))
            #         plt.figure()
            #         plt.imshow(image[5,:,:])
            #
            # plt.show()


            #tijdelijk
            # images_3D = np.concatenate((ED_images_3D, ES_images_3D), axis=0)
            # gt_images_3D = np.concatenate((gt_ED_images_3D, gt_ES_images_3D), axis=0)
            # ED_images_3D = images_3D
            # gt_ED_images_3D = gt_images_3D

            softdicelist0, softdicelist1, softdicelist2, softdicelist3 = [], [], [], []
            multiclass_softdicelist = []
            for i in range(len(ES_images_3D)):
                ES_image = ES_images_3D[i]
                gt_ES_image = gt_ES_images_3D[i]

                multiclass_softdice = 0
                # calculate dice per channel for each 3D volume
                for label in range(ES_image.shape[3]):
                    softdice = round(softdice_coef_np(gt_ES_image[:,:,:,label], ES_image[:,:,:,label]),4)
                    print("softdice {} for label {}".format(softdice, label))

                    # for i in range(gt_ED_image.shape[0]):
                    #     gt = gt_ED_image[i,:,:,label]
                    #     ED = ED_image[i,:,:,label]
                    #     plt.figure()
                    #     plt.imshow(gt)
                    #     plt.figure()
                    #     plt.imshow(ED)


                    multiclass_softdice += softdice

                    if label == 0:
                        softdicelist0.append(softdice)
                    elif label == 1:
                        softdicelist1.append(softdice)
                    elif label == 2:
                        softdicelist2.append(softdice)
                    elif label == 3:
                        softdicelist3.append(softdice)

                # add multiclass softdice to list
                multiclass_softdicelist.append(multiclass_softdice)



            # average dice per image, min/max
            # average dice for whole test set, min/max
            print("Minimum softdice for channel 0 is {}".format(np.min(softdicelist0)))
            print("Maximum softdice for channel 0 is {}".format(np.max(softdicelist0)))
            print("Average softdice for channel 0 is {}".format(np.mean(softdicelist0)))

            print("Minimum softdice for channel 1 is {}".format(np.min(softdicelist1)))
            print("Maximum softdice for channel 1 is {}".format(np.max(softdicelist1)))
            print("Average softdice for channel 1 is {}".format(np.mean(softdicelist1)))

            print("Minimum softdice for channel 2 is {}".format(np.min(softdicelist2)))
            print("Maximum softdice for channel 2 is {}".format(np.max(softdicelist2)))
            print("Average softdice for channel 2 is {}".format(np.mean(softdicelist2)))

            print("Minimum softdice for channel 3 is {}".format(np.min(softdicelist3)))
            print("Maximum softdice for channel 3 is {}".format(np.max(softdicelist3)))
            print("Average softdice for channel 3 is {}".format(np.mean(softdicelist3)))

            print("Multiclass softdices {}".format(multiclass_softdicelist))
            print("Multiclass softdices divided by 4 {}".format(np.array(multiclass_softdicelist) / 4))
            print("Minimum multiclass softdice {}".format(np.min(multiclass_softdicelist)))
            print("Maximum multiclass softdice {}".format(np.max(multiclass_softdicelist)))
            print("Average multiclass softdice {}".format(np.mean(multiclass_softdicelist)))
            print("Average multiclass softdice divided by 4 is {}".format((np.mean(multiclass_softdicelist))/4))


            print(test_ED_shape.shape)
            print(ED_images_3D.shape)
            print(test_ED_shape[0].shape)
            print(ED_images_3D[0].shape)

            # plt.figure()
            # plt.imshow(ED_images_3D[6][5,:,:,3])
            # plt.show()

            # functie maken die een 3D plaatje plot met subplot per slice
            # evt erbij dat de verschillende classes zichtbaar zijn
            # for i in range(19):
            #     visualize3Dimage(ES_images_3D[i], test_ES_shape[i])
            #
            #
            # plt.show()
            # EF uitrekenen

            # for i in range(len(gt_images_3D)):
                # print(np.unique(gt_images_3D[i]))









            # num_test_images = len(test_ED_ims)
            # test_gen = image_generator(test_ED_ims, test_gt_ED_ims, batch_size)
            # preds_ED = unet_2D.predict_generator(test_gen, steps=(num_test_images//batch_size))
            # test_gen = image_generator(test_ES_ims, test_gt_ES_ims, batch_size)
            # preds_ES = unet_2D.predict_generator(test_gen, steps=(num_test_images//batch_size))






        return

    return

main()
