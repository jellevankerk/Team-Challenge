# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:20:04 2019

@author: Administrator
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy

import random
random.seed(1)

from keras.models import Model
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D
from keras.optimizers import SGD
from keras.utils import to_categorical
import keras

from scipy import ndimage
import sklearn.preprocessing


def main():
    paths = glob.glob(r'C:\Users\s127400\Desktop\Data\patient*/patient*_frame*.nii.gz')
    networkpath_2D = r'C:\Users\s127400\Desktop\trainednetwork_2D_test.h5'
    networkpath_3D = r'C:\Users\s127400\Desktop\trainednetwork_3D_test.h5'
    
    trainnetwork = True
    trainingsetsize = 0.8
    testsetsize = 0.2
    nr_epochs = 200
    minibatchsize = 1
    
    # train on 2D or 3D images
    dimensions = 2
    
    # load all images
    ED_ims, ES_ims, gt_ED_ims, gt_ES_ims, spacings = loadImages(paths)
    
    
    # take test samples away so no training will be done on those (e.g. 20% of all samples for testing, other 80% for training and validation)
    testsamples = int(np.ceil(ED_ims.shape[0]*testsetsize))
    allimages_ED_test = ED_ims[:testsamples] # (20,)
    gt_allimages_ED_test = gt_ED_ims[:testsamples] # (20,)
    allimages_ED_training = ED_ims[testsamples:] # (80,)
    gt_allimages_ED_training = gt_ED_ims[testsamples:] # (80,)
    
    allimages_ES_test = ES_ims[:testsamples] # (20,)
    gt_allimages_ES_test = gt_ES_ims[:testsamples] # (20,)
    allimages_ES_training = ES_ims[testsamples:] # (80,)
    gt_allimages_ES_training = gt_ES_ims[testsamples:] # (80,)

    # now test samples are separated and still separated in ED and ES images
    # this will make computing the EF easier later on
    # for training, ES and ED can be joined together
    allimages_training = np.concatenate((allimages_ED_training, allimages_ES_training), axis=0) # (160,)
    gt_allimages_training = np.concatenate((gt_allimages_ED_training, gt_allimages_ES_training), axis=0) # (160,)
    
    # take 95% of intensities for every image (make highest 5% of intensities same as upper limit)
    for i in range(len(allimages_training)):
        im = allimages_training[i]
#        mean = globalmean
#        std = globalstd
        mean = int(np.mean(im))
        std = int(np.mean(im))
        
        up_lim = mean + 2*std
        im_thres = im
        im_thres[im > up_lim] = up_lim
        
        #print(np.mean(im_thres), np.std(im_thres))
        allimages_training[i] = im_thres
    
    # find minimum and maximum value of intensities of the dataset
    maxval = 0
    minval = 1E8
    for i in range(len(allimages_training)):
        if np.max(allimages_training[i]) > maxval:
            maxval = np.max(allimages_training[i])
        if np.min(allimages_training[i]) < minval:
            minval = np.min(allimages_training[i])
   
#    for i in range(len(allimages_training)):
#        plt.figure()
#        plt.imshow(allimages_training[i][5,:,:],cmap='gray')
    
#    print(minval)
#    print(maxval)
            
            
    # normalize images to [0,1] range 
    # (x - x.min()) / (x.max() - x.min()) # values from 0 to 1
    # 2*(x - x.min()) / (x.max() - x.min()) - 1 # values from -1 to 1
    # (x - x.mean()) / x.std() # values from ? to ?, but mean at 0
    for i in range(len(allimages_training)):
        im = allimages_training[i].astype(float)
        im *= 255/(maxval-minval) #normalize to 0,255
        # im = -= np.mean(allimages_training[i]) # mean subtraction
        im = im.astype(int)
        allimages_training[i] = im
#        print(np.max(allimages_training[i]))
#        print(np.min(allimages_training[i]))

    # get number of training samples (e.g. 80% of all samples, other 20% will be for validation)
    trainingsamples = int(np.ceil(allimages_training.shape[0]*trainingsetsize))
    allimages_valid = allimages_training[trainingsamples:] # (32,)
    gt_allimages_valid = gt_allimages_training[trainingsamples:] # (32,)
    allimages_train = allimages_training[:trainingsamples] # (128,)
    gt_allimages_train = gt_allimages_training[:trainingsamples] # (128,)
    
    if dimensions == 2:
        
        # get centers of mass
        train_centers = centerROI(gt_allimages_train)
        valid_centers = centerROI(gt_allimages_valid)
        
        # preprocess images to same x,y dimensions around center of mass
        allimages_train_red = reduceDimensions(images=allimages_train, dims=[144,144], centers=train_centers)
        gt_allimages_train_red = reduceDimensions(images=gt_allimages_train, dims=[144,144], centers=train_centers)
        allimages_valid_red = reduceDimensions(images=allimages_valid, dims=[144,144], centers=valid_centers)
        gt_allimages_valid_red = reduceDimensions(images=gt_allimages_valid, dims=[144,144], centers=valid_centers)

        # display some slices    
        #displaySlices(allimages_train_red, patient=77)
    
        # display reference segmentations over slices
        #displayReferenceSegmentations(allimages_train_red, gt_allimages_train_red, patient=77)
    
        # compute reference ejection fraction
        #computeRefEF(gt_ED_ims_red, gt_ES_ims_red, spacings, patient=77)
        
        
        # convert list of arrays to array
        allimages_train_red = np.vstack(allimages_train_red) # (1189, 144, 144)
        gt_allimages_train_red = np.vstack(gt_allimages_train_red) # (1189, 144, 144)
        allimages_valid_red = np.vstack(allimages_valid_red) # (323, 144, 144)
        gt_allimages_valid_red = np.vstack(gt_allimages_valid_red) # (323, 144, 144)
        
        # need extra dimension so input is (image, rows, cols, channels)
        allimages_train_red = np.expand_dims(allimages_train_red, axis=3)
        allimages_valid_red = np.expand_dims(allimages_valid_red, axis=3)
        
        # one-hot encode the labels
        gt_allimages_train_red = to_categorical(gt_allimages_train_red)
        gt_allimages_valid_red = to_categorical(gt_allimages_valid_red)
         
        # initialize network
        unet_2D = get_unet_2D(144,144)
        unet_2D.summary()
      
        if trainnetwork:
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
            hist = unet_2D.fit(allimages_train_red, gt_allimages_train_red, batch_size=minibatchsize, epochs=nr_epochs, validation_data=(allimages_valid_red, gt_allimages_valid_red), callbacks=[early_stopping], verbose=1, shuffle=True)
            print(hist.history.keys())
            unet_2D.save(networkpath_2D)
            
            visualizeTraining(hist)
        else:
            unet_2D = keras.models.load_model(networkpath_2D)
    
    
        return
    
    if dimensions == 3:
    
        # images are all different size; so to make them the same size we can find
        # the maximum values for the dimensions and pad all images with zeros to this size
        maxdims = findMaxDims([allimages_train, allimages_valid, gt_allimages_train, gt_allimages_valid])
#        allimages_train_padded = padImages(allimages_train, maxdims)
#        allimages_valid_padded = padImages(allimages_valid, maxdims)
#        gt_allimages_train_padded = padImages(gt_allimages_train, maxdims)
#        gt_allimages_valid_padded = padImages(gt_allimages_valid, maxdims)
        
        allimages_train_padded = allimages_train
        allimages_valid_padded = allimages_valid
        gt_allimages_train_padded = gt_allimages_train
        gt_allimages_valid_padded = gt_allimages_valid
        
        print(maxdims)
        print(allimages_train.shape)
        print(allimages_train_padded.shape)
        print(allimages_valid_padded.shape)
        print(gt_allimages_train_padded.shape)
        print(gt_allimages_valid_padded.shape)
        
        # need extra dimension for channels for network training
#        allimages_train_padded = np.expand_dims(allimages_train_padded, axis=4)
#        allimages_valid_padded = np.expand_dims(allimages_valid_padded, axis=4)
        
        print(allimages_train_padded.shape)
        print(allimages_valid_padded.shape)
        
        # one-hot encode the labels
#        gt_allimages_train_padded = to_categorical(gt_allimages_train_padded, num_classes=4)
#        gt_allimages_valid_padded = to_categorical(gt_allimages_valid_padded, num_classes=4)
        
        print(gt_allimages_train_padded.shape)
        print(gt_allimages_valid_padded.shape)
        
        # initialize network
        #unet_3D = get_unet_3D(maxdims[0], maxdims[1], maxdims[2])
        unet_3D = get_unet_3D(None, None, None)
        unet_3D.summary()
    
        if trainnetwork:
            hist = unet_3D.fit(allimages_train_padded, gt_allimages_train_padded, batch_size=minibatchsize, epochs=nr_epochs, validation_data=(allimages_valid_padded, gt_allimages_valid_padded), verbose=1, shuffle=True)
            print(hist.history.keys())
            unet_3D.save(networkpath_3D)
            
            visualizeTraining(hist)
        else:
            unet_3D = keras.models.load_model(networkpath_3D)
    
    return

def visualizeTraining(hist):
    # Plot training & validation accuracy values
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def normalize(scans, images):
    
    for i in range(len(scans)):
        flat = scans[i].flatten()
        std = np.std(flat)
        images[i] = images[i]/std

def centerROI(gt_images):
    centers = []
    for i in range(len(gt_images)):
        sumy = 0
        sumx = 0 
        div = len(gt_images[i])
        for j in range(len(gt_images[i])):
       
            im = gt_images[i][j]
            #im[im>1] = 1
            center=ndimage.measurements.center_of_mass(im)
            if np.isnan(center[0]) == False or np.isnan(center[1]) == False:
                sumy = sumy + center[0]
                sumx = sumx + center[1]
            else:
                div = div - 1
        centers.append((np.ceil(sumy/div),np.ceil(sumx/div)))
    return centers

def findMaxDims(images):
    # find maximum dimensions of all (training and validation) images
    maxdims = [0, 0, 0]
    for i in range(len(images)):
        for j in range(len(images[i])):
            im = images[i][j]
            
            maxdims[0] = max(im.shape[0],maxdims[0])
            maxdims[1] = max(im.shape[1],maxdims[1])
            maxdims[2] = max(im.shape[2],maxdims[2])
    #maxdims = [20, 512, 432]
    return maxdims

def padImages(images, maxdims):
    # pad all images to max dimensions
    padded_images = np.zeros([len(images), maxdims[0], maxdims[1], maxdims[2]])
    for i in range(len(images)):
        padded_im = np.zeros(maxdims)
        im = images[i]
        padded_im[:im.shape[0],:im.shape[1],:im.shape[2]] = im
        padded_images[i] = padded_im
    return padded_images

#def generator(inputs, labels):
#    i = 0
#    while True:
#        inputs_batch = np.expand_dims([inputs[i%len(inputs)]], axis=2)
#        labels_batch = np.array([labels[i%len(inputs)]])
#        yield inputs_batch, labels_batch
#        i+=1

def loadImages(paths):
    print("Loading images...")
    ED_paths, ES_paths = [], []
    
    # get paths for first frame
    for i in range(0,len(paths),4):
        ED_paths.append(paths[i])
        
    # get paths for last frame
    for i in range(2,len(paths),4):   
        ES_paths.append(paths[i])
        
    # get paths for first frame ground truth
    gt_ED_paths = copy.deepcopy(ED_paths)
    for i in range(len(ED_paths)):
        gt_ED_paths[i] = ED_paths[i].replace('.nii.gz', '_gt.nii.gz')
    
    # get paths for last frame ground truth
    gt_ES_paths = copy.deepcopy(ES_paths)
    for i in range(len(ES_paths)):
        gt_ES_paths[i] = ES_paths[i].replace('.nii.gz', '_gt.nii.gz')
        
    ED_ims, ES_ims, gt_ED_ims, gt_ES_ims, spacings = [], [], [], [], []
    
    for i in range(len(ED_paths)):
        # load images
        ED_im = sitk.ReadImage(ED_paths[i])
        ES_im = sitk.ReadImage(ES_paths[i])
        ED_gt_im = sitk.ReadImage(gt_ED_paths[i])
        ES_gt_im = sitk.ReadImage(gt_ES_paths[i])
        
        # add spacings to a list
        spacings.append(ED_im.GetSpacing())
        
        # make np array of the images
        ED_im = sitk.GetArrayFromImage(ED_im)
        ES_im = sitk.GetArrayFromImage(ES_im)
        gt_ED_im = sitk.GetArrayFromImage(ED_gt_im)
        gt_ES_im = sitk.GetArrayFromImage(ES_gt_im)
        
        #print(ED_im.shape)
        
        ED_ims.append(ED_im)
        ES_ims.append(ES_im)
        gt_ED_ims.append(gt_ED_im)
        gt_ES_ims.append(gt_ES_im)
    
    # make np array of the lists of arrays
    ED_ims = np.asarray(ED_ims)
    ES_ims = np.asarray(ES_ims)
    gt_ED_ims = np.asarray(gt_ED_ims)
    gt_ES_ims = np.asarray(gt_ES_ims)
    
    print("Images loaded.")
    return ED_ims, ES_ims, gt_ED_ims, gt_ES_ims, spacings

def displaySlices(images, patient):
    im = images[patient]
    # plot a grid of all 2D slices over one frame
    plt.rcParams["figure.figsize"] = (22, 26) # (w, h)
    n_rows = np.ceil(np.sqrt(im.shape[0]))
    n_cols = np.ceil(np.sqrt(im.shape[0]))
    
    for z in range(im.shape[0]): 
      plt.subplot(n_rows, n_cols, 1 + z)
      plt.imshow(im[z, :, :], clim=(0, 150), cmap='gray')
      plt.title('Slice {}'.format(z + 1))
    plt.show()
    return

def displayReferenceSegmentations(images, gt_images, patient):
    im = images[patient]
    gt_im = gt_images[patient]

    n_rows = np.ceil(np.sqrt(im.shape[0]))
    n_cols = np.ceil(np.sqrt(im.shape[0]))
    
    for z in range(im.shape[0]): 
      plt.subplot(n_rows, n_cols, 1 + z)
      plt.imshow(im[z, :, :], clim=(0, 150), cmap='gray')
      plt.imshow(np.ma.masked_where(gt_im[z, :, :]!=2, gt_im[z, :, :]==2), alpha=0.6, cmap='Blues', clim=(0, 1))  
      plt.imshow(np.ma.masked_where(gt_im[z, :, :]!=3, gt_im[z, :, :]==3), alpha=0.6, cmap='Reds', clim=(0, 1))
      plt.title('Slice {}'.format(z + 1))
    plt.show()
    return
    
def computeRefEF(gt_ED_ims, gt_ES_ims, spacings, patient):
    spacing = spacings[patient]
    gt_ED_im = gt_ED_ims[patient]
    gt_ES_im = gt_ED_ims[patient]
    
    voxelvolume = spacing[0]*spacing[1]*spacing[2]
    ED_volume = np.sum(gt_ED_im==3)*voxelvolume
    ES_volume = np.sum(gt_ES_im==3)*voxelvolume
    
    strokevolume = ED_volume - ES_volume
    LV_EF = (strokevolume/ED_volume)*100
    
    print('LV stroke volume is {:.2f} ml and ejection fraction is {:.2f}%'.format(strokevolume*0.001, LV_EF))
    return

# reduces x and y dimensions, keeps z the same
def reduceDimensions(images, dims, centers):
    images_red = np.empty_like(images)
    # for every image in the list
    for i in range(len(images)):
        xmid = int(centers[i][1]) #int(np.ceil(images[i].shape[2]/2))
        ymid = int(centers[i][0])#int(np.ceil(images[i].shape[1]/2))
        if xmid <= dims[1]/2:
            dif = round(dims[1]/2 - xmid)
            x1 = xmid - int(dims[1]/2) + dif
            x2 = xmid + int(dims[1]/2) + dif

            
        elif ymid <= dims[0]/2:
            dif = round(dims[0]/2 - ymid)
            y1 = ymid - int(dims[0]/2) + dif
            y2 = ymid + int(dims[0]/2) + dif
        else:

            y1 = ymid - int(dims[0]/2)
            y2 = ymid + int(dims[0]/2)
            x1 = xmid - int(dims[1]/2)
            x2 = xmid + int(dims[1]/2)

        im = images[i]
        images_red[i] = im[:,y1:y2,x1:x2]
        
        #print(images_red[i].shape)
    return images_red

def get_unet_2D(img_rows, img_cols):
    inputs = Input(shape=(img_rows, img_cols, 1)) #(1, dim1, dim2, channels)
    print(inputs.shape)
    
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(pool1)
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(pool2)
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
    
    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(pool3)
    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv4)
    
    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = concatenate([conv3, up1], axis=3)
    conv5 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(up1)
    conv5 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv5)
    
    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([conv2, up2], axis=3)
    conv6 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(up2)
    conv6 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv6)
    
    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = concatenate([conv1, up3], axis=3)
    conv7 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(up3)
    conv7 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv7)
    
    conv7 = Convolution2D(4, (1, 1), activation='softmax')(conv7)
    
    unet_2D = Model(inputs=inputs, outputs=conv7)
    
    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    unet_2D.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return unet_2D

def get_unet_3D(img_rows, img_cols, img_slices):
    inputs = Input(shape=(img_rows, img_cols, img_slices, 1)) #(1, dim1, dim2, dim3, channels)
    print(inputs.shape)
    
    conv1 = Convolution3D(10, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1 = Convolution3D(10, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)
    
    conv2 = Convolution3D(20, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool1)
    conv2 = Convolution3D(20, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
    
    conv3 = Convolution3D(40, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool2)
    conv3 = Convolution3D(40, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)
    
    conv4 = Convolution3D(80, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool3)
    conv4 = Convolution3D(80, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv4)
    
    up1 = UpSampling3D(size=(2, 2, 2))(conv4)
    up1 = concatenate([conv3, up1], axis=4)
    conv5 = Convolution3D(40, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up1)
    conv5 = Convolution3D(40, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv5)
    
    up2 = UpSampling3D(size=(2, 2, 2))(conv5)
    up2 = concatenate([conv2, up2], axis=4)
    conv6 = Convolution3D(20, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up2)
    conv6 = Convolution3D(20, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv6)
    
    up3 = UpSampling3D(size=(2, 2, 2))(conv6)
    up3 = concatenate([conv1, up3], axis=4)
    conv7 = Convolution3D(10, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up3)
    conv7 = Convolution3D(10, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv7)
    
    conv7 = Convolution3D(4, (1, 1, 1), activation='softmax')(conv7)
    
    unet_3D = Model(inputs=inputs, outputs=conv7)
    
    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    unet_3D.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    return unet_3D

main()