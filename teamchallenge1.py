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
from keras.optimizers import SGD
from keras.utils import to_categorical
import keras

def main():
    paths = glob.glob(r'C:/Users/Administrator/Documents/Team_Challenge/Analysis/TeamChallenge/Data/patient*/patient*_frame*.nii.gz')
    networkpath_2D = r'PATH/trainednetwork_2D.h5'
    
    trainnetwork = True
    trainingsetsize = 0.8
    nr_epochs = 200
    minibatchsize = 8
    
    # load all images
    ED_ims, ES_ims, gt_ED_ims, gt_ES_ims, spacings = loadImages(paths)  

    # preprocess images to same x,y dimensions
    ED_ims_red = reduceDimensions(images=ED_ims, dims=[144,144])
    ES_ims_red = reduceDimensions(images=ES_ims, dims=[144,144])
    gt_ED_ims_red = reduceDimensions(images=gt_ED_ims, dims=[144,144])
    gt_ES_ims_red = reduceDimensions(images=gt_ES_ims, dims=[144,144])
    
    # display some slices    
    #displaySlices(ED_ims_red, patient=77)
    
    # display reference segmentations over slices
    #displayReferenceSegmentations(ED_ims_red, gt_ED_ims_red, patient=77)
    
    # compute reference ejection fraction
    #computeRefEF(gt_ED_ims_red, gt_ES_ims_red, spacings, patient=77)
       
    # now "..."_train are arrays of all 2D slices
    ED_ims_train = np.vstack(ED_ims_red)
    ES_ims_train = np.vstack(ES_ims_red)
    gt_ED_ims_train = np.vstack(gt_ED_ims_red)
    gt_ES_ims_train = np.vstack(gt_ES_ims_red)
    
    #print(ED_ims_train.shape) #(951,144,144)
    
    
    # add the ED and ES images together
    allimages = np.concatenate((ED_ims_train, ES_ims_train), axis=0)
    gt_allimages = np.concatenate((gt_ED_ims_train, gt_ES_ims_train), axis=0)
    
    print(allimages.shape)
    print(gt_allimages.shape)
    
    # training/validatie set maken etc
    trainingsamples = int(np.ceil(allimages.shape[0]*trainingsetsize)) # will be 1522 (and thus validation will be 380)
    
    # create training set (X) and corresponding ground truth labels (Y)
    Xtrain = allimages[:trainingsamples]
    Ytrain = gt_allimages[:trainingsamples]
    

    
    # create validation set (X) and corresponding ground truth labels (Y)
    Xvalid = allimages[trainingsamples:]
    Yvalid = gt_allimages[trainingsamples:]
    
    # need extra dimension, input should be (image, rows, cols, channels)
    Xtrain = np.expand_dims(Xtrain, axis=3)
    Xvalid = np.expand_dims(Xvalid, axis=3)
   
    Ytrain = to_categorical(Ytrain)
    Yvalid = to_categorical(Yvalid)
    
    print(Xtrain.shape) #1522,144,144,1
    print(Xvalid.shape) #380,144,144,1
    
    print(Ytrain.shape) #1522,144,144,4
    print(Yvalid.shape) #380,144,144,4
    
    
    # initialize network
    unet_2D = get_unet_2D(144,144)
    unet_2D.summary()
    
    if trainnetwork:
        hist = unet_2D.fit(Xtrain, Ytrain, batch_size=minibatchsize, epochs=nr_epochs, validation_data=(Xvalid, Yvalid), verbose=1, shuffle=True)
        print(hist.history.keys())
        unet_2D.save(networkpath_2D)
    else:
        unet_2D = keras.models.load_model(networkpath_2D)
    
    return



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
        # load imagess
        ED_im = sitk.ReadImage(ED_paths[i])
        ES_im = sitk.ReadImage(ES_paths[i])
        ED_gt_im = sitk.ReadImage(gt_ED_paths[i])
        ES_gt_im = sitk.ReadImage(gt_ES_paths[i])
        
        # add spacings to a list
        spacings.append(ED_im.GetSpacing())
        
        # make np array of the images and add to list of images
        ED_ims.append(sitk.GetArrayFromImage(ED_im))
        ES_ims.append(sitk.GetArrayFromImage(ES_im))
        gt_ED_ims.append(sitk.GetArrayFromImage(ED_gt_im))
        gt_ES_ims.append(sitk.GetArrayFromImage(ES_gt_im))
    
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
def reduceDimensions(images, dims):
    images_red = np.empty_like(images)
    # for every image in the list
    for i in range(len(images)):
        xmid = int(np.ceil(images[i].shape[2]/2))
        ymid = int(np.ceil(images[i].shape[1]/2))
        y1 = ymid - int(dims[0]/2)
        y2 = ymid + int(dims[0]/2)
        x1 = xmid - int(dims[1]/2)
        x2 = xmid + int(dims[1]/2)

        im = images[i]
        images_red[i] = im[:,y1:y2,x1:x2]
        
        #print(images_red[i].shape)
    return images_red

def get_unet_2D(img_rows, img_cols):
    inputs = Input(shape=(img_rows, img_cols, 1)) #(batch, dim1, dim2, channels)
    print(inputs.shape)
    
    conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(pool1)
    conv2 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    
    conv3 = Convolution2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(pool2)
    conv3 = Convolution2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
  
    conv4 = Convolution2D(512, (3, 3), activation='relu', padding='same', data_format='channels_last')(pool3)
    conv4 = Convolution2D(512, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv4)
    
    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = concatenate([conv3, up1], axis=3)
    conv5 = Convolution2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(up1)
    conv5 = Convolution2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv5)
    
    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([conv2, up2], axis=3)
    conv6 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(up2)
    conv6 = Convolution2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv6)
    
    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = concatenate([conv1, up3], axis=3)
    conv7 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(up3)
    conv7 = Convolution2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv7)
    
    conv7 = Convolution2D(1, (1, 1), activation='softmax')(conv7)
    
    unet = Model(inputs=inputs, outputs=conv7)
    
    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    unet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return unet

main()