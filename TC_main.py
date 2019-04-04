'''
Team challange group 1
Writen by: Colin Nieuwlaat, Jelle van Kerkvoorde, Mandy de Graaf, Megan Schuurmans & Inge van der Schagt

General description:
    This program performs segmentation of the left vertricle, myocardium, right ventricle
    and backgound of Cardiovascular Magnetic Resonance Images, with use of a convolutional Unet based
    neural network. From each patient,  both a 3D end systolic image and a 3D end diastolic image is 
    with its ground truth is available. The data set is devided into a training set, validation set and a test set. 
    
    First, The images are obtained from the stored location. Subsequently, these images are 
    pre-processed which includes normalisation, removal of outliers and cropping. The training
    subset is used to train the Network, afterwards, the Network is validated and further trained
    by using the validation subset.
    
    After the network is trained, the network is evaluated using the subset regarding testing. 
    The test images are segmented, using the trained network and these segmentations are 
    evaluated by compairing them to the ground truth. The Dice Coefficient is calculated
    to evaluate the overlay of the segmentation and the ground truth. Furthermore, the 
    Hausdorff Distance is computed. 
    
    From the segmentations of the left ventricular cavity during the end systole and
    end diastole, the ejection fraction is calculated. This value is compared to
    the computed ejection fraction calculated from the ground truth.

Contents program:
    - TC_main.py:  Current python file, run this file to run the program.
    - TC_model.py: Contains functions that initializes the network.
    - TC_data.py:  Contains functions that initializes the data, preprocessing and 
                   metrics used in training, evaluation & testing.
    - TC_test.py:  Contains functions that show results of testing. 
    - TC_visualization.py: visualises the intermediated and final results.
    - TC_helper_functions.py: contains functions to make the main more clean
    - Data: a map with all the patient data.
    
    
Variables:
    
    trainnetwork:       Can be set to True or False. When set to True, the network
                        is trained. When set to False, a Network is loaded from the
                        networkpath.
    evaluatenetwork:    Can be set to True or False. When set to True, the network is
                        evaluated. If set to False, no evaluation is performed
    networkpath:        Path to the stored Network 
    trainingsetsize:    Number between 0 and 1 which defines the fraction of the data
                        that is used for training. 
    validationsetsize:  Number between 0 and 1 which defines the fraction of the 
                        training set that will be used for validation.
                        
    num_epochs:         Integer that defines the number of itarations. Should be increased
                        when the network should train more and should be decreased when
                        the network does not learn any more.
       

    dropout:            Can be set to True or False in order to involve Drop-out
                        in the Network or not.
    dropoutpct:         Float between 0 and 1 which defines the amount of Drop-out
                        you want to use. The higher the value, the more feature maps
                        are removed         

    lr:                 Float which defines the initial learning rate. Should be increased 
                        when decreases very slowly.
    momentum:           COLIN KUN JIJ DIT UITLEGGEN? :)
    nesterov:           Can be set to True or False.
    
    
    

Python external modules installed (at least version):
    - glob2 0.6
    - numpy 1.15.4
    - matplotlib 3.0.1
    - keras 2.2.4
    - SimpleITK 1.2.0
    - scipy 1.1.0

How to run:
    Places all the files of zip file in the same map. 
    Make sure all modules from above in you python interpreter.
    Run TC_main in a python compatible IDE.
    If you want to train your network, set  trainnetwork to True in main()
    If you want to evaluate your network, set evaluationnetwork to True in main()
    (you can find these at global settings).



'''



#%%
#IMPORT MODULES & FUNCTIONS

#import modules
import glob2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from contextlib import redirect_stdout

from keras.utils import to_categorical
import keras

#import own pythonfiles 
from TC_data import *
from TC_model import *
from TC_visualization import *
from TC_test import *
from TC_helper_functions import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%%

def main():
    
#--------------------------------------------------------------------------
# GLOBAL SETTINGS 
    
    trainnetwork = False
    evaluatenetwork = True

    # networkpath only used when trainnetwork = False but evaluatenetwork = True
    networkpath = r'Networks/network_2D_epochs=50_bs=1_channels=64-512'

    trainingsetsize = 0.8 # part of data to be used for training
    validationsetsize = 0.2 # part of train set to be used for validation
    dimensions = 2
    cropdims = [144, 144]
    num_epochs = 500
    batchsize = 1

    # model settings
    dropout = True # add dropout layers
    dropoutpct = 0.2 # amount of dropout
    activation = 'relu'

    # settings for stochastic gradient descent optimizer
    lr = 0.001
    momentum = 0.0
    decay = 1e-4
    nesterov = True
    sgd = keras.optimizers.SGD(lr, momentum, decay, nesterov)
    
#--------------------------------------------------------------------------
# LOADING IMAGES & PREPROCESSING
    
    # make list of all patients
    patients = os.listdir('Data')

    # split data for training and testing
    trainsamples = int(np.ceil(len(patients)*trainingsetsize))
    train_patients = patients[:trainsamples]
    test_patients = patients[trainsamples:]

    # now load images for all patients of the training set
    ED_ims, ES_ims, gt_ED_ims, gt_ES_ims, spacings = load_images_from_set(train_patients)
    
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
    
#--------------------------------------------------------------------------
# INITIALIZE SLICES & NETWORK FOR TRAINING OR EVALUATION

    if dimensions == 2:

        # crop images to same size (e.g. [144, 144], smallest image dimensions are x=154 and y=176)
        # first get centers of mass from the ground truth images
        CoM_ims = center_of_mass(gt_ims)

        # then reduce the x and y dimensions around the centers of mass
        train_ims = reduceDimensions(train_ims, cropdims, CoM_ims)
        gt_ims = reduceDimensions(gt_ims, cropdims, CoM_ims)

        # now dataset can be split into validation and training set
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
        unet_2D = get_unet_2D(cropdims[0], cropdims[1], sgd, dropoutpct, dropout, activation)
        unet_2D.summary()
        
#--------------------------------------------------------------------------
# TRAINING NETWORK
        
        if trainnetwork:
            # get nr of channels of model to put in network filename
            nr_channels_start = unet_2D.layers[1].output_shape[3]
            nr_channels_end = unet_2D.layers[12].output_shape[3]
            networkpath = r'Networks/network_{}D_epochs={}_bs={}_lr={}_channels={}-{}'.format(dimensions, num_epochs, batchsize, lr, nr_channels_start, nr_channels_end)

            # write model summary to file
            with open(r'{}_model.txt'.format(networkpath), 'w') as f:
                with redirect_stdout(f):
                    unet_2D.summary()

            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
            csv_logger = keras.callbacks.CSVLogger(r'{}_log.csv'.format(networkpath), separator='|', append=False)

            # train and save model
            hist = unet_2D.fit(train_ims, gt_train_ims, batch_size=batchsize, epochs=num_epochs, validation_data=(val_ims, gt_val_ims), callbacks=[csv_logger, early_stopping], verbose=1, shuffle=True)
            unet_2D.save(r'{}.h5'.format(networkpath))

            # plot loss and multilabel software of the training and save to files
            visualizeTraining(hist, dimensions, num_epochs, batchsize, nr_channels_start, nr_channels_end)

#--------------------------------------------------------------------------
# EVALUATION OF NETWORK 
            
        # evaluate the network on test data
        if evaluatenetwork:

            #load network
            keras.backend.clear_session()
            unet_2D = keras.models.load_model(r'{}.h5'.format(networkpath), custom_objects={'softdice_coef_multilabel': softdice_coef_multilabel, 'softdice_multilabel_loss': softdice_multilabel_loss, 'tversky_loss': tversky_loss})

            print("Preparing test data...")
            test_ED_ims, test_ES_ims, test_gt_ED_ims, test_gt_ES_ims, test_spacings = load_images_from_set(test_patients)
            
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

#--------------------------------------------------------------------------
# RESULTS EVALUATION
            
            # make predictions for the test images
            print("Making segmentations...")
            pred_ED = unet_2D.predict(test_ED_ims, batch_size=batchsize, verbose=0)
            pred_ES = unet_2D.predict(test_ES_ims, batch_size=batchsize, verbose=0)

            # make the segmentations according to the probabilities
            predict_label_seg(pred_ED)
            predict_label_seg(pred_ES)
       

            # reconstruct to 3D images in order to be able to calculate the EF
            ED_images_3D, gt_ED_images_3D = reconstruct_3D(test_ED_shape,test_gt_ED_shape, test_gt_ED_ims, pred_ED)
            ES_images_3D, gt_ES_images_3D = reconstruct_3D(test_ES_shape,test_gt_ES_shape, test_gt_ES_ims, pred_ES)
        
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
                
#--------------------------------------------------------------------------
#VISUALIZATION OF RESULTS    
                
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
