import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from TC_data import *

def calculateEF(seg_ED_images_3D, seg_ES_images_3D, test_spacings, patient):
    '''
    Preforms the calculations of the Ejaction fraction using the difference 
    between the segmentations of the diastole and systole.
    
    Parameters: 
        seg_ED_images_3D: an ndarray of the 3D segmented images of diastole
        seg_ES_images_3D: an ndarray of the 3D segmented images of systole 
        test_spacings: an ndarray of all the pixel spacings of the test images
        patient: an int that defines the location of the the patient data in the iterators seg_ED_images_3D and sef_ES_images_3D
        
    Returns:
        strokevolume: an float that describes the difference between the ED and the ES volumes
        LVEF: an float that describes the left ventrical ejection fraction
    '''
    
    spacing = test_spacings[patient]
    seg_ED_im = seg_ED_images_3D[patient]
    seg_ES_im = seg_ES_images_3D[patient]

    # take only the LV segmentation
    seg_ED_im = seg_ED_im[:,:,:,3]
    seg_ES_im = seg_ES_im[:,:,:,3]

    voxelvolume = spacing[0]*spacing[1]*spacing[2]
    ED_volume = np.sum(seg_ED_im)*voxelvolume
    ES_volume = np.sum(seg_ES_im)*voxelvolume

    strokevolume = ED_volume - ES_volume
    LVEF = (strokevolume/ED_volume)*100

    return strokevolume, LVEF

def saveResults(images_3D, gt_images_3D, text_file, type="ED"):
    '''
    Calculates the softdice for all 4 gt labels and uses thes results to 
    calculates the multiclass softdice. It does this for every slice in the 3D
    images. The results of the minimum, maximum and average softdices and multiclassdice for every 
    label are printed and writen to text file. 
    
    Parameters:
        images_3D: an ndarray of the 3D images segmentation
        gt_images_3D: an ndarray of the groundtruth labels for each segmentation
        text_file: an string of the path where the data is to be saved.
        type: an string; describes the type of images you want to use. ED: diastole images. ES: systole images
    
    Returns:
        None
        
    '''
    
    softdicelist0, softdicelist1, softdicelist2, softdicelist3 = [], [], [], []
    multiclass_softdicelist = []
    
    for i in range(len(images_3D)):
        image = images_3D[i]
        gt_image = gt_images_3D[i]
        multiclass_softdice = 0
        
        # calculate dice per channel for each 3D volume
        for label in range(image.shape[3]):
            softdice = round(softdice_coef_np(gt_image[:,:,:,label], image[:,:,:,label]),4)
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
        multiclass_softdicelist.append(round(multiclass_softdice,4))

    # with open(r'{}_results.txt'.format(networkpath), 'w') as text_file:
    print("For {} images: {}".format(type, '\n'), file=text_file)
    print("Softdices for channel 0: {}".format(softdicelist0), file=text_file)
    print("Softdices for channel 1: {}".format(softdicelist1), file=text_file)
    print("Softdices for channel 2: {}".format(softdicelist2), file=text_file)
    print("Softdices for channel 3: {}{}".format(softdicelist3, '\n'), file=text_file)
    print("Minimum softdice for channel 0 is {}".format(np.min(softdicelist0)), file=text_file)
    print("Maximum softdice for channel 0 is {}".format(np.max(softdicelist0)), file=text_file)
    print("Average softdice for channel 0 is {}{}".format(round(np.mean(softdicelist0),4), '\n'), file=text_file)
    print("Minimum softdice for channel 1 is {}".format(np.min(softdicelist1)), file=text_file)
    print("Maximum softdice for channel 1 is {}".format(np.max(softdicelist1)), file=text_file)
    print("Average softdice for channel 1 is {}{}".format(round(np.mean(softdicelist1),4), '\n'), file=text_file)
    print("Minimum softdice for channel 2 is {}".format(np.min(softdicelist2)), file=text_file)
    print("Maximum softdice for channel 2 is {}".format(np.max(softdicelist2)), file=text_file)
    print("Average softdice for channel 2 is {}{}".format(round(np.mean(softdicelist2),4), '\n'), file=text_file)
    print("Minimum softdice for channel 3 is {}".format(np.min(softdicelist3)), file=text_file)
    print("Maximum softdice for channel 3 is {}".format(np.max(softdicelist3)), file=text_file)
    print("Average softdice for channel 3 is {}{}".format(round(np.mean(softdicelist3),4), '\n'), file=text_file)
    print("Multiclass softdices {}{}".format(multiclass_softdicelist, '\n'), file=text_file)
    multiclass_softdicelist_d4 = np.round(np.array(multiclass_softdicelist)/4,4)
    print("Multiclass softdices divided by 4 {}{}".format(multiclass_softdicelist_d4, '\n'), file=text_file)
    print("Minimum multiclass softdice {}".format(np.min(multiclass_softdicelist)), file=text_file)
    print("Maximum multiclass softdice {}".format(np.max(multiclass_softdicelist)), file=text_file)
    print("Average multiclass softdice {}{}".format(round(np.mean(multiclass_softdicelist),4), '\n'), file=text_file)
    print("Average multiclass softdice divided by 4 is {}{}".format(round(np.mean(multiclass_softdicelist_d4),4), '\n\n'), file=text_file)
