import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from TC_data import *

def evaluatePrediction(segmentation, ground_truth):
    softdicelist = []
    # for each channel print softdice value and also the mean softdice
    for label in range(segmentation.shape[2]):
        seg = segmentation[:,:,label]
        gt = ground_truth[:,:,label]

        softdice = softdice_coef_np(seg, gt)
        print("softdice {} for class {}".format(softdice, label))
        softdicelist.append(softdice)

    return softdicelist


def softdice(segmentation, ground_truth):
    # output shape should be (144,144,4)
    softdicelist = []

    # for every channel
    for i in range(segmentation.shape[2]):
        # check what is the threshold that gives highest softdice value
        sdsclist = []
        thresholds = np.linspace(0, 1.0, num=20, endpoint=True)
        for threshold in thresholds:
            seg = segmentation[:,:,i]
            gt = ground_truth[:,:,i]

            seg[seg>threshold] = 1
            seg[seg<=threshold] = 0

            seg = seg.astype('uint8')
            gt = gt.astype('uint8')

            sdsc = ((2*np.sum(seg & gt))+1) / (np.sum(gt) + np.sum(seg)+1)
            sdsclist.append(sdsc)

        # find max softdice for this channel, using different thresholds
        maxsdsc = np.max(sdsclist)
        max_index = sdsclist.index(maxsdsc)
        maxthreshold = thresholds[max_index]

        softdicelist.append("Max softdice value {} for channel {} found for threshold {}".format(maxsdsc, i, maxthreshold))



    #sdsc = np.mean(softdicelist)
    sdsc="test"
    return softdicelist, sdsc

def prepareTestData(test_ED_ims, test_ES_ims, test_gt_ED_ims, test_gt_ES_ims, test_spacings, cropdims):
    # normalize the images
    test_ED_ims = normalize(test_ED_ims)
    test_ES_ims = normalize(test_ES_ims)
    CoM_test_ED_ims = center_of_mass(test_gt_ED_ims)
    test_ED_ims = reduceDimensions(test_ED_ims, cropdims, CoM_test_ED_ims)
    test_gt_ED_ims = reduceDimensions(test_gt_ED_ims, cropdims, CoM_test_ED_ims)
    CoM_test_ES_ims = center_of_mass(test_gt_ES_ims)
    test_ES_ims = reduceDimensions(test_ES_ims, cropdims, CoM_test_ES_ims)
    test_gt_ES_ims = reduceDimensions(test_gt_ES_ims, cropdims, CoM_test_ES_ims)

    test_ED_ims = create2DArray(test_ED_ims)
    test_ES_ims = create2DArray(test_ES_ims)
    test_gt_ED_ims = create2DArray(test_gt_ED_ims)
    test_gt_ES_ims = create2DArray(test_gt_ES_ims)

    test_ED_ims = np.expand_dims(test_ED_ims, axis=3)
    test_ES_ims = np.expand_dims(test_ES_ims, axis=3)

    test_gt_ED_ims = to_categorical(test_gt_ED_ims, num_classes=4)
    test_gt_ES_ims = to_categorical(test_gt_ES_ims, num_classes=4)

    return test_ED_ims, test_ES_ims, test_gt_ED_ims, test_gt_ES_ims, test_spacings



def make_predictions(patient):

    # read images
    ED_im, ES_im, gt_ED_im, gt_ES_im, spacing = loadImages(patient)

    print(ED_im.shape)
    print(ES_im.shape)
    print(gt_ED_im.shape)
    print(gt_ES_im.shape)

    # for every 2D slice in the image
    for i in range(ED_im.shape[0]):
        ED_im_2D = ED_im[i,:,:]




    return

def test_image_generator(test_ims, gt_test_ims, batch_size, aug=None):
    num = 0
    while True:
        images = []
        gt_images = []
        # keep looping until we reach batch size
        while len(images) < batch_size:
            image = test_ims[num]
            gt_image = gt_train_ims[num]

            # make extra dimension for feature channels
            image = np.expand_dims(image, axis=3)

            # one-hot encode the labels
            gt_image = to_categorical(gt_image, num_classes=4)

            images.append(image)
            gt_images.append(gt_image)
            num += 1

        yield(np.array(images), np.array(gt_images))

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
