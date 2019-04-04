import SimpleITK as sitk
import copy
import numpy as np

from scipy import ndimage
from keras import backend as K

def tversky_loss(y_true, y_pred):
    """
    Performs the calculation of the tversky loss
    
    Parameters:
        y_true: array consisting of the ground truth 
        y_pred: array consisting of probabilities
    Results:
        tv_loss: float, which includes the tversky loss
    """
    alpha=0.5
    beta=0.5
    # ones = K.ones(K.shape(y_true))
    ones = K.ones_like(y_true)
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true

    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))

    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    tv_loss= Ncl-T
    return tv_loss 

def softdice_coef(y_true, y_pred):
    """
    Performs the calculation of the softdice coefficient
    
    Parameters:
        y_true: array consisting of the ground truth 
        y_pred: array consisting of probabilities
    Results:
        sft_coef: float, which includes the softdice coefficient
        
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    sft_coef= ((2. * K.sum(y_pred_f * y_true_f)) + 1) / ((K.sum(y_true_f) + K.sum(y_pred_f)) + 1)

    return sft_coef


def softdice_coef_np(y_true, y_pred):
    """
    Performs the calculation of the numpy softdice coefficient
    
    Parameters:
        y_true: array consisting of the ground truth 
        y_pred: array consisting of probabilities
    Results:
        sft_coef_np: Float, which includes the numpy softdice coefficient
    
    """
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    
    sft_coef_np= (2. * np.sum(y_pred_f * y_true_f) + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1)

    return sft_coef_np

def softdice_coef_multilabel(y_true, y_pred, num_labels=4):
    """
    Performs the calculation of the multilabel softdice coefficient
    
    Parameters:
        y_true: array consisting of the ground truth 
        y_pred: array consisting of probabilities
        num_labels: int, number of labels
    Results:
        softdice: Float, which includes the multilabel softdice coefficient
    """
    softdice=0
    for label in range(num_labels):
        #weighting = weightings[label]
        softdice += softdice_coef(y_true[:,:,:,label], y_pred[:,:,:,label])
    return softdice

def softdice_loss(y_true, y_pred):
    """
    Computes the softdice loss
    
    Parameters:
        y_true: array consisting of the ground truth 
        y_pred: array consisting of probabilities
    Results:
        sft_loss: Float, which includes the softdice loss
        
    """
    sft_loss=1 - softdice_coef(y_true, y_pred)
    return sft_loss

def softdice_multilabel_loss(y_true, y_pred):
    """
    Computes the multilabel softdice loss
    
      Parameters:
        y_true: array consisting of the ground truth 
        y_pred: array consisting of probabilities
    Results:
        sft_multilabel_loss: Float, which includes the multilabel softdice loss
        
    """
    sft_multilabel_loss= 4 - softdice_coef_multilabel(y_true, y_pred, num_labels=4)
    return sft_multilabel_loss

def loadImages(patient):
    """
    Performs the reading, of the images for the end diastole as well for the
    end systole, and gets the voxel spacing. This is also performed for the 
    ground truth
    
    Parameters: 
        patient: str of location where data is stored
    
    Returns:
        ED_im:      array of 8int of the 3D image during end diastole
        ES_im:      array of 8int of the 3D image during end systole
        gt_ED_im:   array of 8int of the ground truth 3D image during end diastole,
                    consisting of 0, 1, 2 and 3 values
        gt_ES_im:   array of 8int of the ground truth 3D image during end systole,
                    consisting of 0, 1, 2 and 3 values
        spacing:    tuple with size 3, consisting of x,y and z direction spacing
        
    """
    with open('Data/{}/Info.cfg'.format(patient), 'r') as file:
        data=file.read().replace('\n', '')
        ED_nr = data[data.find('ED:')+4:data.find('ES:')]
        ES_nr = data[data.find('ES:')+4:data.find('Group:')]

    # if frame number is 1 digit (e.g. 1) it should become 01 for in the filenames
    if len(ED_nr) != 2:
        ED_nr = "{:02d}".format(int(ED_nr))

    if len(ES_nr) != 2:
        ES_nr = "{:02d}".format(int(ES_nr))

    # load images
    ED_im = sitk.ReadImage('Data/{}/{}_frame{}.nii.gz'.format(patient, patient, ED_nr))
    ES_im = sitk.ReadImage('Data/{}/{}_frame{}.nii.gz'.format(patient, patient, ES_nr))
    gt_ED_im = sitk.ReadImage('Data/{}/{}_frame{}_gt.nii.gz'.format(patient, patient, ED_nr))
    gt_ES_im = sitk.ReadImage('Data/{}/{}_frame{}_gt.nii.gz'.format(patient, patient, ES_nr))

    # get spacing
    spacing = ED_im.GetSpacing()

    # make np array of the images
    ED_im = sitk.GetArrayFromImage(ED_im)
    ES_im = sitk.GetArrayFromImage(ES_im)
    gt_ED_im = sitk.GetArrayFromImage(gt_ED_im)
    gt_ES_im = sitk.GetArrayFromImage(gt_ES_im)

    return ED_im, ES_im, gt_ED_im, gt_ES_im, spacing

def create2DArray(images):
    """
    Creates 2D array out of 3D array
    
    Parameters:
        images: ndarray of numpy module of 3D images
    Returns:
        ims: array of cropped 2D slices for all slices 
    """
    ims = []
    for i in range(len(images)):
        im_3D = images[i]
        for j in range(im_3D.shape[0]):
            im_2D = im_3D[j,:,:]
            ims.append(im_2D)
    ims = np.array(ims)
    return ims

def resample(images, spacings):
    """
    Performs resampling of the voxels in order to get the same voxelspacing 
    for each image. The spacing of the output includes the mean spacing of all 
    images
    
    Parameters:
        images: array of 3D images 
        spacings: float64 of x,y,z spacings for each image
        
    Returns: 
        images: array of 3D images with equal spacings
    """
    
    # first get the average x (and y) spacing of all images rounded to 1 decimal
    avgspacing = np.round(np.mean([spacings[i][0] for i in range(len(spacings))]), 1)
    # then resample every image
    for i in range(len(images)):
        spacingvector = [avgspacing, avgspacing, spacings[i][2]]
        im = images[i]
        im = sitk.GetImageFromArray(im)
        im.SetSpacing(spacingvector)
        im = sitk.GetArrayFromImage(im)
        images[i] = im
    return(images)

def removeOutliers(images):
    """
    Performs removal of ouliers, which include 5 procent of the pixels with 
    the highest value. These fixels get the value of the upper limit intensity
    
    Parameters: 
        images: array of all images
    Returns: 
        images: array of all images without the "oetliers"
    """
    # take 95% of intensities for every image
    # the highest 5% of the images will get the value of the upper limit intensity
    for i in range(len(images)):
        im = images[i]
        mean = int(np.mean(im))
        std = int(np.mean(im))

        up_lim = mean + 2*std
        im_thres = im
        im_thres[im > up_lim] = up_lim

        images[i] = im_thres

    return images

def normalize(images):
    """
    Performs normalisation at each image of the images
    
    Parameters:
        images: Array of the 3D arrays of the images
    Returns:
        images: array of the normalised 3D array of the images
    """
    # normalize images to range of zero mean, unit variance
    for i in range(len(images)):
        im = images[i]
        im = (im - np.mean(im)) / np.std(im)
        images[i] = im

    return images

def center_of_mass(gt_images):
    """
    Calculates the centre of mass of the ground truh images
    
    Parameters:
        gt_images: object consisting of array of groundtruth images
    Returns:
        centers: list of centers of mass of each ground truth image
    """
    # get a list of centers of mass from a list of images
    centers = []
    for i in range(len(gt_images)):
        sumy = 0
        sumx = 0
        div = len(gt_images[i])

        for j in range(len(gt_images[i])):
            im = gt_images[i][j]
            center=ndimage.measurements.center_of_mass(im)
            if np.isnan(center[0]) == False or np.isnan(center[1]) == False:
                sumy = sumy + center[0]
                sumx = sumx + center[1]

            else:
                div = div - 1

        centers.append((np.ceil(sumy/div),np.ceil(sumx/div)))
    return centers

def reduceDimensions(images, dims, centers):
    """
    Performs the cropping of the images witch the center of mass as origin
    
    Parameters:
        images: object consisting of the 3D arrays of the images
        dims: list of in including new x and y dimension
        centers: list of center of mass location of each image
        
    Returns:
        images_red: list of cropped images
    """
    # reduces x and y dimensions, keeps z the same
    images_red = np.empty_like(images)
    for i in range(len(images)):
        xmid = int(centers[i][1])
        ymid = int(centers[i][0])
        if xmid <= dims[1]/2:
            dif = round(dims[1]/2 - xmid)
            x1 = xmid - int(dims[1]/2) + dif
            x2 = xmid + int(dims[1]/2) + dif
        else:
            x1 = xmid - int(dims[1]/2)
            x2 = xmid + int(dims[1]/2)

        if ymid <= dims[0]/2:
            dif = round(dims[0]/2 - ymid)
            y1 = ymid - int(dims[0]/2) + dif
            y2 = ymid + int(dims[0]/2) + dif
        else:
            y1 = ymid - int(dims[0]/2)
            y2 = ymid + int(dims[0]/2)

        im = images[i]
        images_red[i] = im[:,y1:y2,x1:x2]
    return images_red

def to_categorical(y, num_classes=None):
    """
    Devides the ground truth images consisting of 4 classes into 4 images with
    the segmentation of each class with binary values.
    
    Parameters: 
        y: 2D array of the ground truth images
        num_classes: int of the number of classes used 
    Returns:
        categorical: 4 binary images of the ground truth of each segmentation
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
