import numpy as np

from TC_data import loadImages


def predict_label_seg(pred):
    '''
    make the segmentations according to the probabilities.
    every class label with probability >= 0.5 gets value of class label (0, 1, 2 or 3).
    
    Parameters:
        pred: an ndarray of the prediction images made by the network
    
    Returns:
        None
    '''
    for i in range(pred.shape[0]):
        image = pred[i,:,:,:]
        for label in range(image.shape[2]):
            im = image[:,:,label]
            im = np.where(im >= 0.5, 1, 0)
            image[:,:,label] = im
            pred[i,:,:,:] = image
            
def load_images_from_set(patients):
    '''
    Loads all images from a certain set of paths and returns the images , 
    groundtruths and spacings of the ES and ED images. you have the trainingset
    and the test set
    
    Paratmeters: 
        patients: a list of strings, with each path of certain set of images
    
    Returns:
        ED_ims: a list of all the ED images
        ES_ims: a list of all the ES images
        gt_ED_ims: a list  of all the groundtruth ED images
        gt_ES_ims: a list of all the groundtruth ES images
        spacings: a list of all the spacings for of the images
    '''
    ED_ims, ES_ims, gt_ED_ims, gt_ES_ims, spacings = [], [], [], [], []
    
    # do for every patient
    for i in range(len(patients)):
        patient = patients[i]
        ED_im, ES_im, gt_ED_im, gt_ES_im, spacing = loadImages(patient)
        ED_ims.append(ED_im)
        ES_ims.append(ES_im)
        gt_ED_ims.append(gt_ED_im)
        gt_ES_ims.append(gt_ES_im)
        spacings.append(spacing)
        
    return ED_ims, ES_ims, gt_ED_ims, gt_ES_ims, spacings

def reconstruct_3D(im_shape, gt_shape, gt_ims, pred):
    '''
    Makes an reconstruction of 3D images 
    
    '''
    
    images_3D = np.empty_like(im_shape)
    gt_images_3D = np.empty_like(gt_shape)
    for i in range(len(im_shape)):
          # use saved shape to determine how many slices each 3D image should have
          slices = int(im_shape[i].shape[0])
          image_3D = pred[i:(i+slices),:,:,:]
          gt_image_3D = gt_ims[i:(i+slices),:,:,:]
          images_3D[i] = image_3D
          gt_images_3D[i] = gt_image_3D
          
    return images_3D , gt_images_3D
        