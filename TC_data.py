import SimpleITK as sitk
import copy
import numpy as np



from scipy import ndimage

# import warnings
# warnings.filterwarnings('error')

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

        ED_ims.append(ED_im)
        ES_ims.append(ES_im)
        gt_ED_ims.append(gt_ED_im)
        gt_ES_ims.append(gt_ES_im)

    # make np array of the lists of arrays
    ED_ims = np.asarray(ED_ims)
    ES_ims = np.asarray(ES_ims)
    gt_ED_ims = np.asarray(gt_ED_ims)
    gt_ES_ims = np.asarray(gt_ES_ims)
    spacings = np.asarray(spacings)

    print("Images loaded.")
    return ED_ims, ES_ims, gt_ED_ims, gt_ES_ims, spacings

def resample(images):


    return(images)

def removeOutliers(images):
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
    # normalize images to range of [0,1]
    for i in range(len(images)):
        im = images[i]
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        images[i] = im

    return images

def center_of_mass(gt_images):
    # get a list of centers of mass from a list of images
    centers = []
    for i in range(len(gt_images)):
        sumy = 0
        sumx = 0
        div = len(gt_images[i])

        for j in range(len(gt_images[i])):
            im = gt_images[i][j]

            # when the gt slice is empty, measuring the center_of_mass gives a warning
            # this can be ignored
            # try:
            center=ndimage.measurements.center_of_mass(im)
            #     warnings.warn(Warning())
            # except Warning:
            #     pass

            if np.isnan(center[0]) == False or np.isnan(center[1]) == False:
                sumy = sumy + center[0]
                sumx = sumx + center[1]

            else:
                div = div - 1

        centers.append((np.ceil(sumy/div),np.ceil(sumx/div)))

    return centers

def reduceDimensions(images, dims, centers):
    # reduces x and y dimensions, keeps z the same
    images_red = np.empty_like(images)
    for i in range(len(images)):
        xmid = int(centers[i][1])
        ymid = int(centers[i][0])
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

    return images_red
