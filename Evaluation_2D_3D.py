# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:53:29 2019

@author: s131234
"""
import numpy as np
import SimpleITK as sitk

def Evaluation(segmentation, ground_truth):
    
    # DICE
    # return posive pixels
    positivesamples_pr = list(np.nonzero(segmentation))
    positivesamples_gr = list(np.nonzero(ground_truth))
    
    # compute the overlapping area
    common_area = list(set(positivesamples_pr[0]).intersection(set(positivesamples_gr[0])))

    # compute the Dice Coefficient
    Dice= 2*(len(common_area))/((len(positivesamples_pr[0])^2)+(len(positivesamples_gr[0])^2))
    
    #HAUSDORFF
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(ground_truth, segmentation)
    Hausdorff = hausdorff_distance_filter.GetHausdorffDistance()
    
    
    return Dice, Hausdorff


# get segmented image
segmentation=sitk.ReadImage(r'seg.png', sitk.sitkUInt8)

# get grund truth
ground_truth =sitk.ReadImage(r'ground.png', sitk.sitkUInt8)

[Dice, Hausdorff] = Evaluation(segmentation, ground_truth)
print(Dice)
print(Hausdorff)