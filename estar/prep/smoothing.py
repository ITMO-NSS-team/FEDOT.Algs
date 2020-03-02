#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:33:43 2020

@author: mike_ubuntu
"""

from scipy.ndimage import gaussian_filter
import numpy as np

#Extend pool of smoothers

def Smoothing(data, kernel_fun, **params):
    smoothed = np.empty(data.shape)
    if kernel_fun == 'gaussian':
        for time_idx in np.arange(data.shape[0]):
            smoothed[time_idx, :, :] = gaussian_filter(data[time_idx, :, :], sigma = params['sigma'])
    else:
        raise Exception('Wrong kernel passed into function')
    
    return smoothed