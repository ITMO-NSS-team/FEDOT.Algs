#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:41:16 2020

@author: mike_ubuntu
"""

import numpy as np
import datetime
import multiprocessing as mp

from prep.cheb import Process_Point_Cheb
from prep.smoothing import Smoothing

def Preprocess_derivatives(field, output_file_name = None, mp_poolsize = 4, max_order = 2, polynomial_window = 9):
    '''
    
    Main preprocessing function for the calculation of derivatives on uniform grid
    
    Parameters:
    ---------
    
    field : numpy.ndarray
        The values of studied field on uniform grid. The dimensionality of the tensor is not restricted;
        
    output_file_name : string, optional
        Name of the file, in which the tensors of caluclated derivatives will be saved; if it is not given, function returns the tensor
        
    mp_poolsize : integer, optional
        The number of workers for multiprocessing.pool, that would be created for derivative evaluation;
        
    max_order : integer, optional
        The maximum order of the derivatives to be calculated;
        
    polynomial_window : integer, optional
        The number of points, for which the polynmial will be fitted, in order to later analytically differentiate it and obtain the derivatives. 
        Shall be defined with odd number or if it is even, expect polynomial_window + 1 - number of points to be used.
    
    Returns:
    --------

    derivatives : np.ndarray
        If the output file name is not defined, or set as None, - tensor of derivatives, where the first dimentsion is the order 
        and the axis of derivative in such manner, that at first, all derivatives for first axis are returned, secondly, all 
        derivatives for the second axis and so on. The next dimensions match the dimensions of input field.
    
    '''
    t1 = datetime.datetime.now()

    polynomial_boundary = polynomial_window//2 + 1

    print('Executing on grid with uniform nodes:')
    dim_coords = []
    for dim in np.arange(np.ndim(field)):
        dim_coords.append(np.linspace(0, field.shape[dim]-1, field.shape[dim]))

        
    grid = np.meshgrid(*dim_coords, indexing = 'ij')

    field = Smoothing(field, 'gaussian', sigma = 9)
    index_array = []
    
    for idx, _ in np.ndenumerate(field):
        index_array.append((idx, field, grid, polynomial_window, max_order, polynomial_boundary))
    

    pool = mp.Pool(mp_poolsize)
    derivatives = pool.map_async(Process_Point_Cheb, index_array)
    pool.close()
    pool.join()
    derivatives = derivatives.get()
    t2 = datetime.datetime.now()

    print('Start:', t1, '; Finish:', t2)
    print('Preprocessing runtime:', t2 - t1)
        
    #np.save('ssh_field.npy', field)   
    if output_file_name:
        if not '.npy' in output_file_name:
            output_file_name += '.npy'        
        np.save(output_file_name, derivatives)
    else:
        return derivatives        
