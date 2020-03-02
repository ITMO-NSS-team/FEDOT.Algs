#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:27:22 2020

@author: mike_ubuntu
"""

import numpy as np
from prep.derivatives import Preprocess_derivatives

if __name__ == "__main__":
    op_file_name = 'Preprocessing/Derivatives.npy'; filename = 'Preprocessing/wave_HP.npy' 

    if 'npy' in filename:
        field = np.load(filename)
    else:
        shape = (201, 201, 201)
        field = np.loadtxt(filename)
        field = field.reshape(shape)
    field = np.transpose(field, (2, 0, 1))
    Preprocess_derivatives(field, op_file_name, mp_poolsize=24)