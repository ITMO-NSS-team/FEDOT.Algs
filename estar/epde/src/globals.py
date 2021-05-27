#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:14:57 2021

@author: mike_ubuntu
"""

from epde.src.cache.cache import Cache

def init_caches(set_grids = False):
    global tensor_cache, grid_cache
    tensor_cache = Cache()
    if set_grids:
        grid_cache = Cache()
    else:
        grid_cache = None
        
def delete_cache():
    global tensor_cache, grid_cache
    try:
        del tensor_cache
    except NameError:
        print('Failed to delete tensor cache due to its inexistance')
    try:
        del grid_cache
    except NameError:
        print('Failed to delete grid cache due to its inexistance')
        