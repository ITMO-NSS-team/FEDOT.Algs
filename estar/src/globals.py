#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:14:57 2021

@author: mike_ubuntu
"""

from src.cache.cache import Cache

def init_caches(set_grids = False):
    global tensor_cache, grid_cache
    tensor_cache = Cache()
    if set_grids:
        grid_cache = Cache()
    else:
        grid_cache = None