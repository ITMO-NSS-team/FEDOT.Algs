#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:29:59 2021

@author: mike_ubuntu
"""

#import pytest
import numpy as np
from collections import OrderedDict
import sys
sys.path.append('/media/mike_ubuntu/DATA/ESYS/')

import src.globals as global_var
from src.token_family import Token_family
from src.factor import Factor
from src.cache.cache import upload_grids


def mock_evaluator(factor):
    return np.ones((10, 10, 10))

class mock_token_family(Token_family):
    def __init__(self, names = [], evaluator = None):
        super().__init__('mock')
        super().use_glob_cache()
        super().set_status()
      
        mock_equal_params = {'not_power' : 0, 'power' : 0}
        mock_params = OrderedDict([('not_power', (1, 4)), ('power', (1, 1))])
        super().set_evaluator(evaluator)
        super().set_params(names, mock_params, mock_equal_params)       

def test_factor():
    global_var.init_caches(set_grids=False)
    
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 25)    
    names = ['mock1', 'mock2', 'mock3']
    mock = mock_token_family(names, mock_evaluator)
    test_factor_1 = Factor(names[0], mock, randomize=True)
    test_factor_2 = Factor(names[0], mock, randomize=True)
    print(test_factor_1.params, test_factor_1.params_description) 
    print(test_factor_2.params, test_factor_2.params_description)     
#    print(test_factor_3.params, test_factor_3.params_description)
    
    assert type(test_factor_1.cache_label) == tuple and type(test_factor_1.cache_label[0]) == str and type(test_factor_1.cache_label[1]) == tuple
    assert np.all(test_factor_1.evaluate() == test_factor_2.evaluate())
    print(test_factor_1.params, test_factor_2.params)
#    assert test_factor_1 == test_factor_2, 'Equally defined tokens are not equal'
    
    test_factor_3 = Factor(names[1], mock, randomize=False)
    test_factor_3.Set_parameters(random=False, not_power = 2, power = 1)
    test_factor_4 = Factor(names[1], mock, randomize=False)
    test_factor_4.Set_parameters(random=False, not_power = 2, power = 1)
    assert test_factor_3 == test_factor_4, 'Equally defined tokens are not equal'
    
    print(test_factor_3.name)
#    assert False
    
def test_cache():
    global_var.init_caches(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 25)  
    
    x = np.linspace(0, 2*np.pi, 100)
    global_var.grid_cache.memory_usage_properties(obj_test_case=x, mem_for_cache_frac = 25)  

    upload_grids(x, global_var.grid_cache)
    print(global_var.grid_cache.memory.keys(), global_var.grid_cache.memory.values())
    assert '0' in global_var.grid_cache 
    assert x in global_var.grid_cache 
    assert (x, False) in global_var.grid_cache 
    assert not (x, True) in global_var.grid_cache
    
    x_returned = global_var.grid_cache.get('0')
    assert np.all(x == x_returned)
    global_var.grid_cache.clear(full = True)
    
    y = np.linspace(0, 10, 200)
    grids = np.meshgrid(x, y)
    upload_grids(grids, global_var.grid_cache)
    print('memory for cache:', global_var.grid_cache.available_mem, 'B')
    print('consumed memory:', global_var.grid_cache.consumed_memory, 'B')
    print(global_var.grid_cache.memory.keys())
    global_var.grid_cache.delete_entry(entry_label = '0')
    assert not '0' in global_var.grid_cache.memory.keys()
    assert False
#    assert
    
    