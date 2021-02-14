'''
The cache object is introduced to reduce memory usage by storing the values of terms/factors of the discovered equations.

Functions:
    upload_simple_tokens : uploads the basic factor into the cache with its value in ndimensional numpy.array
    download_variable : download a variable from the disc by its and its derivatives file names, select axis for time (for normalization purposes) & cut values near area boundary
    
Objects:
    Cache : see object description (tbd)

The recommended way to declare the cache object isto declare it as a global variable: 
    >>> import src.globals as global_var
    >>> global_var.cache.memory_usage_properties(obj_test_case=XX, mem_for_cache_frac = 25) #  XX - np.ndarray from np.meshgrid, mem_for_cache_frac - max part of memory to be used for cache, %
    >>> print(global_var.cache.consumed_memory)

'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gc
import sys
import psutil
import time
from copy import deepcopy

#class Stored_Tensor(object):
#    def __init__(self, label, tensor):
#        self.label = label
#        self.tensor = tensor
#        
#    @property
#    def size(self):
#        return sys.getsizeof(self.tensor)
#    
#    def __mul__(self, other):
#        return self.label + ' * ' + other.label, np.multiply(self.tensor, other.tensor)

def upload_simple_tokens(labels, tensors, cache):
    for idx, label in enumerate(labels):
        label_completed = label + ' power 1'
        cache.add(label_completed, tensors[idx])
        cache.add_base_matrix(label_completed)
    
def download_variable(var_filename, deriv_filename, boundary, time_axis = 0):
    var = np.load(var_filename)
    initial_shape = var.shape
    var = np.moveaxis(var, time_axis, 0)
    derivs = np.load(deriv_filename)

    tokens_tensor = np.ones((1 + derivs.shape[1], ) + tuple([shape - 2*boundary for shape in var.shape]))
    tokens_tensor[0, :] = var[boundary:-boundary, boundary:-boundary, boundary:-boundary]
    for i_outer in range(0, derivs.shape[1]):
        tokens_tensor[i_outer+1] = np.moveaxis(derivs[:, i_outer].reshape(initial_shape)[boundary:-boundary, boundary:-boundary, boundary:-boundary],
                     source=time_axis, destination=0)
        
    return tokens_tensor

class Cache(object):
    def __init__(self):
        self.memory = dict()
        self.memory_normalized = dict()
        self.mem_prop_set = False
        self.base_tensors = dict() #storage of non-normalized tensors, that will not be affected by change of variables
#        
#    def set_base_matrices(self, base_labels = []):
#        self.base_labels = base_labels
#        for label in self.base_labels:
#            assert label in self.memory.keys()
#            self.add(label, self.memory[label])
        
    def add_base_matrix(self, label):
        assert label in self.memory.keys()
        self.base_tensors[label] = deepcopy(self.memory[label])
        
    def memory_usage_properties(self, obj_test_case = None, mem_for_cache_frac = None, mem_for_cache_abs = None):
        '''
        Properties:
        ...
        
        '''
        assert not (type(mem_for_cache_frac) == type(None) and type(mem_for_cache_abs) == type(None)), 'Avalable memory space not defined'
        assert type(obj_test_case) != None or len(self.memory) > 0, 'Method needs sample of stored matrix to evaluate memory allocation'        
        if type(mem_for_cache_abs) == type(None):
            self.available_mem = mem_for_cache_frac / 100. * psutil.virtual_memory().total # Allocated memory for tensor storage, bytes
        else:
            self.available_mem = mem_for_cache_abs

        assert self.available_mem < psutil.virtual_memory().available            
        
        if len(self.memory) == 0:
            assert type(obj_test_case) != None
            self.max_allowed_tensors = np.int(np.floor(self.available_mem/obj_test_case.nbytes)/2)
        else:
            self.max_allowed_tensors = np.int(np.floor(self.available_mem/self.memory[list(np.random.choice(self.memory.keys()))].nbytes))

        eps = 1e-7            
        if np.abs(self.available_mem) < eps:
            print('The memory can not containg any tensor even if it is entirely free (This message can not appear)')
      
        
    def clear(self, full = False):
        if full:
            del self.memory, self.memory_normalized, self.base_tensors
            self.memory = dict()
            self.memory_normalized = dict()
        else:
            memory_new = dict(); memory_new_norm = dict()
            for key, value in self.base_tensors():
               memory_new[key] = self.memory[key]
               memory_new_norm[key] = self.memory_normalized[key]
            del self.memory, self.memory_normalized
            self.memory = memory_new; self.memory_normalized = memory_new_norm
            
    def change_variables(self, increment):
        random_key = np.random.choice(list(self.memory.keys()))
#        print(random_key)
        increment = np.reshape(increment, newshape=self.memory[random_key].shape)
        del self.memory_normalized , self.memory
        self.memory = deepcopy(self.base_tensors); self.memory_normalized = dict()
        for key in self.memory.keys():
#            print(self.memory[key].shape, increment.shape)
            assert np.all(self.memory[key].shape == increment.shape) 
            self.memory[key] = self.memory[key] - increment


    def add(self, label, tensor, normalized = False):
        '''
        Method for addition of a new tensor into the cache. Returns True if there was enough memory and the tensor was save, and False otherwise. 
        '''
        if normalized:
            if len(self.memory_normalized) + len(self.memory) < self.max_allowed_tensors and label not in self.memory_normalized.keys():
                self.memory_normalized[label] = tensor
#                print('Enough space for saved normalized term ', label, tensor.nbytes)
                return True
            elif label in self.memory_normalized.keys():
                eps = 1e-7
                assert np.all(np.abs(self.memory_normalized[label] - tensor) < eps)
#                print('The term already present in normalized cache, no addition required', label, tensor.nbytes)
                return True
            else:
#                print('Not enough space for term ', label, tensor.nbytes)
                return False            
        else:
            if len(self.memory_normalized) + len(self.memory) < self.max_allowed_tensors and label not in self.memory.keys():
                self.memory[label] = tensor
#                print('Enough space for saved unnormalized term ', label, tensor.nbytes)
                return True
            elif label in self.memory.keys():
                eps = 1e-7
                assert np.all(np.abs(self.memory[label] - tensor) < eps)
#                print('The term already present in unnormalized cache, no addition required', label, tensor.nbytes)
                return True
            else:
#                print('Not enough space for term ', label, tensor.nbytes)
                return False
        
    def get(self, label, normalized = False, saved_as = None):
        if normalized:
            try:
                return self.memory_normalized[label]
            except KeyError:
                print('memory keys: ', self.memory_normalized.keys())
                print('fetched label:', label, ' prev. known as ', saved_as)
                raise KeyError('Can not fetch tensor from cache with normalied data')
        else:
            try:
                return self.memory[label]
            except KeyError:
                print('memory keys: ', self.memory.keys())
                print('fetched label:', label, ' prev. known as ', saved_as)
                raise KeyError('Can not fetch tensor from cache with non-normalied data')
    
    @property
    def consumed_memory(self):
        return np.sum([value.nbytes for _, value in self.memory.items()])