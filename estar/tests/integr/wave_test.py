#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:44:03 2021

@author: mike_ubuntu
"""

'''
По данным из решения волнового уравнения получаем 1 уравнение (тесты методов src.structure.)

'''

import time
import sys
sys.path.append('/media/mike_ubuntu/DATA/ESYS/')

import numpy as np
import copy
from collections import OrderedDict

from src.moeadd.moeadd import *
from src.moeadd.moeadd_supplementary import *

import src.globals as global_var
import src.structure as structure
from src.supplementary import memory_assesment

from src.evaluators import simple_function_evaluator
from src.cache.cache import Cache, upload_simple_tokens, download_variable
from src.token_family import Evaluator, Token_family, constancy_hard_equality
from src.supplementary import Define_Derivatives, factor_params_to_str
from src.evo_optimizer import Operator_director, Operator_builder
import src.sys_search_operators as operators

import matplotlib.pyplot as plt

def test_single_token_type():
    seed = None
    if type(seed) != type(None):
        np.random.seed(seed)

    folder= sys.path[-1] + 'preprocessing/Wave/'    
    boundary = 15
    print(sys.path)
    u_tensors = download_variable(folder + 'wave_HP.npy', folder + 'Derivatives.npy', boundary, time_axis=0)
    u_names = Define_Derivatives('u', 3, 2)

    global_var.init_caches(set_grids=False)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=u_tensors[0, ...], mem_for_cache_frac = 25)
    upload_simple_tokens(u_names, u_tensors, global_var.tensor_cache)
        
    u_tokens = Token_family('U')
    u_tokens.set_status(unique_specific_token=False, unique_token_type=False, 
                        meaningful = True, unique_for_right_part = True)

    equal_params = {'power' : 0}
    u_token_params = OrderedDict([('power', (1, 2))])    
    u_tokens.set_params(u_names, u_token_params, equal_params)

    u_tokens.use_glob_cache()
#    u_eval_params = {'params_names':['power'], 'params_equality':{'power' : 0}}
    u_tokens.set_evaluator(simple_function_evaluator)

    
    director = Operator_director()
    director.operator_assembly()    
    
    tokens = [u_tokens,]
    pop_constructor = operators.systems_population_constructor(tokens=tokens, terms_number=5, 
                                                               max_factors_in_term=1, 
                                                               eq_search_evo = director.constructor.operator)
    
    equation_creation_params = {'eq_search_iters':2}
    optimizer = moeadd_optimizer(pop_constructor, 7, 20, equation_creation_params, delta = 1/50., neighbors_number = 3)
    evo_operator = operators.sys_search_evolutionary_operator(operators.mixing_xover, 
                                                              operators.gaussian_mutation)

    optimizer.set_evolutionary(operator=evo_operator)
    best_obj = np.concatenate(np.ones([1]), 
                              np.zeros(shape=len([1 for token_family in tokens if token_family.status['meaningful']])))    
    print(best_obj)
    raise NotImplementedError
    
#test_single_token_type()
    
#    del u_tensors

    
    
    
    
#    u_tensors = 
    