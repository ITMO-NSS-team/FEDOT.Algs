#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:09:38 2021

@author: mike_ubuntu
"""

import time
from src.supplementary import memory_assesment

import numpy as np
import copy
from collections import OrderedDict

from moeadd.moeadd import *
from moeadd.moeadd_supplementary import *


#from src.struct import SoEq

import src.globals as global_var
import src.structure as structure
from src.cache import Cache, upload_simple_tokens, download_variable
from src.token_family import Evaluator, Token_family, constancy_hard_equality
from src.supplementary import Define_Derivatives, factor_params_to_str
from src.evo_optimizer import Operator_director, Operator_builder
import src.sys_search_operators as operators

import matplotlib.pyplot as plt

seed = 14
np.random.seed(seed)

def simple_function_evaluator(factor):
    
    '''
    
    Example of the evaluator of token values, appropriate for case of trigonometric functions to be calculated on grid, with results in forms of tensors
    
    OLD DESCRIPTION
    
    Parameters
    ----------
    token: {'u', 'du/dx', ...}
        symbolic form of the function to be evaluated: 
    token_params: dictionary: key - symbolic form of the parameter, value - parameter value
        names and values of the parameters for trigonometric functions: amplitude, frequency & dimension
    eval_params : dict
        Dictionary, containing parameters of the evaluator: in this example, it contains coordinates np.ndarray with pre-calculated values of functions, 
        names of the token parameters (power). Additionally, the names of the token parameters must be included with specific key 'params_names', 
        and parameters range, for which each of the tokens if consedered as "equal" to another, like sin(1.0001 x) can be assumed as equal to (0.9999 x)
    
    Returns
    ----------
    value : numpy.ndarray
        Vector of the evaluation of the token values, that shall be used as target, or feature during the LASSO regression.
        
    '''
    
#    assert 'token_matrices' in eval_params
#    print(token)
    if factor.params['power'] == 1:
        value = global_var.cache.get(factor.cache_label)
        return value
    else:
        value = global_var.cache.get(factor_params_to_str(factor, set_default_power = True))
        value = value**(factor.params['power'])
        return value

#def name_and_tensor_dict(names, tensors):
#    res = {}
#    for var_idx in np.arange(tensors.shape[0]):
#        res[names[var_idx]] = tensors[var_idx]        
#    return res
    

if __name__ == '__main__':    
    folder = 'preprocessing/wolfram/data/'    

#    memory_assesment()
#    time.sleep(5)

    XX = np.load(folder + 'xx.npy')
    YY = np.load(folder + 'yy.npy')
    TT = np.load(folder + 'tt.npy')

#    memory_assesment()
#    print('Downloaded grid matrices', XX.shape, XX.nbytes)
#    time.sleep(5)
    
    boundary = 15    
    global_var.init()
#    global_var.cache.set_neccessary_matrices(neccessary_matrices=['p', 'u', 'v'])
    global_var.cache.memory_usage_properties(obj_test_case=XX, mem_for_cache_frac = 25)    
    
#    memory_assesment()
#    print('Declared global cache')
#    time.sleep(5)
    
    u_tensors = download_variable(folder + 'u_wolfram.npy', folder + 'u_derivs_w.npy', boundary, time_axis=0)
#    u_names = Define_Derivatives('u', u_tensors[0].ndim, int((u_tensors.shape[0]-1)/u_tensors[0].ndim))   
    u_names = Define_Derivatives('u', 3, 3)       
    upload_simple_tokens(u_names, u_tensors, global_var.cache)
#    u_tensors_named = name_and_tensor_dict(u_names, u_tensors)
    del u_tensors    
    
    v_tensors = download_variable(folder + 'v_wolfram.npy', folder + 'v_derivs_w.npy', boundary, time_axis=0)
#    v_names = Define_Derivatives('v', v_tensors[0].ndim, int((v_tensors.shape[0]-1)/v_tensors[0].ndim))
    v_names = Define_Derivatives('v', 3, 3)       
    upload_simple_tokens(v_names, v_tensors, global_var.cache) 
    del v_tensors      
#    v_tensors_named = name_and_tensor_dict(v_names, v_tensors)

    p_tensors = download_variable(folder + 'p_wolfram.npy', folder + 'p_derivs_w.npy', boundary, time_axis=0)
#    p_names = Define_Derivatives('p', p_tensors[0].ndim, int((p_tensors.shape[0]-1)/p_tensors[0].ndim))      
    p_names = Define_Derivatives('p', 3, 3)       
#    p_names = p_names[:-3]; p_tensors = p_tensors[:-3, :, :, :]
    upload_simple_tokens(p_names, p_tensors, global_var.cache)

    del p_tensors
#    p_tensors_named = name_and_tensor_dict(p_names, p_tensors)
        
    memory_assesment()
    time.sleep(5)
    
    u_tokens = Token_family('U')
    u_tokens.use_glob_cache()
    u_eval_params = {'params_names':['power'], 'params_equality':{'power' : 0}}
    u_tokens.set_evaluator(simple_function_evaluator, **u_eval_params)
    u_token_params = OrderedDict([('power', (1, 1))])
    u_tokens.set_params(u_names, u_token_params)
    u_tokens.set_status(unique_specific_token=True, unique_token_type=True, meaningful = True, unique_for_right_part = True)
    
    
    v_tokens = Token_family('V')
    v_tokens.use_glob_cache()
    v_eval_params = {'params_names':['power'], 'params_equality':{'power' : 0}}
    v_tokens.set_evaluator(simple_function_evaluator, **v_eval_params)
    v_token_params = OrderedDict([('power', (1, 1))])
    v_tokens.set_params(v_names, v_token_params)
    v_tokens.set_status(unique_specific_token=True, unique_token_type=True, meaningful = True, unique_for_right_part = True)
    
    p_tokens = Token_family('P')
    p_tokens.use_glob_cache()
    p_eval_params = {'params_names':['power'], 'params_equality':{'power' : 0}}
    p_tokens.set_evaluator(simple_function_evaluator, **p_eval_params)
    p_token_params = OrderedDict([('power', (1, 1))])
    p_tokens.set_params(p_names, p_token_params)
    p_tokens.set_status(unique_specific_token=True, unique_token_type=True, meaningful = True, unique_for_right_part = True)

#    u_tokens.chech_constancy(constancy_hard_equality, epsilon = 1e-5)
#    v_tokens.chech_constancy(constancy_hard_equality, epsilon = 1e-5)
#    p_tokens.chech_constancy(constancy_hard_equality, epsilon = 1e-5)

#    memory_assesment()
#    time.sleep(5)
#    
#    test_equation = structure.Equation([u_tokens, v_tokens, p_tokens], [], 6, 2)


    director = Operator_director()
    director.operator_assembly()    

#    memory_assesment()
#    time.sleep(5)
#    
    test_system = structure.SoEq([u_tokens, v_tokens, p_tokens], 16, 1, sparcity = (0.05, 0.05, 0.05), eq_search_iters = 30)
    test_system.set_eq_search_evolutionary(director.constructor.operator)
#    test_system.set_cache(cache)
    print('Equation search operator set. Type - ', type(test_system.eq_search_evolutionary_operator))
    test_system.create_equations(population_size=16)
    for equation in test_system.structure:
        print(equation.text_form)
    
    
    

#    sys_search_operator = operators.sys_search_evolutionary_operator()
#    
#    population_constructor = operators.systems_population_constructor(tokens, terms_number,
#                                                                      max_factors, eq_search_iters, equations_num)
#    
#    optimizer = moeadd_optimizer(pop_constr, 20, 100, None, delta = 1/50., neighbors_number = 5)
