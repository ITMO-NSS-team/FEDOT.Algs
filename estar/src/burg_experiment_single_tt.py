#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:00:50 2020

@author: mike_ubuntu
"""

import numpy as np
import copy
from collections import OrderedDict

from src.supplementary import Define_Derivatives
from src.term import normalize_ts,Term
from src.trainer import Equation_Trainer
from src.equation import Equation

def simple_function_evaluator(token, t_params, eval_params):
    
    '''
    
    Example of the evaluator of token values, appropriate for case of derivatives with pre-calculated values, defined on grid, that take form of tensors
    
    Parameters
    ----------
    token : string
        Symbolic name of the token in term of equation, for which the evaluation is done.
    token_params : dict
        Dictionary, containing parameters of the evaluator: in this example, they are 
        'token matrices' : list/numpy.martix of token (derivatives) values on the grid, 'parameter_indexes' : dictionary of orders of token parameters during the encoding. 
        In simplest case of only power parameter: 'parameter_indexes':{'power':0}.

    
    Returns
    ----------
    value : numpy.ndarray
        Vector of the evaluation of the token values, that shall be used as target, or feature during the LASSO regression.
        
    '''
    
    assert 'token_matrices' in eval_params
#    print(t_params)
    value = copy.deepcopy(eval_params['token_matrices'][token])
    value = value**(t_params['power'])
#    if normalize:
#        value = normalize_ts(value)
#    if normalize and np.ndim(value) != 1:
#        value = normalize_ts(value)    
#    elif normalize and np.ndim(value) == 1 and np.std(value) != 0:
#        value = (value - np.mean(value))/np.std(value)
#        print(np.mean(value), np.std(value))        
#    value = value.reshape(value.size)
    return value    


def trigonometric_evaluator(token, token_params, eval_params):
    
    '''
    
    Example of the evaluator of token values, appropriate for case of trigonometric functions to be calculated on grid, with results in forms of tensors
    
    Parameters
    ----------
    token: {'sin', 'cos'}
        symbolic form of the function to be evaluated: 
    token_params: dictionary: key - symbolic form of the parameter, value - parameter value
        names and values of the parameters for trigonometric functions: amplitude, frequency & dimension
    normalize: bool
        If True, values will be standartized (case for sparse regression, designed to filter valuable terms). If False, use original values 
        (case for final linear regression)
    eval_params : dict
        Dictionary, containing parameters of the evaluator: in this example, it contains coordinates np.meshgrid with coordinates for points
    
    Returns
    ----------
    value : numpy.ndarray
        Vector of the evaluation of the token values, that shall be used as target, or feature during the LASSO regression.
        
    '''
    
    assert 'grid' in eval_params
    trig_functions = {'sin' : np.sin, 'cos' : np.cos}
    function = trig_functions[token]
    grid_function = np.vectorize(lambda *args: function(token_params['freq']*args[token_params['dim']])**token_params['power'])
    value = grid_function(*eval_params['grid'])
#    if normalize and np.ndim(eval_params['grid'][0]) != 1:
#        value = normalize_ts(value)    
#    elif normalize and np.ndim(eval_params['grid'][0]) == 1 and np.std(value, axis=0) != 0:       
#        value = (value - np.mean(value, axis=0))/np.std(value, axis=0)
#    value = value.reshape(value.size)
    return value
    

def Set_grid(template_matrix, steps):
    assert np.ndim(template_matrix) == len(steps)
    sd_grids = []
    for dim in np.arange(np.ndim(template_matrix)):
        sd_grids.append(np.arange(0, template_matrix.shape[dim]*steps[dim], steps[dim]))
    return np.meshgrid(*sd_grids, indexing = 'ij')
    

if __name__ == '__main__':
    u_initial = np.load('Preprocessing/burgers/burgers.npy')
    u_initial = u_initial[0:100, :, :] #np.transpose(u_initial, (2, 0, 1))
    print(u_initial.shape)
    
    derivatives = np.load('Preprocessing/burgers/Derivatives.npy')
    variables = np.ones((1 + derivatives.shape[1], ) + u_initial.shape)
    variables[0, :] = u_initial
    for i_outer in range(0, derivatives.shape[1]):
        variables[i_outer+1] = derivatives[:, i_outer].reshape(variables[i_outer+1].shape) 

    skipped_elems = 25 
    timeslice = (skipped_elems, -skipped_elems)
    variables = variables[:, timeslice[0]:timeslice[1]]#, skipped_elems:-skipped_elems, skipped_elems:-skipped_elems]    
    
    token_names_der = list(Define_Derivatives(u_initial.ndim, max_order = 2))
    token_names_trig = ['sin', 'cos']
    token_names = {'derivatives' : token_names_der}#, 'trigonometric' : token_names_trig}
    evaluators = {'derivatives' : simple_function_evaluator}#, 'trigonometric' : trigonometric_evaluator}
    
    step = 4*np.pi/999
    grid = Set_grid(variables[0], steps = (1/1000., 1/10., 1/10.))
    simple_functions = {}
    for var_idx in np.arange(variables.shape[0]):
        simple_functions[token_names_der[var_idx]] = variables[var_idx]    
    
    eval_params = {'derivatives':{'token_matrices':simple_functions, 'params_names':['power'], 'params_equality':{'power' : 0}, 'unique' : True}}#, 
#                   'trigonometric':{'grid':grid, 'params_names':['power',  'freq', 'dim'], 'params_equality':{'power': 0, 'freq':0.15, 'dim':0}, 
#                                    'unique' :  False}}
    token_params = {'derivatives' : OrderedDict([('power', (1, 2))])}#, 
#                    'trigonometric' : OrderedDict([('power', (1, 1)), 
#                                                   ('freq', (0.7, 1.5)),
#                                                   ('dim', (0, u_initial.ndim))])}
    token_status = {'derivatives':{'single', 'mandatory'}}#, 'trigonometric':{'single'}}

    basic_terms = []



    Trainer = Equation_Trainer(tokens = token_names, token_params = token_params,
                               evaluator = {'derivatives':simple_function_evaluator},#, 'trigonometric':trigonometric_evaluator}, 
                               evaluator_params = eval_params, basic_terms = basic_terms, token_status = token_status)
    Trainer.Parameters_grid(('alpha', 'a_proc', 'r_crossover', 'r_param_mutation', 'r_mutation', 'mut_chance', 'pop_size', 'eq_len', 'max_factors', 'test_output'), 
                            ((0.01, 0.3, 10), 0.2, 0.6, 0.8, 0.5, 0.8, 20, 4, 2, False))
    Trainer.Train(epochs = 250)    