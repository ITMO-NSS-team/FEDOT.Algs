#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:31:07 2020

@author: mike_ubuntu
"""

import numpy as np
import copy
from collections import OrderedDict

from src.supplementary import Define_Derivatives
from src.term import Term
from src.trainer import Equation_Trainer

from src.evo_optimizer import Operator_director, Operator_builder
from src.token_family import Evaluator, Token_family

def simple_function_evaluator(token, t_params, eval_params):
    
    '''
    
    Example of the evaluator of token values, appropriate for case of trigonometric functions to be calculated on grid, with results in forms of tensors
    
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
    
    
    assert 'token_matrices' in eval_params
    value = copy.deepcopy(eval_params['token_matrices'][token])
    value = value**(t_params['power'])
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
    eval_params : dict
        Dictionary, containing parameters of the evaluator: in this example, it contains coordinates np.meshgrid with coordinates for points, 
        names of the token parameters (frequency, axis and power). Additionally, the names of the token parameters must be included with specific key 'params_names', 
        and parameters range, for which each of the tokens if consedered as "equal" to another, like sin(1.0001 x) can be assumed as equal to (0.9999 x)
    
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
    return value


def Set_grid(template_matrix, steps):
    assert np.ndim(template_matrix) == len(steps)
    sd_grids = []
    for dim in np.arange(np.ndim(template_matrix)):
        sd_grids.append(np.arange(0, template_matrix.shape[dim]*steps[dim], steps[dim]))
    return np.meshgrid(*sd_grids, indexing = 'ij')
    

if __name__ == '__main__':
    u_initial = np.load('Preprocessing/Fill366/fill366.npy')
    print(u_initial.shape)
    
    derivatives = np.load('Preprocessing/Fill366/Derivatives.npy')
    variables = np.ones((1 + derivatives.shape[1], ) + u_initial.shape)
    variables[0, :] = u_initial
    for i_outer in range(0, derivatives.shape[1]):
        variables[i_outer+1] = derivatives[:, i_outer].reshape(variables[i_outer+1].shape) 

    token_names_der = list(Define_Derivatives(u_initial.ndim, max_order = 1))
    token_names_trig = ['sin', 'cos']
#    token_names = {'derivatives' : token_names_der, 'trigonometric' : token_names_trig}
#    evaluators = {'derivatives' : simple_function_evaluator, 'trigonometric' : trigonometric_evaluator}
    
    step = 4*np.pi/999
    grid = Set_grid(variables[0], steps = (step,))
    simple_functions = {}
    for var_idx in np.arange(variables.shape[0]):
        simple_functions[token_names_der[var_idx]] = variables[var_idx]    
        
    derivatives_tokens = Token_family('Derivatives')
    der_eval_params = {'token_matrices':simple_functions, 'params_names':['power'], 'params_equality':{'power' : 0}}
    derivatives_tokens.set_evaluator(simple_function_evaluator, **der_eval_params)
    der_token_params = OrderedDict([('power', (1, 1))])
    derivatives_tokens.set_params(token_names_der, der_token_params)
    derivatives_tokens.set_status(unique_specific_token=True, mandatory = True, unique_for_right_part = True)
    
    trigonometric_tokens = Token_family('Trigonometric')
    trig_eval_params = {'grid':grid, 'params_names':['power',  'freq', 'dim'], 'params_equality':{'power': 0, 'freq':0.05, 'dim':0}}
    trigonometric_tokens.set_evaluator(trigonometric_evaluator, **trig_eval_params)
    trig_token_params = OrderedDict([('power', (1, 1)), ('freq', (0.9, 1.1)), ('dim', (0, u_initial.ndim))])
    trigonometric_tokens.set_params(token_names_trig, trig_token_params)
    trigonometric_tokens.set_status(unique_token_type=True)
    
    tokens = [trigonometric_tokens, derivatives_tokens]
    basic_terms = []

    director = Operator_director()
    director.operator_assembly(sparcity = 0.5)    

    Trainer = Equation_Trainer(tokens = tokens, basic_terms = basic_terms)
    Trainer.parameters_grid(('pop_size', 'eq_len', 'max_factors', 'test_output'), 
                            (20, 4, 2, False))
    Trainer.train(epochs = 150, evolutionary_operator = director.constructor.operator)    