#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:27:11 2020

@author: mike_ubuntu
"""

import numpy as np
import copy
import time
import sys
from collections import OrderedDict

from src.supplementary import Define_Derivatives
from src.term import Term
from src.trainer import Equation_Trainer
from src.supplementary_media import Visual

from src.evo_optimizer import Evolutionary_builder
import src.ESTAR_baseline_classes_mp_experiments as baseline #

from src.evo_optimizer import Operator_director, Operator_builder
from src.token_family import Evaluator, Token_family


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
    grid_internal = copy.copy(eval_params['grid'])
    grid_internal = token_params['freq'] * eval_params['grid'][token_params['dim']]
    value = function(grid_internal)
    value = np.power(value, token_params['power'])
    return value
    

def Set_grid(template_matrix, steps):
    assert np.ndim(template_matrix) == len(steps)
    sd_grids = []
    for dim in np.arange(np.ndim(template_matrix)):
        sd_grids.append(np.arange(0, template_matrix.shape[dim]*steps[dim], steps[dim]))
    return np.meshgrid(*sd_grids, indexing = 'ij')
    
if __name__ == '__main__':
    skipped_elems = 10 
    timeslice = (10, -10)

    domains_x = 8; domains_y = 22
    total_domains = domains_x*domains_y
    
    processed_indexes = [52] #np.arange(45, 66)

    for i in range(domains_x):
        for j in range(domains_y):
            cell_idx = j + domains_y*i
            if not cell_idx in processed_indexes:
                continue
            print('Processing domain ', i, j, cell_idx)                
            temp = sys.stdout
            sys.stdout = file = open("Experiments_output/output_logs_"+ str(i) + '_' + str(j) + ".txt", "w")
            der_file_name = 'Preprocessing/Arctic/ssh_1_derivatives_' + str(i) + '_'+str(j) + '.npy'
            u_file_name = 'Preprocessing/Arctic/ssh_1_slice_' + str(i) + '_' + str(j) + '.npy' #, field_domain)
            u = np.load(u_file_name)
            derivatives = np.load(der_file_name) # Пропишите путь к файлу с производными 
            grid = Set_grid(u, steps = (1, 1, 1))
                        
            variables = np.ones((np.ndim(u) + 1 + derivatives.shape[1], ) + u.shape)
            for i in np.arange(np.ndim(u)):
                variables[i, :] = grid[i]
                print((variables[i, :] == grid[i]).all())
                
            variables[np.ndim(u), :] = u
            for i_outer in range(0, derivatives.shape[1]):
                variables[i_outer+1+np.ndim(u)] = derivatives[:, i_outer].reshape(variables[i_outer+1+np.ndim(u)].shape) 
            
            token_names_der = ['x1', 'x2', 'x3'] + list(Define_Derivatives(u.ndim, max_order = 2))
            token_names_trig = ['sin', 'cos']                
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
            trig_token_params = OrderedDict([('power', (1, 1)), ('freq', (1/48., 4)), ('dim', (0, u.ndim))])
            trigonometric_tokens.set_params(token_names_trig, trig_token_params)
            trigonometric_tokens.set_status(unique_token_type=True)            
            
            tokens = [trigonometric_tokens, derivatives_tokens] # 
            basic_terms = []
            
            alphas = np.full(4, 1e-2)
            epochs = 150
            
            for alpha_idx, alpha in np.ndenumerate(alphas): #np.logspace(-3, -1, 4):
                director = Operator_director()
                director.operator_assembly(sparcity = alpha)
                ff_filename = 'graphic_output/ff_'+str(i)+'_'+str(j)+'_'+str(alpha_idx[0])+'.npy'
                
                Trainer = Equation_Trainer(tokens = tokens, basic_terms = [])
                Trainer.parameters_grid(('pop_size', 'eq_len', 'max_factors', 'test_output'), 
                                    (10, 6, 2, False))
                t1 = time.time()                
                Trainer.train(epochs = epochs, evolutionary_operator = director.constructor.operator, fitness_file=ff_filename)
                t2 = time.time()
                print('training time:', t2-t1, ' for a = ', alpha)
            file.close()
            sys.stdout = temp