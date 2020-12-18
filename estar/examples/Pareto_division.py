#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:01:59 2020

@author: mike_ubuntu
"""

import numpy as np
import copy
import time
from collections import OrderedDict
import sys

from src.supplementary import Define_Derivatives
from src.term import Term
from src.trainer import Equation_Trainer

from src.evo_optimizer import Evolutionary_builder
import src.ESTAR_baseline_classes as baseline

from src.evo_optimizer import Operator_director, Operator_builder
from src.token_family import Evaluator, Token_family


# The dataset for the experiment is located at drive https://drive.google.com/drive/folders/1l27FvBTkWPSTX2wgWsLbdLmjkDkBYDdH?usp=sharing

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


def Divide_domain(domain, parts, axis = None):
    if len(axis) == 0:
        return [domain,]
    else:
        cut_domains = []
        temp_cuts = np.split(domain, parts, axis=axis[0])    
        for temp_domain in temp_cuts:
            cut_domains.extend(Divide_domain(temp_domain, parts, axis = axis[1:]))          
        return cut_domains
        

if __name__ == "__main__":
    for noise_idx in np.arange(2, 10):
        t1 = time.time()
        # Add the path to the data of initial function
        u_name = 'Preprocessing/Wave/wave_HP_noised_' + str(noise_idx) + '.npy'
        u_initial = np.load(u_name) 
        u_initial = np.transpose(u_initial, (2, 0, 1)) 
        print(u_initial.shape)
        
        grid = Set_grid(u_initial, steps = (1, 1, 1))    
        
        # Add the path to precalculated derivatives
        derivatives_name = 'Preprocessing/Wave/Derivatives_noised_' + str(noise_idx) + '.npy'
        derivatives = np.load(derivatives_name)  
            
        variables = np.ones((np.ndim(u_initial) + 1 + derivatives.shape[1], ) + u_initial.shape)
        for dim_idx in np.arange(np.ndim(u_initial)):
            variables[dim_idx, :] = grid[dim_idx]
            print((variables[dim_idx, :] == grid[dim_idx]).all())
            
        variables[np.ndim(u_initial), :] = u_initial
        for i_outer in range(0, derivatives.shape[1]):
            variables[i_outer+1+np.ndim(u_initial)] = derivatives[:, i_outer].reshape(variables[i_outer+1+np.ndim(u_initial)].shape) 
        
        skipped_elements = 10
        variables = variables[:, skipped_elements+1:-skipped_elements, skipped_elements+1:-skipped_elements, skipped_elements+1:-skipped_elements]        
        print(variables.shape)
    
        divisions = [2, 4, 5, 8, 10]
    
        for div in divisions:
            temp_domains = Divide_domain(variables, div, axis = [1, 2]) 
        
            
            for domain_idx in range(len(temp_domains)):
                grid = Set_grid(temp_domains[domain_idx][0, :, :, :], steps = (1, 1, 1))    
                token_names_der = ['x1', 'x2', 'x3'] + list(Define_Derivatives(u_initial.ndim, max_order = 2))
                token_names_trig = ['sin', 'cos']                
                simple_functions = {}
                for var_idx in np.arange(temp_domains[domain_idx].shape[0]):
                    simple_functions[token_names_der[var_idx]] = temp_domains[domain_idx][var_idx]    
                
            
                derivatives_tokens = Token_family('Derivatives')
                der_eval_params = {'token_matrices':simple_functions, 'params_names':['power'], 'params_equality':{'power' : 0}}
                derivatives_tokens.set_evaluator(simple_function_evaluator, **der_eval_params)
                der_token_params = OrderedDict([('power', (1, 1))])
                derivatives_tokens.set_params(token_names_der, der_token_params)
                derivatives_tokens.set_status(unique_specific_token=True, mandatory = True, unique_for_right_part = True)
                
                trigonometric_tokens = Token_family('Trigonometric')
                trig_eval_params = {'grid':grid, 'params_names':['power',  'freq', 'dim'], 'params_equality':{'power': 0, 'freq':0.05, 'dim':0}}
                trigonometric_tokens.set_evaluator(trigonometric_evaluator, **trig_eval_params)
                trig_token_params = OrderedDict([('power', (1, 1)), ('freq', (1/48., 4)), ('dim', (0, u_initial.ndim))])
                trigonometric_tokens.set_params(token_names_trig, trig_token_params)
                trigonometric_tokens.set_status(unique_token_type=True)            
                
                tokens = [trigonometric_tokens, derivatives_tokens] # 
                basic_terms = []
                
                
                alphas = np.logspace(-3, -1, 3)
                for alpha_idx, alpha in np.ndenumerate(alphas):
                    print('Processing alpha', alpha_idx)
                    temp = sys.stdout
                    # Add file to the text file, which will be used for output
                    sys.stdout = file = open("Experiments_output/Domain_division/output_logs_wave_division_" + str(noise_idx) + '_' + str(div) + "_" + 
                                             str(domain_idx) + '_' + str(alpha_idx[0]) + ".txt", "w")
                    print('test output')
                    for run_idx in np.arange(10):
                        print('Iteration', run_idx)
                        
                        director = Operator_director()
                        director.operator_assembly(sparcity = alpha, mutation_params = {'indiv_mutation_prob' : 0.8, 'type_probabilities' : [], 'elitism' : 1, 'r_mutation' : 0.2})
                
                        ff_filename = 'graphic_output/wave_'+str(alpha_idx[0])+'_'+str(run_idx)+'.npy'
                        
                        Trainer = Equation_Trainer(tokens = tokens, basic_terms = [])
                        Trainer.parameters_grid(('pop_size', 'eq_len', 'max_factors', 'test_output'), 
                                                (12, 4, 1, False))
                        Trainer.train(epochs = 50, evolutionary_operator = director.constructor.operator, fitness_file=ff_filename)
                        history = np.squeeze(np.array(Trainer.history))
                        
                        history = history.item()
                        print('iteration max time:', np.max(history.iteration_history), 'iteration min time:', np.min(history.iteration_history))
                        print('iteration average:', np.mean(history.iteration_history))
                        t2 = time.time()
                        print('time:', t2-t1)
                    file.close()
                    sys.stdout = temp