#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Feb 14 13:11:46 2020

@author: mike_ubuntu
"""

import numpy as np
import collections

from src.supplementary import Define_Derivatives
from src.term import normalize_ts,Term
from src.trainer import Equation_Trainer



def derivative_evaluator(term, normalize, eval_params):
    
    '''
    
    Example of the evaluator of token values, appropriate for case of derivatives with pre-calculated values, defined on grid, that take form of tensors
    
    Parameters
    ----------
    term : term.Term, or numpy.ndarray
        Object for term of the equation, or its gene, for which the evaluation is done; necessary for the evaluation.
    eval_params : dict
        Dictionary, containing parameters of the evaluator: in this example, they are 
        'token matrices' : list/numpy.martix of token (derivatives) values on the grid, 'parameter_indexes' : dictionary of orders of token parameters during the encoding. 
        In simplest case of only power parameter: 'parameter_indexes':{'power':0}.
    
    Returns
    ----------
    value : numpy.ndarray
        Vector of the evaluation of the token values, that shall be used as target, or feature during the LASSO regression.
        
    '''
    
    assert 'token_matrices' in eval_params and 'parameter_indexes' in eval_params
    if type(term) == Term:
        term = term.gene
    token_matrices = eval_params['token_matrices']
    value = np.copy(token_matrices[0])
    for var_idx in np.arange(term.shape[0]):
        power = (term[var_idx + eval_params['parameter_indexes']['power']])
        value *= eval_params['token_matrices'][int(var_idx / (float(eval_params['parameter_indexes']['power']+1)))] ** int(power)
    if normalize:
        value = normalize_ts(value)
    value = value.reshape(np.prod(value.shape))
    return value    
   

if __name__ == '__main__':
    u_initial = np.load('Preprocessing/Wave_HP/wave_HP.npy')
    u_initial = np.transpose(u_initial, (2, 0, 1))
    print(u_initial.shape)
    
    derivatives = np.load('Preprocessing/Wave_HP/Derivatives.npy')
    variables = np.ones((2 + derivatives.shape[1], ) + u_initial.shape)
    variables[1, :] = u_initial
    for i_outer in range(0, derivatives.shape[1]):
        variables[i_outer+2] = derivatives[:, i_outer].reshape(variables[i_outer+2].shape) 
                
    skipped_elems = 15 
    timeslice = (skipped_elems, -skipped_elems)
    
    token_names = Define_Derivatives(u_initial.ndim, max_order = 2)
    print(token_names)
    token_parameters = collections.OrderedDict([('power', (0, 3))])
    variables = variables[:, timeslice[0]:timeslice[1], skipped_elems:-skipped_elems, skipped_elems:-skipped_elems]
    basic_terms = [{'1':{'power':1}},
                   {'1':{'power':1},  'u':{'power':1}}]        
    

    Trainer = Equation_Trainer(tokens = token_names, token_params = token_parameters, evaluator = derivative_evaluator, 
                               evaluator_params = {'token_matrices':variables, 'parameter_indexes':{'power':0}}, basic_terms = basic_terms)
    Trainer.Parameters_grid(('alpha', 'a_proc', 'r_crossover', 'r_param_mutation', 'r_mutation', 'mut_chance', 'pop_size', 'eq_len', 'max_factors'), 
                            ((0.1, 0.2, 3), 0.2, 0.6, 0.8, 0.5, 0.8, 20, 6, 2))
    Trainer.Train(epochs = 50)
    
