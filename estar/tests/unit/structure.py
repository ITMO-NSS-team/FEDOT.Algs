#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:32:54 2021

@author: mike_ubuntu
"""

import numpy as np
from collections import OrderedDict
import sys
sys.path.append('/media/mike_ubuntu/DATA/ESYS/')

import src.globals as global_var
from src.cache.cache import upload_grids

from src.token_family import Token_family
from src.structure import Term, Equation, SoEq
from src.evaluators import trigonometric_evaluator
from src.evo_optimizer import Operator_director

def mock_evaluator(factor):
    return np.ones((10, 10, 10))

class mock_token_family(Token_family):
    def __init__(self, names = [], evaluator = None):
        super().__init__('mock')
        super().use_glob_cache()
        super().set_status(meaningful = True)
      
        mock_equal_params = {'not_power' : 0, 'power' : 0}
        mock_params = OrderedDict([('not_power', (1, 4)), ('power', (1, 1))])
        super().set_evaluator(evaluator)
        super().set_params(names, mock_params, mock_equal_params)   
        
def test_term():
    '''
    Check both (random and determined) ways to initialize the term, check correctness of term value evaluation & terms equality, 
    output format, latex format.
    '''
    global_var.init_caches(set_grids=False)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 25)  
    
    names = ['mock1', 'mock2', 'mock3']
    mock = mock_token_family(names, mock_evaluator)    
    test_term_1 = Term([mock,])
    print(test_term_1.name)
    print('avalable', test_term_1.available_tokens[0].tokens)
    assert test_term_1.available_tokens[0].tokens == names
    assert type(test_term_1) == Term

    test_term_2 = Term([mock,], passed_term = 'mock3')
    print(test_term_2.name)
    assert type(test_term_2) == Term    
    
    test_term_3 = Term([mock,], passed_term = ['mock3', 'mock1'])
    print(test_term_3.name)
    assert type(test_term_3) == Term   
    
    test_term_2.evaluate, test_term_3.evaluate
#    assert False

def test_equation():
    '''
    Use trigonometric identity sin^2 (x) + cos^2 (x) = 1 to generate data, with it: initialize the equation, 
    equation splitting & weights discovery? output format, latex format. Additionally, test evaluator for trigonometric functions
    '''
    global_var.init_caches(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 25)  
    
    director = Operator_director()
    director.operator_assembly()    
    
    x = np.linspace(0, 2*np.pi, 100)
    global_var.grid_cache.memory_usage_properties(obj_test_case=x, mem_for_cache_frac = 25)  
    upload_grids(x, global_var.grid_cache)
    print(global_var.grid_cache.memory)
    names = ['sin', 'cos'] # simple case: single parameter of a token - power
    
    trig_tokens = Token_family('trig')    
    trig_tokens.set_status(unique_specific_token=False, unique_token_type=False, meaningful = True, 
                           unique_for_right_part = False)
    
    equal_params = {'power' : 0, 'freq' : 0.2, 'dim' : 0}

    trig_tokens.use_glob_cache()
    
    trig_params = OrderedDict([('power', (1., 1.)), ('freq', (0.5, 1.5)), ('dim', (0., 0.))])
    trig_tokens.set_params(names, trig_params, equal_params)
    trig_tokens.set_evaluator(trigonometric_evaluator)  
    
    eq1 = Equation(tokens = [trig_tokens,], basic_structure = [], 
                   terms_number = 3, max_factors_in_term = 2)   # Задать возможности выбора вероятностей кол-ва множителей
#    assert False  
    director.constructor.operator.set_sparcity(sparcity_value = 1.)    
    eq1.select_target_idx(target_idx_fixed = 0)
    eq1.select_target_idx(operator = director.constructor.operator)
    print([term.name for term in eq1.structure])
    print(eq1.fitness_value)
    
    print(eq1.described_variables)
    print(eq1.evaluate(normalize = False, return_val = True))
    print('internal:', eq1.weights_internal, 'final:', eq1.weights_final)
    raise NotImplementedError
    
#def test_single_eq_system():
    # Перемещено в интеграционные тесты (wave_test)
    
    
    
#    mock = token_family_mock(names, mock_evaluator)    
    