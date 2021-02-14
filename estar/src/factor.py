#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:16:43 2020

@author: mike_ubuntu
"""

import numpy as np
from collections import OrderedDict

from src.Tokens import TerminalToken
import src.globals as global_var
from src.supplementary import factor_params_to_str

class Factor(TerminalToken):
    def __init__(self, token_name, token_family):
        self.label = token_name
        self.token = token_family
        self.saved = False
        

        if type(global_var.cache) != type(None):
            self.use_cache()
        else:
            self.cache_linked = False
        
    def Set_parameters(self, random = True, **kwargs):
        '''
        
        Avoid periodic parameters (e.g. phase shift) 
        
        '''
        if not random:
            if set(self.token._evaluator.params['params_names']) != set(kwargs.keys()):
                raise Exception('Incorrect/partial set of parameters used to define factor')
            _params = np.empty(len(kwargs))
            assert len(kwargs) == len(self.token.token_params), 'Not all parameters have been declared. Partial randomization TBD'
            for param_idx, param_name, param_val in enumerate(kwargs):
                _params[param_idx] = param_val
                _params_description[param_idx] = dict('name' = param_name, 
                                                          'bounds' = self.token.token_params[param_name]) 
        else:
            _params = np.empty(len(self.token.token_params))#OrderedDict()
            for param_idx, param_name, param_range in self.token.token_params.items():
                if param_name != 'power':
                    _params[param_idx] = (np.random.randint(param_range[0], param_range[1]) if isinstance(param_range[0], int) 
                    else np.random.uniform(param_range[0], param_range[1])) if param_range[1] > param_range[0] else param_range[0]
                else:
                    _params[param_idx] = 1
                _params_description[param_idx] = dict('name' = param_name, 
                                                      'bounds' = self.token.token_params[param_name]) 
        super().__init__(number_params = _params.size, params_description = _params_description, 
                         params = _params)
        
    def use_cache(self):      
        self.cache_linked = True
        
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        elif self.label != other.label:
            return False
        elif any([abs(self.params[key] - other.params[key]) > self.token._evaluator.params['params_equality'][key] for key in self.params.keys()]):
            return False
        else:
            return True
    
    def __call__(self):
        '''
        
        Return vector of evaluated values
        
        '''
        raise NotImplementedError('Delete me')
        return self.token.evaluate(self)    
    
    def evaluate(self): # Переработать/удалить __call__, т.к. его функции уже тут
#        print(self.cache_linked, self.label)

        assert self.cache_linked
        if self.saved:
            return global_var.cache.get(self.cache_label)
        else:
            value = self.token.evaluate(self)
            if self.params.size > 1:
                raise NotImplementedError('Currently cache processing is implemented only for the single parameter token')
            if self.params.size == 1:
                self.saved = global_var.cache.add(self.cache_label, value)
            return value

    @property
    def cache_label(self):
        cache_label = factor_params_to_str(self)
#        print('construced_')
        return cache_label
            
    @property
    def name(self):
        form = self.label + '{' 
        for param_idx in enumerate(self.params): # param_name, param_val 
            form += list(self.params.items())[param_idx][0] + ': ' + str(self.params[param_idx])
            if param_idx < len(self.params.items()) - 1:
                form += ', '
        form += '}'
        return form

#
#class DerivPlaceholder(object):
#    def __init__(self, axis, order):
#        self.axis = axis
#        self.order = order
#
#
#class DerivFactor(Factor, DerivPlaceholder):
#    def __init__(self, token_name, token, axis = 0, deriv_order = 0):
#        super(Factor, self).__init__(token_name, token)
#        super(DerivPlaceholder, self).__init__(axis, deriv_order)
#     
#    def __eq__(self, other):
#        if type(self) == self(other):
#            return False
#        elif self.label != other.label or self.axis != other.axis:
#            return False
#        elif any([abs(self.params[key] - other.params[key]) > self.token._evaluator.params['params_equality'][key] for key in self.params.keys()]):
#            return False
#        else:
#            return True        
##    
#    
#class Factor_m(object):   
#    def Set_parameters(self, args_array):
#        '''
#        
#        Avoid periodic parameters (e.g. phase shift) 
#        
#        '''
#        if np.size(self.token._evaluator.params) != np.size(args_array):
#            raise Exception('Incorrect/partial set of parameters used to define factor')
#        self.params = args_array
#        
#        
#    def __eq__(self, other):
#        if self.label != other.label:
#            return False
#        elif any([abs(self.params[idx] - other.params[idx]) > self.token._evaluator.params['params_equality'][idx] for idx in np.arange(self.params.size)]):
#            return False
#        else:
#            return True
#
#    
#    def __call__(self):
#        '''
#        
#        Return vector of evaluated values
#        
#        '''
#        return self.token.evaluate(self.label, self.params)
#    
#    
#    @property
#    def text_form(self):
#        form = self.label + '{' 
#        for param_idx in range(len(self.params.items())): # param_name, param_val 
#            form += list(self.params.items())[param_idx][0] + ': ' + str(list(self.params.items())[param_idx][1])
#            if param_idx < len(self.params.items()) - 1:
#                form += ', '
#        form += '}'
#        return form