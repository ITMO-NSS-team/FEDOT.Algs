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
    def __init__(self, token_name, status, randomize = False):#, token_family, randomize = False):
        self.label = token_name
        self.status = status
        self.saved = False
        self.grid_set = False

        if type(global_var.tensor_cache) != type(None):
            self.use_cache()
        else:
            self.cache_linked = False

        if self.status['requires_grid']:
            self.use_grids_cache()

        if randomize:
            self.Set_parameters()

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status_dict):
        self._status = status_dict
        
    def Set_parameters(self, random = True, **kwargs):
        '''
        
        Avoid periodic parameters (e.g. phase shift) 
        
        '''
        _params_description = {}
        if not random:
#            assert set(self.token._evaluator.params['params_names']) != set(kwargs.keys()), 'Incorrect/partial set of parameters used to define factor'
#                raise Exception()
            _params = np.empty(len(kwargs))
            assert len(kwargs) == len(self.token.token_params), 'Not all parameters have been declared. Partial randomization TBD'
            for param_idx, param_info in enumerate(kwargs.items()): #param_name, param_val 
                _params[param_idx] = param_info[1]
                _params_description[param_idx] = {'name' : param_info[0], 
                                                          'bounds' : self.token.token_params[param_info[0]]} 
        else:
            _params = np.empty(len(self.token.token_params))#OrderedDict()
            for param_idx, param_info in enumerate(self.token.token_params.items()):
                if param_info[0] != 'power':
#                    if param_info[1][1] > param_info[1][0]:
#                        if isinstance(param_info[1][0], int):
#                            _params[param_idx] = np.random.randint(param_info[1][0], param_info[1][1])
#                            print('_params[param_idx] is int', _params[param_idx])
#                        else:
#                            _params[param_idx] = np.random.uniform(param_info[1][0], param_info[1][1])
#                            print('_params[param_idx] is float', _params[param_idx])
#                    else:
#                        _params[param_idx] = param_info[1][0]
                    _params[param_idx] = (np.random.randint(param_info[1][0], param_info[1][1]) if isinstance(param_info[1][0], int) 
                    else np.random.uniform(param_info[1][0], param_info[1][1])) if param_info[1][1] > param_info[1][0] else param_info[1][0]
#                    print('random value with type', np.random.randint(param_info[1][0], param_info[1][1]), _params[param_idx], type(_params[param_idx]))
                else:
                    _params[param_idx] = 1
                _params_description[param_idx] = {'name' : param_info[0], 
                                                      'bounds' : param_info[1]} 
        super().__init__(number_params = _params.size, params_description = _params_description, 
                         params = _params)
        if not self.grid_set:
            self.use_grids_cache()
        
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        elif self.label != other.label:
            return False
        elif any([abs(self.params[idx] - other.params[idx]) > self.token.equality_ranges[self.params_description[idx]['name']] 
                                                for idx in np.arange(self.params.size)]):
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
            return global_var.tensor_cache.get(self.cache_label)
        else:
            value = self.token.evaluate(self)
#            print(self.cache_label)
#            if self.params.size > 1:
#                raise NotImplementedError('Currently cache processing is implemented only for the single parameter token')
#            if self.params.size == 1:
            self.saved = global_var.tensor_cache.add(self.cache_label, value)
            return value

    @property
    def cache_label(self):
        cache_label = factor_params_to_str(self)
        return cache_label

    @property
    def name(self):
        form = self.label + '{' 
        for param_idx, param_info in self.params_description.items(): # param_name, param_val 
#            print(param_idx, param_info)
            form += param_info['name'] + ': ' + str(self.params[param_idx])
            if param_idx < len(self.params_description.items()) - 1:
                form += ', '
        form += '}'
        return form

    @property
    def grids(self):
        return global_var.grid_cache.get(str(self.grid_idx))

    def use_grids_cache(self):
        dim_param_idx = np.inf
        dim_set = False
        for param_idx, param_descr in self.params_description.items():
            if param_descr['name'] == 'dim': 
                dim_param_idx = param_idx
                dim_set = True
#        if dim_set:
#            assert self.params[name_param_idx] != np.inf, 'No dimension parameter for grid selection'
        self.grid_idx = int(self.params[dim_param_idx]) if dim_set else 0
#        else:
#            self.grid_idx = 0
        self.grid_set = True
    
    def use_cache(self):      
        self.cache_linked = True