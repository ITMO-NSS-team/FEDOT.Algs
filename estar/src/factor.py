#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:16:43 2020

@author: mike_ubuntu
"""


class Factor(object):
    def __init__(self, token_name, token):
        self.label = token_name
        self.token = token
   
    def Set_parameters(self, **kwargs):
        '''
        
        Avoid periodic parameters (e.g. phase shift) 
        
        '''
        if set(self.token._evaluator.params['params_names']) != set(kwargs.keys()):
            raise Exception('Incorrect/partial set of parameters used to define factor')
        self.params = kwargs
        
        
    def __eq__(self, other):
        if self.label != other.label:
            return False
        elif any([abs(self.params[key] - other.params[key]) > self.token._evaluator.params['params_equality'][key] for key in self.params.keys()]):
            return False
        else:
            return True

    
    def __call__(self):
        '''
        
        Return vector of evaluated values
        
        '''
        return self.token.evaluate(self.label, self.params)
    
    
    @property
    def text_form(self):
        form = self.label + '{' 
        for param_idx in range(len(self.params.items())): # param_name, param_val 
            form += list(self.params.items())[param_idx][0] + ': ' + str(list(self.params.items())[param_idx][1])
            if param_idx < len(self.params.items()) - 1:
                form += ', '
        form += '}'
        return form