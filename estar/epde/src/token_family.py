#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:39:18 2020

@author: mike_ubuntu
"""

import numpy as np

import epde.src.globals as global_var
from epde.src.factor import Factor

def constancy_hard_equality(tensor, epsilon = 1e-7):
    print(np.abs(np.max(tensor) - np.min(tensor)), epsilon, type(np.abs(np.max(tensor) - np.min(tensor))),  type(epsilon))
    return np.abs(np.max(tensor) - np.min(tensor)) < epsilon
    
class Evaluator(object):
    """
    Class for evaluator of token (factor of the term in the sought equation) values with arbitrary function
       
    Attributes
    ----------
 
    _evaluator : function
        a function, which returns the vector of token values, evaluated on the studied area;
    params : dict
        dictionary, containing parameters of the evaluator (like grid, on which the function is evaluated or matrices of pre-calculated function)


    Methods
    ----------
    
    set_params(**params)
        set the parameters of the evaluator, using keyword arguments
        
    apply(token, token_params)
        apply the defined evaluator to evaluate the token with specific parameters
    """
    def __init__(self, eval_function, eval_kwargs_keys):
        self._evaluator = eval_function
        self.eval_kwargs_keys = eval_kwargs_keys
#    def set_params(self, **params):
#        """
#        Set the parameters of the evaluator, using keyword arguments
#        """
#        self.params = params
        
    def apply(self, token, structural = False, **kwargs):
        """
        Apply the defined evaluator to evaluate the token with specific parameters.
        
        Parameters:
        ----------
        token_label : string
            symbolic label of the specific token, e.g. 'cos';
        
        token_params : dict
            dictionary with keys, naming the token parameters (such as frequency, axis and power for trigonometric function) 
            and values - specific values of corresponding parameters.
            
        Raises:
        ----------
        TypeError
            If the evaluator could not be applied to the token.
        
        """
        assert list(kwargs.keys()) == self.eval_kwargs_keys
#        try:
        return self._evaluator(token, structural, **kwargs)
#        except TypeError:
#            raise TypeError('Wrong parameters passed into the evaluator.')
            

class Token_family(object):
    """
    Class for the type (family) of tokens, from which the tokens are taken as factors in the terms of the equation
    
    Attributes:
    -----------
    type : string
        the symbolic name of the token family (e.g. 'logarithmic', 'trigonometric', etc.)
        
    status : dict
        dictionary, containing markers, describing the token properties. Key - property, value - bool variable:
            
        'mandatory' - if True, a token from the family must be present in every term; 
        
        'unique_token_type' - if True, only one token of the family can be present in the term; 
        
        'unique_for_right_part' - if True, the tokens, present in the "right part" of the equation, can not be present in the terms of the "left part". 
        Recommended to select "True", if any token of the familiy can have 0 values on the majority of studied area;
    
        'unique_specific_token' - if True, a specific token can be present only once per term;
        
    _evaluator : Evaluator object
        Evaluator, which is used to get values of the tokens from that family;
    
    tokens : list of strings
        List of function names, describing all of the functions, belonging to the family. E.g. for 'trigonometric' token type, this list will be ['sin', 'cos']
   
    token_params : OrderedDict
        Available range for token parameters. Ordered dictionary with key - token parameter name, and value - tuple with 2 elements:
        (lower boundary, higher boundary), while type of boundaries describes the avalable token params: 
        if int - the parameters will be integer, if float - float.
        
    Methods:
    -----------
    set_status(mandatory = False, unique_specific_token = False, unique_token_type = False, unique_for_right_part = False)
        Method to set the markers of the token status;
        
    set_params(tokens, token_params)
        Method to set the list of tokens, present in the family, and their parameters;
        
    set_evaluator(eval_function, **eval_params)
        Method to set the evaluator for the token family & its parameters;
        
    test_evaluator()
        Method to test, if the evaluator and tokens are set properly
        
    evaluate(token, token_params)
        Method, which uses the specific token evaluator to evaluate the passed token with its parameters
    
    """
    def __init__(self, token_type):
        """
        Initialize the token family;
        
        Parameters:
        -----------
        token_type : string
            The name of the token family; must be unique among other families.
        """
        self.type = token_type
        self.evaluator_set = False; self.params_set = False; self.cache_set = False
        
    def set_status(self, meaningful = False, s_and_d_merged = True, 
                   unique_specific_token = False, unique_token_type = False, 
                   unique_for_right_part = False, requires_grid = False):
        """
        Set the status of the elements of the token family; 
        
        Parameters:
        -----------            
        mandatory : Bool
            if True, a token from the family must be present in every term; 
        
        unique_token_type : Bool
            if True, only one token of the family can be present in the term; 
        
        unique_for_right_part : Bool
            if True, the tokens, present in the "right part" of the equation, can not be present in the terms of the "left part". 
            Recommended to select "True", if any token of the familiy can have 0 values on the majority of studied area;
    
        unique_specific_token : Bool
            if True, a specific token can be present only once per term;
        """
        self.status = {}
        self.status['meaningful'] = meaningful
        self.status['structural_and_defalut_merged'] = s_and_d_merged
        self.status['unique_specific_token'] = unique_specific_token
        self.status['unique_token_type'] = unique_token_type
        self.status['unique_for_right_part'] = unique_for_right_part
        self.status['requires_grid'] = requires_grid

    def set_params(self, tokens, token_params, equality_ranges):
        """
        Define the token family with list of tokens and their parameters
        
        Parameters:
        -----------
        tokens : list of strings
            List of function names, describing all of the functions, belonging to the family. E.g. for 'trigonometric' token type, 
            this list will be ['sin', 'cos']
       
        token_params : OrderedDict
            Available range for token parameters. Ordered dictionary with key - token parameter name, and value - tuple with 2 elements:
            (lower boundary, higher boundary), while type of boundaries describes the avalable token params: 
            if int - the parameters will be integer, if float - float.

        Example:
        ----------
        >>> token_names_trig = ['sin', 'cos']        
        >>> trig_token_params = OrderedDict([('power', (1, 1)), ('freq', (0.9, 1.1)), ('dim', (0, u_initial.ndim))])
        >>> trigonometric_tokens.set_params(token_names_trig, trig_token_params)
        
        """
        self.tokens = tokens; self.token_params = token_params
        self.params_set = True
        self.equality_ranges = equality_ranges
        if self.evaluator_set:
            self.test_evaluator()

    def set_evaluator(self, eval_function, eval_kwargs_keys = []):#, **eval_params):    #Test, if the evaluator works properly
        """
        Define the evaluator for the token family and its parameters
        
        Parameters:
        ------------
        eval_function : function
            Function, used in the evaluator
            
        **eval_params : keyword arguments
            The parameters for evaluator; must contain params_names (names of the token parameters) &
            param_equality (for each of the token parameters, range in which it considered as the same), 
        
        
        Example:
        -----------
        >>> def trigonometric_evaluator(token, token_params, eval_params):
        >>>     
        >>>     '''
        >>>     
        >>>     Example of the evaluator of token values, appropriate for case of trigonometric functions to be calculated on grid, with results in forms of tensors
        >>>     
        >>>     Parameters
        >>>     ----------
        >>>     token: {'sin', 'cos'}
        >>>         symbolic form of the function to be evaluated: 
        >>>     token_params: dictionary: key - symbolic form of the parameter, value - parameter value
        >>>         names and values of the parameters for trigonometric functions: amplitude, frequency & dimension
        >>>     eval_params : dict
        >>>         Dictionary, containing parameters of the evaluator: in this example, it contains coordinates np.meshgrid with coordinates for points, 
        >>>         names of the token parameters (frequency, axis and power). Additionally, the names of the token parameters must be included with specific key 'params_names', 
        >>>         and parameters range, for which each of the tokens if consedered as "equal" to another, like sin(1.0001 x) can be assumed as equal to (0.9999 x)
        >>>     
        >>>     Returns
        >>>     ----------
        >>>     value : numpy.ndarray
        >>>         Vector of the evaluation of the token values, that shall be used as target, or feature during the LASSO regression.
        >>>         
        >>>     '''
        >>>     
        >>>     assert 'grid' in eval_params
        >>>     trig_functions = {'sin' : np.sin, 'cos' : np.cos}
        >>>     function = trig_functions[token]
        >>>     grid_function = np.vectorize(lambda *args: function(token_params['freq']*args[token_params['dim']])**token_params['power'])
        >>>     value = grid_function(*eval_params['grid'])
        >>>     return value
        >>> 
        >>> der_eval_params = {'token_matrices':simple_functions, 'params_names':['power'], 'params_equality':{'power' : 0}}
        >>> trig_eval_params = {'grid':grid, 'params_names':['power',  'freq', 'dim'], 'params_equality':{'power': 0, 'freq':0.05, 'dim':0}}
        >>> trigonometric_tokens.set_evaluator(trigonometric_evaluator, **trig_eval_params)
        
        """
        self._evaluator = Evaluator(eval_function, eval_kwargs_keys)
#        self._evaluator.set_params(**eval_params)
        self.evaluator_set = True
        if self.params_set:
            self.test_evaluator()

    def use_glob_cache(self):
        self.cache_set = True

#    def use_grid_cache(self):
#        self.grid_cache_set = True

    def test_evaluator(self):
        """
        Method to test, if the evaluator and tokens are set properly
        
        Raises Exception, if the evaluator does not work properly.
        """
        assert self.cache_set, 'Cache not passed into the token familiy before test of evaluator'
        _, self.test_token = self.create()
#        self.test_token.Set_parameters(random = True)
        self.test_token.use_cache()
        if self.status['requires_grid']:
            self.test_token.use_grids_cache()
        print(self.test_token.grid_idx, self.test_token.params)
#        self.test_params = {}
#        for key in self.token_params.keys():
#            if self.token_params[key][0] == self.token_params[key][1]:
#                self.test_params[key] = self.token_params[key][0]
#            else:
#                if isinstance(self.token_params[key][0], float):
#                    self.test_params[key] = np.random.uniform(low = self.token_params[key][0], high = self.token_params[key][0]) 
#                else:
#                    self.test_params[key] = np.random.randint(self.token_params[key][0], self.token_params[key][1])
#        try:
        self.test_token.scaled = False
        self.test_evaluation = self._evaluator.apply(self.test_token)
        print('Test evaluation performed correctly')
#        except:
#            raise Exception('Something went wrong during the test evaluation')

    def chech_constancy(self, test_function, **tfkwargs):
        '''
        Method to check, if any single simple token in the studied domain is constant, or close to it. The constant token is to be displayed and deleted from tokens and cache.
        '''
        assert self.params_set
        constant_tokens_labels = []
        for label in self.tokens:
            print(type(global_var.tensor_cache.memory[label + ' power 1']))
            constancy = test_function(global_var.tensor_cache.memory[label + ' power 1'], **tfkwargs)
            if constancy:
                constant_tokens_labels.append(label)
        
        for label in constant_tokens_labels:
            print('Function ', label, 'is assumed to be constant in the studied domain. Removed from the equaton search')
            self.tokens.remove(label)
            global_var.tensor_cache.delete_entry(label + ' power 1')
      
    def evaluate(self, token):    # Return tensor of values of applied evaluator
        raise NotImplementedError('Method has been moved to the Factor class')
        if self.evaluator_set and self.cache_set:
            return self._evaluator.apply(token)
        else:
            raise TypeError('Evaluator function or its parameters not set brfore evaluator application.')
    
    def create(self, label = None, occupied : list = [], def_term_tokens = [], **factor_params):
        if type(label) == type(None): 
#            print('tokens in the term:', def_term_tokens)
            label = np.random.choice([token for token in self.tokens 
                                      if not token in occupied and 
                                      not def_term_tokens.count(token) >= self.token_params['power'][1] ])
        new_factor = Factor(token_name = label, 
                            status = self.status, family_type = self.type)
        
        if self.status['unique_token_type']:
            occupied_by_factor = self.tokens
        elif self.status['unique_specific_token']:
            occupied_by_factor = [label,]
        else:
            occupied_by_factor = []
        if len(factor_params) == 0: 
            new_factor.Set_parameters(params_description = self.token_params, 
                                      equality_ranges = self.equality_ranges, 
                                      random = True)
        else:
            new_factor.Set_parameters(params_description = self.token_params, 
                                      equality_ranges = self.equality_ranges, 
                                      random = False, 
                                      **factor_params)            
        new_factor.Set_evaluator(self._evaluator)
        return occupied_by_factor, new_factor
    
#    def change_variables(self, prev_operator):
#        assert 'token_matrices' in self.eval_params
#        for key, value in self.set_params['token_matrices'].items():
#            self.set_params['token_matrices'][key] = value - prev_operator # С индексом? 

    def cardinality(self, occupied : list = []):
        return len([token for token in self.tokens if not token in occupied])
            
#    def update_family(self, occupied):
    
class TF_Pool(object):
    '''
    
    '''
    def __init__(self, families):
        self.families = families
        
    @property
    def pool_tokens(self):
        raise NotImplementedError 
        
    @property
    def families_meaningful(self):
        return [family for family in self.families if family.status['meaningful']]

    @property
    def families_supplementary(self):
        return [family for family in self.families if not family.status['meaningful']]

       
    def families_cardinality(self, meaningful_only : bool = False, occupied : list = []):
        if meaningful_only:
            return np.array([family.cardinality(occupied) for family in self.families_meaningful])
        else:
            return np.array([family.cardinality(occupied) for family in self.families])
        
    def create(self, label = None, create_meaningful : bool = False, 
                      occupied : list = [], def_term_tokens = [], **kwargs) -> (str, Factor):
        if create_meaningful:
#            print('a', self.families, 'p', self.families_cardinality(True, occupied))
            if np.sum(self.families_cardinality(True, occupied)) == 0:
                print('occupied', occupied)
                raise ValueError('Tring to create a term from an empty pool')
            return np.random.choice(a = self.families_meaningful, 
                                    p = self.families_cardinality(True, occupied) / np.sum(self.families_cardinality(True, occupied))).create(label = label, 
                                                                 occupied = occupied,
                                                                 def_term_tokens = def_term_tokens,
                                                                 **kwargs)
        else:
            return np.random.choice(a = self.families, 
                                    p = self.families_cardinality(False)  / np.sum(self.families_cardinality(False))).create(label = label, 
                                                                 occupied = occupied,
                                                                 def_term_tokens = def_term_tokens,
                                                                 **kwargs)
    
    def __add__(self, other):
        return TF_Pool(families = self.families + other.families)
    
