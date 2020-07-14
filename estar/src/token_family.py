#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:39:18 2020

@author: mike_ubuntu
"""

import numpy as np

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
    def __init__(self, eval_function):
        self._evaluator = eval_function
        
    def set_params(self, **params):
        """
        Set the parameters of the evaluator, using keyword arguments
        """
        self.params = params
        
    def apply(self, token_label, token_params):
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
        try:
            return self._evaluator(token_label, token_params, self.params)
        except TypeError:
            raise TypeError('Wrong parameters passed into the evaluator.')

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
        self.evaluator_set = False; self.params_set = False
        
    def set_status(self, mandatory = False, unique_specific_token = False, unique_token_type = False, unique_for_right_part = False):
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
        self.status['mandatory'] = mandatory
        self.status['unique_specific_token'] = unique_specific_token
        self.status['unique_token_type'] = unique_token_type
        self.status['unique_for_right_part'] = unique_for_right_part

    def set_params(self, tokens, token_params):
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
        if self.evaluator_set:
            self.test_evaluator()

    def set_evaluator(self, eval_function, **eval_params):    #Test, if the evaluator works properly
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
        self._evaluator = Evaluator(eval_function)
        self._evaluator.set_params(**eval_params)
        self.evaluator_set = True
        if self.params_set:
            self.test_evaluator()

    def test_evaluator(self):
        """
        Method to test, if the evaluator and tokens are set properly
        
        Raises Exception, if the evaluator does not work properly.
        """
        self.test_token = np.random.choice(self.tokens)
        self.test_params = {}
        for key in self.token_params.keys():
            if self.token_params[key][0] == self.token_params[key][1]:
                self.test_params[key] = self.token_params[key][0]
            else:
                if isinstance(self.token_params[key][0], float):
                    self.test_params[key] = np.random.uniform(low = self.token_params[key][0], high = self.token_params[key][0]) 
                else:
                    self.test_params[key] = np.random.randint(self.token_params[key][0], self.token_params[key][1])
        try:
            self.test_evaluation = self._evaluator.apply(self.test_token, self.test_params)
            print('Test evaluation performed correctly')
        except:
            raise Exception('Something went wrong during the test evaluation')
      
    def evaluate(self, token, token_params):    # Return tensor of values of applied evaluator
        if self.evaluator_set and self.params_set:
            return self._evaluator.apply(token, token_params)
        else:
            raise TypeError('Evaluator function or its parameters not set brfore evaluator application.')