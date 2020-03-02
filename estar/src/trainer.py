#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:33:54 2020

@author: mike_ubuntu
"""

import numpy as np
import datetime
from src.population import Population


def Discover_Equtaion(tokens, token_params, basic_terms, **kwargs): 
    '''
    
    
    
    '''
    t1 = datetime.datetime.now()

    assert 'evaluator' in kwargs.keys() and 'eval_params' in kwargs.keys()
    
    alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else 0.2
    a_proc = kwargs['a_proc'] if 'a_proc' in kwargs.keys() else 0.2
    r_crossover = kwargs['r_crossover'] if 'r_crossover' in kwargs.keys() else 0.3
    r_param_mutation = kwargs['r_param_mutation'] if 'r_param_mutation' in kwargs.keys() else 0.7
    r_mutation = kwargs['r_mutation'] if 'r_mutation' in kwargs.keys() else 0.3 
    mut_chance = kwargs['mut_chance'] if 'mut_chance' in kwargs.keys() else 0.6
    iter_number = kwargs['iter_number'] if 'iter_number' in kwargs.keys() else 100
    pop_size = kwargs['pop_size'] if 'pop_size' in kwargs.keys() else 8
    eq_len = kwargs['eq_len'] if 'eq_len' in kwargs.keys() else 6
    max_factors = kwargs['max_factors'] if 'max_factors' in kwargs.keys() else 2
    
    print('in "process cell" function: ', type(tokens), tokens)
    population = Population(kwargs['evaluator'], kwargs['eval_params'], tokens, token_params,
                                   pop_size = pop_size, basic_terms = basic_terms, a_proc = a_proc,
                                   r_crossover = r_crossover, r_param_mutation = r_param_mutation,
                                   r_mutation=r_mutation, mut_chance = mut_chance,
                                   alpha = alpha, eq_len = eq_len, max_factors_in_terms = max_factors)

    best_fitnesses = population.Initiate_Evolution(iter_number = iter_number, estimator_type='Lasso', log_file = None, test_indicators = True)
    
    print('Achieved best fitness:', best_fitnesses[-1])
    
    population.Calculate_True_Weights(kwargs['evaluator'], kwargs['eval_params'])
                    
    t2 = datetime.datetime.now()
    res = ((t1, t2), (population.target_term, population.zipped_list), best_fitnesses) 
    print('result:', res[:-1])            
#    return best_fitnesses


class Equation_Trainer:
    def __init__(self, tokens, token_params, evaluator, evaluator_params, basic_terms):
        '''
        
        The initializer for evolutionary algorithm of equation discovery.
        
        Parameters:
        -----------
        
        tokens : list, or tuple
            Symbolic tokens, that will be used as building blocks in the process of equation discovery;
            
        token_params : collections.OrderedDict
            The ordered dictionary, containing information about the token parameters. The order of dictionary represents the order, 
            in which these parameters would be encoded in gene for evolutionary algorithm; the key is the name of the parameter; the value is the 
            tuple of two elements, where the 1st value is the lowest boundary of the parameter, and 2nd is the highest. Note, that the type of the values
            defines the type of the parameter (for example, parameter with allowed interval of (0, 5), will take only integer values from 0 to 5, while (0., 5.), 
            can take all real values in this interval. Complex values have not been implemented).
        
        evaluator : function
            Function, that allows the evaluation of token values. For further information, look into the User Guide.
            
        eval_params : dict
            Dictionary, containing parameters of the evaluator. Additional details are defined by the requirements of ``evaluator`` function.
            
        basic_terms : list of dictionaries 
            Defines the mandatory terms for each equation (it is not guaranteed, that the coefficients for these terms will not be 0). Has the structure: 
            list contains dictionaries with keys - tokens, that are present in term, and values - dictionaries, where the key is the token parameter and the value is its value  
            
        Example:
        -------
        
        >>> import numpy as np
        >>> import collections
        >>> from src.supplementary import Define_Derivatives
        >>> from src.trainer import Equation_Trainer
        >>> u = np.load('U.npy')
        >>> derivatives = np.load('Derivatives.npy')
        >>> variables = np.ones((2 + derivatives.shape[1], ) + u_initial.shape)
        >>> variables[1, :] = u
        >>> for i in range(0, derivatives.shape[1]):
        ...:    variables[i+2] = derivatives[:, i].reshape(variables[i+2].shape)             
        >>> token_names = Define_Derivatives(u_initial.ndim, max_order = 2)
        >>> token_parameters = collections.OrderedDict([('power', (0, 3))])
        >>> basic_terms = [{'1':{'power':1}}, {'1':{'power':1}, 'u':{'power':1}}]
        >>> Trainer = Equation_Trainer(tokens = token_names, 
        ...:                           token_params = token_parameters,
        ...:                           evaluator = derivative_evaluator, 
        ...:                           evaluator_params = {'token_matrices':variables, 
        ...:                                               'parameter_indexes':{'power':0}}, 
        ...:                                               basic_terms = basic_terms)
        >>> Trainer.Parameters_grid(('alpha', 'a_proc', 'r_crossover', 'r_param_mutation', 
        ...:                         'r_mutation', 'mut_chance', 'pop_size', 
        ...:                         'eq_len', 'max_factors'), 
        ...:                         ((0.01, 0.1, 5), 0.2, 0.6, 0.8, 0.5, 0.8, 10, 6, 2))
        >>> Trainer.Train(epochs = 100)
        
        '''
        
        self.tokens = tokens; self.token_params = token_params
        self.evaluator = evaluator
        self.evaluator_params = evaluator_params
        self.basic_terms = basic_terms
        self.tuning_grid = None
        print('Set parameters for the evolutionary algorithm:')
        print('')
    
    
    def Parameters_grid(self, parameters_order, params):
        '''
        
        Define the evolutionary algorithm hyperparameters. Possibility to define them on the grid - 
        to better analyze the response of the result to different hyperparameters
        
        Parameters:
        -----------
        
        parameters_order : tuple of strings
            Sequence of parameters names (symbolic forms) with order, in which they would be defined in `params` ; You can define following 
            hyperparameters: 
                'alpha' - sparsity constant for Lasso regression;
                
                'a_proc' - part of population, permitted to procreate;
                
                'r_crossover' - probability of individual term to exchange between parent equations during crossover;
            
                'r_mutation' - probability of individual term in equation to mutate during mutation process;
                
                'r_param_mutation' - probability of each token attribute of term to mutate, if such type of mutation is available for token;
                
                'mut_chance' - probability of equation to population except 1st "elite" one to mutate;
                
                'pop_size' - size of the population of equations, for which the evolution will be held;
                
                'eq_len' - maximum number of terms in equation. Note, that the real number will probably be lower due to filtration of 
                insignificant terms with Lasso;
                
                'max_factors' - maximum number otuplef tokens, taken as factors for the terms.
                
        params : tuple of integer/real values
            Sequence of parameters values, which wolud be used to create grid. If you want to set a with a single value, 
            then initialize them with a single int/float value; if you want to test multiple values of the parameter, set them in tuple with
            order (start, stop, num) for the ``numpy.linspace()`` function, which will be used to create values series.
            
        Example:
        ---------

        >>> Trainer.Parameters_grid(('alpha', 'a_proc', 'r_crossover', 'r_param_mutation', 
        ...:                         'r_mutation', 'mut_chance', 'pop_size', 
        ...:                         'eq_len', 'max_factors'), 
        ...:                         ((0.01, 0.1, 5), 0.2, 0.6, 0.8, 0.5, 0.8, 10, 6, 2))
       
        Notes:
        ---------
        The main influence on the result of the equation discovery process is held by alpha - sparcity constant for Lasso regression, and eq_len - length of the equation. 
        
        Alpha defines the number of terms, present in equation, and it can be close to impossible to predict its optimal value.
        
        The length of the equation can be set to resonably high value (8 or more): insignificant extra terms would be filtered via Lasso regression. 
        
        Other parameters mainly influence the speed of algorithm convergence.
    
        '''
        
        self.parameters_order = parameters_order
        parameter_arrays = []
        for parameter in params:
            parameter_arrays.append(parameter if (isinstance(parameter, int) or isinstance(parameter, float)) else np.linspace(parameter[0], parameter[1], parameter[2]))
        self.tuning_grid = np.meshgrid(*parameter_arrays, indexing = 'ij')
      
    def Delete_grid(self):
        '''
        
        Delete the parameters grid for the trainer
        
        '''
        
        self.tuning_grid = None
    
    def Train(self, epochs, parameters_order = None, parameters = None):
        '''
        
        Method to train the equation to obtain its symbolic form;
        
        Parameters:
        -----------
        
        epochs : int
            Number of epochs for evolutionary algorithm
        
        parameters : tuple of integer/real values
        
        parameters_order : tuple of strings
            Additional possibility to define a single set of hyperparameters for EA. Described in details in 
            ``Equation_Trainer.Parameters_grid()``  
        '''
        
        if self.tuning_grid:
            print('Using parameters from grid')
            use_params = np.vectorize(Discover_Equtaion, otypes = None, excluded = ['tokens', 'token_params', 'basic_terms', 'evaluator', 'eval_params', 'iter_number'])
            use_params(tokens = self.tokens, token_params = self.token_params, basic_terms = self.basic_terms,
                       evaluator = self.evaluator, eval_params = self.evaluator_params, iter_number = epochs, 
                       alpha = self.tuning_grid[self.parameters_order.index('alpha')], 
                       a_proc = self.tuning_grid[self.parameters_order.index('a_proc')], 
                       r_crossover = self.tuning_grid[self.parameters_order.index('r_crossover')], 
                       r_param_mutation = self.tuning_grid[self.parameters_order.index('r_param_mutation')],
                       r_mutation = self.tuning_grid[self.parameters_order.index('r_mutation')],
                       mut_chance = self.tuning_grid[self.parameters_order.index('mut_chance')],
                       pop_size = self.tuning_grid[self.parameters_order.index('pop_size')],
                       eq_len = self.tuning_grid[self.parameters_order.index('eq_len')],
                       max_factors = self.tuning_grid[self.parameters_order.index('max_factors')])
        elif parameters: # .any()
            print('Using single vector of parameters')
            Discover_Equtaion(tokens = self.tokens, token_params = self.token_params, basic_terms = self.basic_terms,
                         evaluator = self.evaluator, eval_params = self.evaluator_params, iter_number = epochs, 
                         alpha = parameters[parameters_order.index('alpha')], 
                         a_proc = parameters[parameters_order.index('a_proc')], 
                         r_crossover = parameters[parameters_order.index('r_crossover')], 
                         r_mutation = parameters[parameters_order.index('r_mutation')],
                         mut_chance = parameters[parameters_order.index('mut_chance')],
                         pop_size = parameters[parameters_order.index('pop_size')],
                         eq_len = parameters[parameters_order.index('eq_len')],
                         max_factors = parameters[parameters_order.index('max_factors')])
        else:
            raise ValueError('The EA hyperparameters are not defined')