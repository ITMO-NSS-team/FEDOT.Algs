#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:33:54 2020

@author: mike_ubuntu
"""

import numpy as np
#import datetime
from src.population import Population
from src.supplementary_media import Training_History
import time

def save_fitness(filename, fitnesses):
    np.save(filename, np.array(fitnesses))


def set_argument(var, fun_kwargs, base_value):
    try:
        res = fun_kwargs[var]
    except KeyError:
        res = base_value
    return res


def discover_Equation(tokens, evolutionary_operator, basic_terms, **kwargs): 
#    t1 = datetime.datetime.now()
    
    iter_number = set_argument('iter_number', kwargs, 100)
    pop_size = set_argument('pop_size', kwargs, 8)
    eq_len = set_argument('eq_len', kwargs, 6)
    max_factors = set_argument('max_factors', kwargs, 2)
    test_output = set_argument('test_output', kwargs, True)
    fitness_file = set_argument('fitness_file', kwargs, None)
    history = Training_History(filename = fitness_file)    
    
    visualizer = set_argument('visualizer', kwargs, None)
    if type(visualizer) != type(None):
        print('launching visualizer', type(visualizer))
        visualizer.launch()

    population = Population(evolutionary_operator, history, tokens, pop_size = pop_size, basic_terms = basic_terms, 
                            eq_len = eq_len, max_factors_in_terms = max_factors, visualizer = visualizer)

    population.Initiate_Evolution(iter_number = iter_number, log_file = None, test_indicators = test_output) #best_fitnesses = 
    
#    history.extend_fitness_history(best_fitnesses)
#    filename = 'Fitness_alpha_' + str(alpha) + '.npy'
#    save_fitness(filename, best_fitnesses)
    
#, 'with alpha =', alpha)
    population.Calculate_True_Weights()
    history.equation = population.name()
    history.iteration_history = population.iteration_times
    print('Achieved best fitness:', history.achieved_fitness)
    print('Discovered equation:', history.equation)
    return history
    
    
#    return eq_name


class Equation_Trainer:
    def __init__(self, tokens, basic_terms):
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
        
        self.tokens = tokens;
        self.basic_terms = basic_terms
        self.tuning_grid = None
        self._history = []


    
    def parameters_grid(self, parameters_order, params):
        '''
        
        Define the evolutionary algorithm population hyperparameters. Possibility to define them on the grid - 
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
      
    def delete_grid(self):
        '''
        
        Delete the parameters grid for the trainer
        
        '''
        
        self.tuning_grid = None
    
    def train(self, epochs, evolutionary_operator, parameters_order = None, parameters = None, fitness_file = None, visualizer = None):
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
        
        # Переделать векторизацию от метода к функции
        
        if self.tuning_grid:
            print('Using parameters from grid')
            use_params = np.vectorize(discover_Equation, otypes = None, 
                                      excluded = ['tokens', 'basic_terms', 'iter_number', 'fitness_file', 'visualizer'],
                                      cache=True)
            history = use_params(tokens = self.tokens, evolutionary_operator = evolutionary_operator, basic_terms = self.basic_terms, 
                                   iter_number = epochs, 
                                   pop_size = self.tuning_grid[self.parameters_order.index('pop_size')],
                                   eq_len = self.tuning_grid[self.parameters_order.index('eq_len')],
                                   max_factors = self.tuning_grid[self.parameters_order.index('max_factors')],
                                   test_output = self.tuning_grid[self.parameters_order.index('test_output')],
                                   fitness_file = fitness_file, visualizer = visualizer)
        elif parameters: # .any()
            print('Using single vector of parameters')
            history = self.discover_Equation(tokens = self.tokens, evolutionary_operator = evolutionary_operator, basic_terms = self.basic_terms, iter_number = epochs, 
                         pop_size = parameters[parameters_order.index('pop_size')],
                         eq_len = parameters[parameters_order.index('eq_len')],
                         max_factors = parameters[parameters_order.index('max_factors')],
                         test_output = self.tuning_grid[self.parameters_order.index('test_output')],
                         fitness_file = fitness_file, visualizer = visualizer)
        else:
            raise ValueError('The evolutionary algorithm hyperparameters are not defined')
        self._history.append(history)
          
    
    @property
    def history(self):
        return self._history        