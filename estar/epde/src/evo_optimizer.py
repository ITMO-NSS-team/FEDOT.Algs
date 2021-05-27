#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:22:20 2020

@author: mike_ubuntu
"""

from abc import ABC, abstractmethod, abstractproperty

import epde.src.eq_search_operators as baseline # 


class Operator_builder(ABC):    
    
    @abstractproperty
    def operator(self):
        pass

class Evolutionary_builder(Operator_builder):
    """
    Class of evolutionary operator builder. 
    
    Attributes:
    ------------
    
    operator : Evolutionary_operator object
        the evolutionary operator, which is being constructed for the evolutionary algorithm, which is applied to a population;
        
    Methods:
    ------------
    
    reset()
        Reset the evolutionary operator, deleting all of the declared suboperators.
        
    set_evolution(crossover_op, mutation_op)
        Set crossover and mutation operators with corresponding evolutionary operators, each of the Specific_Operator type object, to improve the 
        quality of the population.
    
    set_param_optimization(param_optimizer)
        Set parameter optimizer with pre-defined Specific_Operator type object to optimize the parameters of the factors, present in the equation.
        
    set_coeff_calculator(coef_calculator)
        Set coefficient calculator with Specific_Operator type object, which determines the weights of the terms in the equations.
        
    set_fitness(fitness_estim)
        Set fitness function value estimator with the Specific_Operator type object. 
    
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._operator = Evolutionary_operator()
        
    def set_evolution(self, crossover_op, mutation_op):
        self._operator.crossover = crossover_op
        self._operator.mutation = mutation_op
    
    def set_param_optimization(self, param_optimizer):
        self._operator.param_optimization = param_optimizer
        
    def set_coeff_calculator(self, coef_calculator):
        self._operator.coeff_calculator = coef_calculator
        
    def set_fitness(self, fitness_estim):
        self._operator.fitness = fitness_estim
    
    @property
    def operator(self):
        self._operator.check_correctness
        operator = self._operator
        #self.reset()
        return operator


class Operator_director:
    '''
    
    Class for the director, responsible for the creation of default evolutionary operator
    
    Attributes
    ----------
    
    constructor : Evolutionary_builder object
        builder of the evolutionary operator. It is imporatant to note the next attribute:
    constructor.operator : Evolutionary_operator object
        property of the constructor, which returns the spectific operator, which can be applied to the population;
    
    Methods:
    ---------
    operator_assembly(sparcity, **kwargs)
        Construct baseline evolutionary operator, which is able to discover the equation. The operator contains crossover (parents selected by tournament 
        selection, the offsprings are generated with the exchange of terms and factor parameters), mutation (the change of term to a new random one or the 
        random alteration of parameters of factors in a term). The real-valued parts of the equation coefficient are detected by the LASSO regression, and 
        the fitness function are calculated as the inverse value of discrepancy between the left and the right parts of the equation.

    Examples:
    ---------
       
    The simplest definition of evolutionary operator with all default arguments:
        
    >>> # ...
    >>> 
    >>> director = Operator_director()
    >>> director.operator_assembly()
    >>>  
    >>> # Definition of equation trainer & token families
    >>>
    >>> Trainer.train(epochs = 150, evolutionary_operator = director.constructor.operator)    
    '''
    
    def __init__(self, baseline = True):
        self._constructor = None
        if baseline:
            self._constructor = Evolutionary_builder()
    
    @property
    def constructor(self):
        return self._constructor
    
    @constructor.setter
    def constructor(self, constructor):
        self._constructor = constructor

    def operator_assembly(self, **kwargs):
        '''
        
        Construct baseline evolutionary operator, which is able to discover the equation. The operator contains crossover (parents selected by tournament 
        selection, the offsprings are generated with the exchange of terms and factor parameters), mutation (the change of term to a new random one or the 
        random alteration of parameters of factors in a term). The real-valued parts of the equation coefficient are detected by the LASSO regression, and 
        the fitness function are calculated as the inverse value of discrepancy between the left and the right parts of the equation.

        Attributes:
        ------------
            
        selection_params : dict
            Dictionary of parameters for the seletcion operator. In this dictionary, the keys represent parameter name, and key - parameter value. 
            The parameters for the operator must be: 
                
                'part_with_offsprings' - part of the population, allowed for procreation. Base value: 0.2; 
                
                'tournament_groups' - size of the groups, into which the population is divided to participate in the tournament. Base value: 2;
            
        param_mutation : dict
            Dictionary of parameters for the mutation of factor parameters. In this dictionary, the keys represent parameter name, and key - parameter value. 
            The parameters for the operator must be: 
                
            'r_param_mutation' - probability of mutation of individual parameter. Base value: 0.2;
            
            'strict_restrictions' - bool, the marker, if the parameters can take values off the allowed range during the evolution. 
            True can help to converge, to the optimal values of factor parameters, if they are close to the ends of allowed ranges. Base value: True;
            
            'multiplier' - affects the scale of the normal districution, from which the increment is selected
            (N(loc = 0., scale = ...['multiplier']*(right_end_of_interval - left_end_of_interval))). Base value: 0.1;
            
        term_mutation : dict
            Dictionary of parameters for the mutation of total term reconstruction. In this dictionary, the keys represent parameter name, 
            and key - parameter value. The parameters for the operator must be: 
                
            'forbidden_tokens' - list of forbidden token, which can not take part in the creation of new terms. Best to leave empty: it will be filled 
            automatically. Base value: empty list [];
            
        'mutation_params' : dict
            Dictionary of parameters for the general overlay of parameters. In this dictionary, the keys represent parameter name, 
            and key - parameter value. The parameters for the operator must be:             
                
            'indiv_mutation_prob' - probability of an individual to be affected by any type of the mutation. Base value: 0.5;
            
            'type_probabilities' - probabilites of each type of the mutation. Base values: list of 2 elements (for param. mutation and total term recombination)
                [1 - 1/pow(term_params_num, 2), 1/pow(term_params_num, 2)], where term_params_num is the number of parameters in the term.
            
            'elitism' - number of the most fit individuals, excluded from the equation. Base value: 1;
            
            'r_mutation' - probability of a term in equation to be mutated. Base value: 0.3;
            
        'param_crossover_params' : dict
            Dictionary of parameters for the crossover of the parameters. The operator is applied to terms, which are the present in both parent 
            equations with the same factor functions, but with different parameter values. In this dictionary, the keys represent parameter name, 
            and key - parameter value. The parameters for the operator must be:             
            
            'proportion' - proportion, in which the new values of the parameters are taken in interval between the parents' values: 
            param_new = param_parent_1 + ...['proportion'] * (param_parent_2 - param_parent_1). Base value: 0.4;
            
        'term_crossover_params' : dict
            Dictionary of parameters for the crossover of the terms. The operator is applied as the exchange of the completely different terms 
            during the creation of an offspring. In this dictionary, the keys represent parameter name, and key - parameter value. The parameters 
            for the operator must be:             
            
            'crossover_probability' - probability of a term to be exchanged between parents during the crossover. Base value: 0.3
            
        'fitness_eval_params' : dict
            Dictionary of parameters for the fitness function value evaluation. In this dictionary, the keys represent parameter name, 
            and key - parameter value. The parameters for the operator must be:              
                
            'penalty_coeff' - penalty coefficient, to that the value of fitness function is multiplied, if the equation is trivial, taking form t_1 = 0, 
            to allow the more complex structures to flourish. 
            
        '''
        selection = baseline.Baseline_t_selection(['part_with_offsprings', 'tournament_groups'])
        
        param_mutation = baseline.Parameter_mutation(['r_param_mutation', 'strict_restrictions', 'multiplier'])
        term_mutation = baseline.Term_mutation(['forbidden_tokens'])
        mutation = baseline.Baseline_mutation(['indiv_mutation_prob', 'type_probabilities', 'elitism', 'r_mutation'])
        
        param_crossover = baseline.Param_crossover(['proportion'])
        term_crossover = baseline.Term_crossover(['crossover_probability'])
        crossover = baseline.Baseline_crossover([])        
        
        lasso = baseline.Baseline_LASSO(['sparcity'])
        fitness_eval = baseline.Baseline_fitness(['penalty_coeff'])
        
        selection.params = {'part_with_offsprings' : 0.2, 'tournament_groups' : 2} if not 'selection_params' in kwargs.keys() else  kwargs['selection_params']
        param_mutation.params = {'r_param_mutation' : 0.2, 'strict_restrictions' : True, 'multiplier' : 0.1} if not 'param_mutation_params' in kwargs.keys() else kwargs['param_mutation_params']
        term_mutation.params = {'forbidden_tokens': []} if not 'term_mutation_params' in kwargs.keys() else kwargs['term_mutation_params']
        mutation.params = {'indiv_mutation_prob' : 0.5, 'type_probabilities' : [], 'elitism' : 1, 'r_mutation' : 0.3} if not 'mutation_params' in kwargs.keys() else kwargs['mutation_params']
        param_crossover.params = {'proportion' : 0.4} if not 'param_crossover_params' in kwargs.keys() else kwargs['param_crossover_params']
        term_crossover.params = {'crossover_probability' : 0.3} if not 'term_crossover_params' in kwargs.keys() else kwargs['term_crossover_params']
        crossover.params = {} if not 'crossover_params' in kwargs.keys() else kwargs['crossover_params']
        fitness_eval.params = {'penalty_coeff' : 0.5} if not 'fitness_eval_params' in kwargs.keys() else kwargs['fitness_eval_params']
        
        mutation.suboperators = {'Mutation' : [param_mutation, term_mutation], 'Coeff_calc' : lasso, 'Fitness_eval' : fitness_eval}
        crossover.suboperators = {'Selection' : selection, 'Param_crossover' : param_crossover, 'Term_crossover' : term_crossover,
                                  'Coeff_calc' : lasso, 'Fitness_eval' : fitness_eval} 
        
        self._constructor.set_evolution(crossover, mutation)
        self._constructor.set_coeff_calculator(lasso)
        self._constructor.set_fitness(fitness_eval)


class Evolutionary_operator():
    '''
    
    Class of the evolutionary algorithm operator, encasulating the main processing of the population on evolutionary step: 
    evolution (mutation/crossover operators), operator of optimization of elementary functions parameters (optional),
    coeffficients and fitness function values calculators, selection operator
    
    Attributes:
    -----------
    crossover : Specific_Operator object
        an operator of crossover, which takes population as input, performs the crossover process, and returns an evolved population;
    
    mutation : Specific_Operator object
        an operator of mutation, which takes population as input, performs the mutation process, and returns an evolved population;
        
    It is mandatory to have crossover or mutation operators present, but highly advised to have both;
    
    param_optimiation : Specific_Operator object
        an operator, optimizing the parameters of the factors, composing terms of the equations in population
        
    fitness : Specific_Operator object
        operator, calculation fitness function value for the population individuals

    coeff_calculator : Specific_Operator object
        operator, calculation coefficients for the equations, represented by population individuals

    
    '''
    def __init__(self): #, indiv_type
        self.crossover = None; self.mutation = None
        self.param_optimization = None
        self.coeff_calculator = None
        self.fitness = None
        self.reserve_sparcity = None
    
    def check_correctness(self):
        '''
        
        Check if the operator has any evolutionary search suboperator (mutation or crossover), the equation coefficient calculator and fitness function.

        '''        
        
        if self.mutation == None and self.crossover == None:
            raise ValueError('The algorithm requires any evolutionary operator. Declare crossover or mutation.')
        if self.coeff_calculator == None and self.fitness == None:
            raise ValueError('No method to calculate the weights of the equation is defined.')
        
        
    def apply(self, population, separate_vars, show_operations = False):
        '''
        
        Apply operator to the popuation.
        '''
                
        #print(type(population))
        self.check_correctness()
        if type(self.crossover) != type(None):
            population = self.crossover.apply(population, separate_vars)
            if show_operations:
                print('performed crossover')
        if type(self.mutation) != type(None):
            if show_operations:
                print('performed mutation')
            population = self.mutation.apply(population)
        if type(self.param_optimization) != type(None):
            if show_operations:
                print('performed parameter optimization')
            population = self.param_optimization.apply(population)
        return population

        
    def get_fitness(self, individual): # Пр
        self.coeff_calculator.apply(individual)
        return self.fitness.apply(individual)

    def set_sparcity(self, sparcity_value = None):
        if type(sparcity_value) != type(None):
#            self.reserve_sparcity = self.coeff_calculator.params['sparcity']
            self.coeff_calculator.params = {'sparcity' : sparcity_value}
            self.reserve_sparcity = sparcity_value
        elif type(self.reserve_sparcity) != type(None): # Доработать
            self.coeff_calculator.params = {'sparcity' : self.reserve_sparcity}
        else:
            pass