#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:22:20 2020

@author: mike_ubuntu
"""

from abc import ABC, abstractmethod, abstractproperty

import src.ESTAR_baseline_classes as baseline


class Operator_builder(ABC):    
    
    @abstractproperty
    def operator(self):
        pass
    

class Evolutionary_builder(Operator_builder):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._operator = Evolutionary_operator()
        
    def set_evolution(self, crossover_op, mutation_op):
        self._operator.crossover = crossover_op
        self._operator.mutation = mutation_op
        
    def set_selection(self, selector):
        self._operator.selection = selector
    
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
        self.reset()
        return operator


class Operator_director:
    '''
    
    Class for the director, responsible for the creation of default evolutionary operator
    
    Attributes
    ----------
    
    _constructor : Evolutionary_builder object
        builder of the evolutionary operator. It is imporatant to note the next attribute:
    _constructor.operator : Evolutionary_operator object
        property of the constructor, which returns the spectific operator, which can be applied to the population;
    
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

    def operator_assembly(self, sparcity = 1, **kwargs):
        '''
        
        Default operator of evolutionary algorthm
        
        '''
        selection = baseline.Baseline_t_selection(['part_with_offsprings', 'tournament_groups'])
        
        param_mutation = baseline.Parameter_mutation(['r_param_mutation', 'strict_restrictions', 'multiplier'])
        term_mutation = baseline.Term_mutation(['forbidden_tokens'])
        mutation = baseline.Baseline_mutation(['indiv_mutation_prob', 'type_probabilities', 'elitism', 'r_mutation'])
        
        param_crossover = baseline.Param_crossover(['proportion'])
        term_crossover = baseline.Term_crossover(['crossover_probability'])
        crossover = baseline.Baseline_crossover([])        
        
        lasso = baseline.Baseline_LASSO(['sparcity']); lasso.params = {'sparcity' : sparcity}
        fitness_eval = baseline.Baseline_fitness(['penalty_coeff'])
        
        selection.params = {'part_with_offsprings' : 0.2, 'tournament_groups' : 2} if not 'selection_params' in kwargs.keys() else  kwargs['selection_params']
        param_mutation.params = {'r_param_mutation' : 0.2, 'strict_restrictions' : True, 'multiplier' : 0.1} if not 'param_mutation_params' in kwargs.keys() else kwargs['param_mutation_params']
        term_mutation.params = {'forbidden_tokens': []} if not 'term_mutation_params' in kwargs.keys() else kwargs['term_mutation_params']
        mutation.params = {'indiv_mutation_prob' : 0.5, 'type_probabilities' : [], 'elitism' : 1, 'r_mutation' : 0.3} if not 'mutation_params' in kwargs.keys() else kwargs['mutation_params']
        param_crossover.params = {'proportion' : 0.45} if not 'param_crossover_params' in kwargs.keys() else kwargs['param_crossover_params']
        term_crossover.params = {'crossover_probability' : 0.3} if not 'term_crossover_params' in kwargs.keys() else kwargs['term_crossover_params']
        crossover.params = {} if not 'crossover_params' in kwargs.keys() else kwargs['crossover_params']
        fitness_eval.params = {'penalty_coeff' : 0.5} if not 'fitness_eval_params' in kwargs.keys() else kwargs['crossover_params']
        
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
    
    
    def check_correctness(self):
        '''
        
        Check if the operator has any evolutionary search suboperator (mutation or crossover), the equation coefficient calculator and fitness function.

        '''        
        
        if self.mutation == None and self.crossover == None:
            raise ValueError('The algorithm requires any evolutionary operator. Declare crossover or mutation.')
        if self.coeff_calculator == None and self.fitness == None:
            raise ValueError('No method to calculate the weights of the equation is defined.')
        
        
    def apply(self, population):
        '''
        
        Apply operator to the .
        '''
                
        print(type(population))
        self.check_correctness()
        if type(self.crossover) != type(None):
            population = self.crossover.apply(population)
        if type(self.mutation) != type(None):
            population = self.mutation.apply(population)
        if type(self.param_optimization) != type(None):
            population = self.param_optimization.apply(population)
#        print(type(population))
        return population

        
    def get_fitness(self, individual):
        self.coeff_calculator.apply(individual)
        self.fitness.apply(individual)

#
#    def set_reverse_signal(self, *args):
#        
#        
    