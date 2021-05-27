#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:59:24 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy
#import 
from epde.src.structure import SoEq

def set_argument(var, fun_kwargs, base_value):
    try:
        res = fun_kwargs[var]
    except KeyError:
        res = base_value
    return res

class systems_population_constructor(object):
    def __init__(self, pool, terms_number, max_factors_in_term, eq_search_evo, sparcity_interval = (0, 1)):
        self.pool = pool; self.terms_number = terms_number
        self.eq_search_evo = eq_search_evo
        self.max_factors_in_term = max_factors_in_term; #self.eq_search_iters = eq_search_iters
        self.equation_number = len(self.pool.families_meaningful)
        self.sparcity_interval = sparcity_interval
#        print(self.equation_number)
#        raise NotImplementedError
        #len([1 for token_family in self.tokens if token_family.status['meaningful']])
    
    def create(self, **kwargs): # Дописать
        pop_size = set_argument('population_size', kwargs, 16)
        sparcity = set_argument('sparcity', kwargs, np.random.uniform(low = self.sparcity_interval[0], 
                                                                      high = self.sparcity_interval[1],
                                                                      size = self.equation_number))
        eq_search_iters = set_argument('eq_search_iters', kwargs, 50)
        
        print('Creating new equation')
        created_solution = SoEq(pool = self.pool, terms_number = self.terms_number,
                                max_factors_in_term = self.max_factors_in_term, 
                                sparcity = sparcity)
        created_solution.set_eq_search_evolutionary(self.eq_search_evo)
        print('searching equations with ', eq_search_iters, 'iterations, and popsize of ', pop_size)
        created_solution.create_equations(pop_size, sparcity, eq_search_iters)
        print('Equation created')
        return created_solution        
    

class sys_search_evolutionary_operator(object): # Возможно, организовать наследование от эвол. оператора из src.eq_search_operators
    def __init__(self, xover, mutation):
        '''
        Define the evolutionary operator to be used in the search of system of differential equations. 
        
        Parameters:
            xover : function
                The crossover/recombination operator for the evolutionary algorithm, specified in form of function, that
                must take two parent individuals as arguments and returns two offsprings.
                
            mutation : function
                The mutation operator for the evolutionary, which must take an individual as parameter and will return the changed 
                copy of the input individual.
                
                
        '''
        self._xover = xover
        self._mutation = mutation
        
    def mutation(self, solution):
#        output = deepcopy(solution)
#        output.vals = self._mutation(output.vals) #output.vals + np.random.normal(scale = )
        output = self._mutation(solution)
        output.create_equations()
        return output

    def crossover(self, parents_pool):
        offspring_pool = []
        for idx in np.arange(np.int(np.floor(len(parents_pool)/2.))):
#            print(parents_pool[2*idx].vals, parents_pool[2*idx+1].vals)
            offsprings_generated = self._xover((parents_pool[2*idx], parents_pool[2*idx+1]))
            offspring_pool.extend(offsprings_generated)
        for offspring in offspring_pool:
            offspring.create_equations()
        return offspring_pool
    
def gaussian_mutation(solution):
    assert type(solution) == SoEq, 'Object of other type, than the system of equation (SoEq), has been passed to the mutation operator'
    solution_new = deepcopy(solution)
    solution_new.vals += np.random.normal(size = solution_new.vals.size)
    return solution_new
#mixing_xover = lambda parents: parents[1].obj_

def mixing_xover(parents):
    assert all([type(parent) == SoEq for parent in parents]), 'Object of other type, than the system of equation (SoEq), has been passed to the crossover operator'
    proportion = np.random.uniform(low = 1e-6, high = 0.5-1e-6)
    offsprings = [deepcopy(parent) for parent in parents]
    offsprings[0].precomputed_value = False; offsprings[1].precomputed_value = False
    offsprings[0].precomputed_domain = False; offsprings[1].precomputed_domain = False
    
    offsprings[0].vals = parents[0].vals + proportion * (parents[1].vals - parents[0].vals)
    offsprings[1].vals = parents[0].vals + (1 - proportion) * (parents[1].vals - parents[0].vals)
    
    return offsprings

