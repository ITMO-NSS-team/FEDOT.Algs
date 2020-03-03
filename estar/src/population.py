#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:26:03 2020

@author: mike_ubuntu
"""

import numpy as np
import copy 
from sklearn.linear_model import LinearRegression

from src.term import Check_Unqueness
from src.equation import Equation
from src.supplementary import *

class Population:
    def __init__(self, evaluator, eval_params, tokens, token_params, pop_size, basic_terms, a_proc,
                 r_crossover, r_param_mutation, r_mutation, mut_chance, alpha, eq_len = 8, max_factors_in_terms = 3): 
        
        self.tokens = tokens; self.token_params = token_params
        
        self.part_with_offsprings = a_proc
        self.crossover_probability = r_crossover; self.r_param_mutation = r_param_mutation
        self.r_mutation = r_mutation; self.mut_chance = mut_chance
        self.n_params = len(list(token_params.keys()))

        self.pop_size = pop_size
        self.population = [Equation(self.tokens, self.token_params, evaluator, eval_params, basic_terms, alpha, eq_len, max_factors_in_terms) for i in range(pop_size)]
        for eq in self.population:
            eq.Split_data()
            eq.Calculate_Fitness()

    def Genetic_Iteration(self, estimator_type, elitism = 1, strict_restrictions = True):
        self.population = Population_Sort(self.population)
        self.population = self.population[:self.pop_size]
                
        children = Tournament_crossover(self.population, self.part_with_offsprings, self.tokens, 
                                        crossover_probability = self.crossover_probability)

        map(lambda x: x.Split_data, children)
        map(lambda x: x.Calculate_Fitness, children)
        self.population = self.population + children

        for i in range(elitism, len(self.population)):
            if np.random.random() <= self.mut_chance:
                self.population[i].Mutate(r_mutation = self.r_mutation, 
                               r_param_mutation = self.r_param_mutation, strict_restrictions = strict_restrictions)


    def Initiate_Evolution(self, iter_number, estimator_type, log_file = None, test_indicators = False):
        self.fitness_values = np.empty(iter_number)
        for idx in range(iter_number):
            strict_restrictions = False if idx < iter_number - 1 else True
            self.Genetic_Iteration(estimator_type = estimator_type, strict_restrictions = strict_restrictions)
            self.population = Population_Sort(self.population)
            self.fitness_values[idx]= self.population[0].fitness_value
            if log_file: log_file.Write_apex(self.population[0], idx)
            if test_indicators: 
                print('iteration %3d' % idx)
                print('best fitness:', self.population[0].fitness_value, ', worst_fitness', self.population[-1].fitness_value)
                print('gene of target term:', self.population[0].terms[self.population[0].target_idx].gene)
                print('weights:', self.population[0].weights)
        return self.fitness_values

    def Calculate_True_Weights(self, evaluator, eval_params):
        self.population = Population_Sort(self.population)
        self.population = self.population[:self.pop_size]
#        print('Final gene:', self.population[0].terms[self.population[0].target_idx].gene)
#        print(self.population[0].fitness_value, Decode_Gene(self.population[0].terms[self.population[0].target_idx].gene,
#              self.tokens, list(self.token_params.keys()), self.n_params))
#        print('weights:', self.population[0].weights)       
        self.target_term, self.zipped_list = Get_true_coeffs(evaluator, eval_params, self.tokens, list(self.token_params.keys()),
                                                              self.population[0], self.n_params)  
        
def Crossover(equation_1, equation_2, tokens, crossover_probability = 0.1):
    if len(equation_1.terms) != len(equation_2.terms):
        raise IndexError('Equations have diffferent number of terms')
    result_equation_1 = copy.deepcopy(equation_1) #Equation(variables, variables_names, terms_number = len(equation_1.terms))
    result_equation_2 = copy.deepcopy(equation_2) #Equation(variables, variables_names, terms_number = len(equation_2.terms))      

    for i in range(2, len(result_equation_1.terms)):
        if np.random.uniform(0, 1) <= crossover_probability and Check_Unqueness(result_equation_1.terms[i], result_equation_2.terms) and Check_Unqueness(result_equation_2.terms[i], result_equation_1.terms):
            internal_term = result_equation_1.terms[i]
            result_equation_1.terms[i] = result_equation_2.terms[i]
            result_equation_2.terms[i] = internal_term

    return result_equation_1, result_equation_2


def Parent_selection_for_crossover(population, tournament_groups = 2):
    selection_indexes = np.random.choice(len(population), tournament_groups, replace = False)
    candidates = [population[idx] for idx in selection_indexes]
    parent_idx = [idx for _, idx in sorted(zip(candidates, selection_indexes), key=lambda pair: pair[0].fitness_value)][-1] #np.argmax([population[y].fitness_value for y in selection_indexes])
    parent = population[parent_idx]
    return parent, parent_idx


def Tournament_crossover(population, part_with_offsprings, tokens, 
                         tournament_groups = 2, crossover_probability = 0.1):
    children = []
    for i in range(int(len(population)*part_with_offsprings)):
        parent_1, parent_1_idx = Parent_selection_for_crossover(population, tournament_groups)
        parent_2, parent_2_idx = Parent_selection_for_crossover(population, tournament_groups)
        child_1, child_2 =  Crossover(parent_1, parent_2, tokens,
                                                       crossover_probability = crossover_probability)
        child_1.Split_data(); child_2.Split_data()
        children.append(child_1); children.append(child_2)
    return children        


def Get_true_coeffs(evaluator, eval_params, tokens, token_params, equation, n_params = 2):
    target = equation.terms[equation.target_idx]

    target_vals = target.Evaluate(evaluator, False, eval_params)    
    features_list = []
    features_list_labels = []
    for i in range(len(equation.terms)):
        if i == equation.target_idx:
            continue
        idx = i if i < equation.target_idx else i-1
        if equation.weights[idx] != 0:
            features_list_labels.append(Decode_Gene(equation.terms[i].gene, tokens, token_params, n_params))
            features_list.append(equation.terms[i].Evaluate(evaluator, False, eval_params))

    if len(features_list) == 0:
        return Decode_Gene(target.gene, tokens, token_params, n_params), [('0', 1)]
    
    features = features_list[0]
    if len(features_list) > 1:
        for i in range(1, len(features_list)):
            features = np.vstack([features, features_list[i]])
    features = np.transpose(features)  
    
    estimator = LinearRegression()
    try:
        estimator.fit(features, target_vals)
    except ValueError:
        features = features.reshape(-1, 1)
        estimator.fit(features, target_vals)
    weights = estimator.coef_
    return Decode_Gene(target.gene, tokens, token_params, n_params), list(zip(features_list_labels, weights))    