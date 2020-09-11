#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:26:03 2020

@author: mike_ubuntu
"""

import numpy as np
import copy 
from sklearn.linear_model import LinearRegression
from functools import reduce
from abc import ABC, abstractmethod

from src.term import Check_Unqueness
from src.equation import Equation
from src.supplementary import *
from src.supplementary import Detect_Similar_Terms


class Population:
    def __init__(self, evol_operator, history, tokens, pop_size, basic_terms,
                 eq_len = 8, max_factors_in_terms = 3, visualizer = None): 
        
        self.evol_operator = evol_operator
        self.tokens = tokens
        self.visualizer = visualizer

        self.pop_size = pop_size
        self.population = [Equation(self.tokens, basic_terms, eq_len, max_factors_in_terms) for i in range(pop_size)]
        for eq in self.population:
            eq.Split_data()
            eq.check_split_correctness()
            evol_operator.get_fitness(eq)
        
        self.history = history


    def Genetic_Iteration(self, iter_index, strict_restrictions = True):
        self.population = Population_Sort(self.population)
        self.population = self.population[:self.pop_size]
        print(iter_index, self.population[0].fitness_value)
        if type(self.visualizer) != type(None): 
            self.visualizer.update(self.population[0].fitness_value)
            print('Drawing')
        
        if iter_index > 0: 
            del self.prev_population
        self.prev_population = copy.deepcopy(self.population) # deepcopy?

        self.population = self.evol_operator.apply(self.population)


    def Initiate_Evolution(self, iter_number, log_file = None, test_indicators = True):
        if test_indicators:
            print('Evolution performed with intermediate indicators')
        #self.fitness_values = np.empty(iter_number)
        for idx in range(iter_number):
            strict_restrictions = False if idx < iter_number - 1 else True
            
            self.Genetic_Iteration(idx, strict_restrictions = strict_restrictions)
            self.population = Population_Sort(self.population)
            self.history.extend_fitness_history(self.population[0].fitness_value)
            self.history.save_fitness()
            
            if log_file: log_file.Write_apex(self.population[0], idx)
            if test_indicators: 
                print('iteration %3d' % idx)
                print('best fitness:', self.population[0].fitness_value, ', worst_fitness', self.population[-1].fitness_value)
                print('weights:', self.population[0].weights)
#        return self.fitness_values


    def Calculate_True_Weights(self, sort = True):
        if sort:
            self.population = Population_Sort(self.population)
        self.final_weights = Get_true_coeffs(self.tokens, self.population[0])
        
    
    def text_form(self, with_zero_weights = True):
        if type(self.final_weights) == type(None):
            raise Exception('Trying to get the precise equation before applying final linear regression')
        
        form = ''
        for term_idx in range(len(self.population[0].terms)):
            if term_idx != self.population[0].target_idx:
                weight = self.final_weights[term_idx] if term_idx < self.population[0].target_idx else self.final_weights[term_idx-1]
                if weight != 0. or with_zero_weights:    
                    form += str(weight) + ' * ' + self.population[0].terms[term_idx].text_form + ' + '
        form += str(self.final_weights[-1]) + ' = ' + self.population[0].terms[self.population[0].target_idx].text_form
        return form


def Get_true_coeffs(tokens, equation): # Не забыть про то, что последний коэф - для константы
    target = equation.terms[equation.target_idx]

    equation.check_split_correctness()
            
    
    target_vals = target.Evaluate(False)
    features_vals = []
    nonzero_features_indexes = []
    for i in range(len(equation.terms)):
        if i == equation.target_idx:
            continue
        idx = i if i < equation.target_idx else i-1
        if equation.weights[idx] != 0:
            features_vals.append(equation.terms[i].Evaluate(False))
            nonzero_features_indexes.append(idx)
            
    print('Indexes of nonzero elements:', nonzero_features_indexes)
    if len(features_vals) == 0:
        return np.zeros(len(equation.terms)) #Bind_Params([(token.label, token.params) for token in target.gene]), [('0', 1)]
    
    features = features_vals[0]
    if len(features_vals) > 1:
        for i in range(1, len(features_vals)):
            features = np.vstack([features, features_vals[i]])
    features = np.vstack([features, np.ones(features_vals[0].shape)]) # Добавляем константную фичу
    features = np.transpose(features)  
    
    estimator = LinearRegression(fit_intercept=False)
    try:
        estimator.fit(features, target_vals)
    except ValueError:
        features = features.reshape(-1, 1)
        estimator.fit(features, target_vals)
        
    valueable_weights = estimator.coef_
    weights = np.zeros(len(equation.terms))
#    print()
    for weight_idx in range(len(weights)-1):
        if weight_idx in nonzero_features_indexes:
            weights[weight_idx] = valueable_weights[nonzero_features_indexes.index(weight_idx)]
    weights[-1] = valueable_weights[-1]
    
    return weights #target, list(zip(features_list_labels, weights))    


def Get_true_coeffs_old(tokens, equation):
    target = equation.terms[equation.target_idx]

    target_vals = target.Evaluate(False)
    features_list = []
    features_list_labels = []
    for i in range(len(equation.terms)):
        if i == equation.target_idx:
            continue
        idx = i if i < equation.target_idx else i-1
        if equation.weights[idx] != 0:
            features_list_labels.append(equation.terms[i])
            features_list.append(equation.terms[i].Evaluate(False))


    if len(features_list) == 0:
        return Bind_Params([(token.label, token.params) for token in target.gene]), [('0', 1)]
    
    features = features_list[0]
    if len(features_list) > 1:
        for i in range(1, len(features_list)):
            features = np.vstack([features, features_list[i]])
    features = np.vstack([features, np.ones(features_list[0].shape)]) # Добавляем константную фичу
    features_list_labels.append({'1':{'power':1}})
    features = np.transpose(features)  
    
    estimator = LinearRegression(fit_intercept=False)
    try:
        estimator.fit(features, target_vals)
    except ValueError:
        features = features.reshape(-1, 1)
        estimator.fit(features, target_vals)
    weights = estimator.coef_
    return target, list(zip(features_list_labels, weights))    

#def Get_true_coeffs(evaluator, eval_params, tokens, token_params, equation, n_params = 2):
#    target = equation.terms[equation.target_idx]
#
#    target_vals = target.Evaluate(False)
#    features_list = []
#    features_list_labels = []
#    for i in range(len(equation.terms)):
#        if i == equation.target_idx:
#            continue
#        idx = i if i < equation.target_idx else i-1
#        if equation.weights[idx] != 0:
#            features_list_labels.append(Bind_Params([(token.label, token.params) for token in equation.terms[i].gene]))
#            features_list.append(equation.terms[i].Evaluate(False))
#
#
#    if len(features_list) == 0:
#        return Bind_Params([(token.label, token.params) for token in target.gene]), [('0', 1)]
#    
#    features = features_list[0]
#    if len(features_list) > 1:
#        for i in range(1, len(features_list)):
#            features = np.vstack([features, features_list[i]])
#    features = np.vstack([features, np.ones(features_list[0].shape)]) # Добавляем константную фичу
#    features_list_labels.append({'1':{'power':1}})
#    features = np.transpose(features)  
#    
#    estimator = LinearRegression(fit_intercept=False)
#    try:
#        estimator.fit(features, target_vals)
#    except ValueError:
#        features = features.reshape(-1, 1)
#        estimator.fit(features, target_vals)
#    weights = estimator.coef_
#    return Bind_Params([(token.label, token.params) for token in target.gene]), list(zip(features_list_labels, weights))    