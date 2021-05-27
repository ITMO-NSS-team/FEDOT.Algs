#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:48:54 2020

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy
from functools import reduce
from sklearn.linear_model import Lasso, LinearRegression
#from multiprocessing import Process


import epde.src.globals
from epde.src.structure import Term, Check_Unqueness
from epde.src.supplementary import *
from epde.src.supplementary import Filter_powers, Detect_Similar_Terms


def flatten(folded_equation):
    assert type(folded_equation) == list
    return reduce(lambda x,y: x+y, folded_equation)


def try_iterable(arg):
    try:
        _ = [elem for elem in arg]
    except TypeError:
        return False
    return True


class Specific_Operator():
    '''
    Universal class for operator of an arbitrary purpose
    '''
    def __init__(self, param_keys):
        self.param_keys = param_keys
    
    @property
    def params(self):
        return self._params #list(self._params.keys()), self._param.values()
    
    @params.setter
    def params(self, param_dict):
#        print(list(param_dict.keys()), self.param_keys)
        assert self.param_keys == list(param_dict.keys())
        self._params = param_dict

    @property
    def suboperators(self):
        return self._suboperators
        
    @suboperators.setter
    def suboperators(self, operators):
        self._suboperators = operators

    def apply(self, target):
        pass


class Baseline_t_selection(Specific_Operator):
    """
    Basic tournament selection, inherits properties from class Specific_Operator();

    Methods:
    ---------
    
    apply(population)
        return the indexes of the individuals, selected for procreation.
    """
    def apply(self, population):
        """
        Select a pool of pairs, between which the procreation will be later held. The population is divided into groups, and the individual 
        with highest fitness is allowed to take part in the crossover. 
        
        Parameters:
        -----------
        population : list of equation objects
            The population, among which the selection is held.
        
        Returns:
        -----------
        parent_pairs : list of lists (pairs of parents)
            The pairs of parents, chosen to take part in the crossover.
            
        """
        parent_pairs = []
        for pair_idx in range(int(len(population)*self.params['part_with_offsprings']/2.)):
            pair = []
            for parent_idx in np.arange(2):
                selection_indexes = np.random.choice(len(population), self.params['tournament_groups'], replace = False)
                candidates = [population[idx] for idx in selection_indexes]
                pair.append([idx for _, idx in sorted(zip(candidates, selection_indexes), key=lambda pair: pair[0].fitness_value)][-1])
            parent_pairs.append(pair)
        return parent_pairs


class Baseline_crossover(Specific_Operator):
    """
    The crossover operator, combining parameter crossover for terms with same factors but different parameters & 
    full exchange of terms between the completely different ones.
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the Specific_Operator class. 
        Suboperators, performing tasks of parent selection, parameter crossover, full terms crossover, calculation of weights for each terms & 
        fitness function calculation. Dictionary: keys - strings from 'Selection', 'Param_crossover', 'Term_crossover', 'Coeff_calc', 'Fitness_eval'.
        values - corresponding operators (objects of Specific_Operator class).

    Methods:
    -----------
    apply(population)
        return the new population, created with the noted operators and containing both parent individuals and their offsprings.    
    
    """
    def apply(self, population, separate_vars):
        """
        Method to obtain a new population by selection of parent individuals (equations) and performing a crossover between them to get the offsprings.
        
        Attributes:
        -----------
        population : list of Equation objects
            the population, to that the operator is applied;
            
        Returns:
        -----------
        population : list of Equation objects
            the new population, containing both parents and offsprings;
        
        """
        indexes = self.suboperators['Selection'].apply(population)

        offsprings = []    
        for idx_pair in indexes:
            if len(population[idx_pair[0]].structure) != len(population[idx_pair[1]].structure):
                raise IndexError('Equations have diffferent number of terms')
            result_equation_1 = deepcopy(population[idx_pair[0]])
            result_equation_2 = deepcopy(population[idx_pair[1]])    
        
            result_equation_1_terms, result_equation_2_terms = Detect_Similar_Terms(result_equation_1, result_equation_2)
            assert len(result_equation_1_terms[0]) == len(result_equation_2_terms[0]) and len(result_equation_1_terms[1]) == len(result_equation_2_terms[1])
            same_num = len(result_equation_1_terms[0]); similar_num = len(result_equation_1_terms[1])
            result_equation_1.structure = flatten(result_equation_1_terms); result_equation_2.structure = flatten(result_equation_2_terms)
        
            for i in range(same_num, same_num + similar_num):
                temp_term_1, temp_term_2 = self.suboperators['Param_crossover'].apply(result_equation_1.structure[i], result_equation_2.structure[i]) 
                if (Check_Unqueness(temp_term_1, result_equation_1.structure[:i] + result_equation_1.structure[i+1:]) and 
                    Check_Unqueness(temp_term_2, result_equation_2.structure[:i] + result_equation_2.structure[i+1:])):                     
                    result_equation_1.structure[i] = temp_term_1; result_equation_2.structure[i] = temp_term_2

            for i in range(same_num + similar_num, len(result_equation_1.structure)):
                if Check_Unqueness(result_equation_1.structure[i], result_equation_2.structure) and Check_Unqueness(result_equation_2.structure[i], result_equation_1.structure):
                    internal_term = result_equation_1.structure[i]
                    result_equation_1.structure[i] = result_equation_2.structure[i]
                    result_equation_2.structure[i] = internal_term
                    temp_term_1, temp_term_2 = self.suboperators['Term_crossover'].apply(result_equation_1.structure[i], result_equation_2.structure[i])
            result_equation_1.select_target_idx(separate_vars = separate_vars)
            result_equation_2.select_target_idx(separate_vars = separate_vars)
            result_equation_1.check_split_correctness()
            result_equation_2.check_split_correctness()
            offsprings.extend((result_equation_1, result_equation_2))
        
#        processes = []
        for offspring in offsprings:
            self.suboperators['Coeff_calc'].apply(offspring)
            
        for offspring in offsprings:
            self.suboperators['Fitness_eval'].apply(offspring)
#        for offspring in offsprings:
#            p = Process(target = lambda x: self.suboperators['Coeff_calc'].apply(x), args=(offspring,))
#            processes.append(p)
#            p.start()
#        
#        for p in processes:
#            p.join()
#            
#        processes = []
#        for offspring in offsprings:
#            p = Process(target = lambda x: self.suboperators['Fitness_eval'].apply(x), args=(offspring,))
#            processes.append(p)
#            p.start()
#        
#        for p in processes:
#            p.join()            
            
        #map(lambda x: self.suboperators['Coeff_calc'].apply(x), offsprings)
        #map(lambda x: self.suboperators['Fitness_eval'].apply(x), offsprings)
        population.extend(offsprings)             
        return population        


class Param_crossover(Specific_Operator):
    """
    The crossover exchange between parent terms with the same factor functions, that differ only in the factor parameters. 

    Noteable attributes:
    -----------
    params : dict
        Inhereted from the Specific_Operator class. 
        Main key - 'proportion', value - proportion, in which the offsprings' parameter values are chosen.
        
    Methods:
    -----------
    apply(population)
        return the offspring terms, constructed as the parents' factors with parameter values, selected between the parents' ones.        
    """
    def apply(self, term_1, term_2):
        """
        Get the offspring terms, constructed as the parents' factors with parameter values, selected between the parents' ones.
        
        Attributes:
        ------------
        term_1, term_2 : Term objects
            The parent terms.
            
        Returns:
        ------------
        offspring_1, offspring_2 : Term objects
            The offspring terms.
        
        """
        offspring_1 = deepcopy(term_1)  # Потенциальная ошибка
        offspring_2 = deepcopy(term_2)
        offspring_1.reset_saved_state(); offspring_2.reset_saved_state()
        
        if len(offspring_1.structure) != len(offspring_2.structure):
            print([(token.label, token.params) for token in offspring_1.structure], [(token.label, token.params) for token in offspring_2.structure])
            raise Exception('Wrong terms passed:')
        
        for term1_token_idx in np.arange(len(term_1.structure)):
            term2_token_idx = [i for i in np.arange(len(term_2.structure)) if term_2.structure[i].label == term_1.structure[term1_token_idx].label][0]
            for param_idx, param_descr in offspring_1.structure[term1_token_idx].params_description.items():
                if param_descr['name'] == 'power': power_param_idx = param_idx
                if param_descr['name'] == 'dim': dim_param_idx = param_idx                
#            power_param_idx = ; dim_param_idx = 
#            for param in offspring_1.structure[term1_token_idx].params.keys():
#                if param != 'power' and param != 'dim':
#                    try:
#                        offspring_1.structure[term1_token_idx].params[param] = (term_1.structure[term1_token_idx].params[param] + 
#                                                                   self.params['proportion']*(term_2.structure[term2_token_idx].params[param] 
#                                                                   - term_1.structure[term1_token_idx].params[param]))
#                    except KeyError:
#                        print([(token.label, token.params) for token in offspring_1.structure], [(token.label, token.params) for token in offspring_2.structure])
#                        raise Exception('Wrong set of parameters:', offspring_1.structure[term1_token_idx].params.keys(), offspring_2.structure[term1_token_idx].params.keys())
#                    offspring_2.structure[term2_token_idx].params[param] = (term_1.structure[term1_token_idx].params[param] + 
#                                                               (1 - self.params['proportion'])*(term_2.structure[term2_token_idx].params[param] 
#                                                               - term_1.structure[term1_token_idx].params[param]))
        
            for param_idx in np.arange(offspring_1.structure[term1_token_idx].params.size):
                if param_idx != power_param_idx and param_idx != dim_param_idx:
                    try:
                        offspring_1.structure[term1_token_idx].params[param_idx] = (term_1.structure[term1_token_idx].params[param_idx] + 
                                                                   self.params['proportion']*(term_2.structure[term2_token_idx].params[param_idx] 
                                                                   - term_1.structure[term1_token_idx].params[param_idx]))
                    except KeyError:
                        print([(token.label, token.params) for token in offspring_1.structure], [(token.label, token.params) for token in offspring_2.structure])
                        raise Exception('Wrong set of parameters:', offspring_1.structure[term1_token_idx].params_description, offspring_2.structure[term1_token_idx].params_description)
                    offspring_2.structure[term2_token_idx].params[param_idx] = (term_1.structure[term1_token_idx].params[param_idx] + 
                                                               (1 - self.params['proportion'])*(term_2.structure[term2_token_idx].params[param_idx] 
                                                               - term_1.structure[term1_token_idx].params[param_idx]))
        
        
         
        
        offspring_1.Reset_occupied_tokens(); offspring_2.Reset_occupied_tokens()
        return offspring_1, offspring_2


class Term_crossover(Specific_Operator):
    """
    The crossover exchange between parent terms, done by complete exchange of terms. 

    Noteable attributes:
    -----------
    params : dict
        Inhereted from the Specific_Operator class. 
        Main key - 'crossover_probability', value - probabilty of the term exchange.
        
    Methods:
    -----------
    apply(population)
        return the offspring terms, which are the same parents' ones, but in different order, if the crossover occured.
        .        
    """    
    def apply(self, term_1, term_2):
        """
        Get the offspring terms, which are the same parents' ones, but in different order, if the crossover occured.
        
        Attributes:
        ------------
        term_1, term_2 : Term objects
            The parent terms.
            
        Returns:
        ------------
        offspring_1, offspring_2 : Term objects
            The offspring terms.
        
        """        
        if np.random.uniform(0, 1) <= self.params['crossover_probability']:
            return term_2, term_1
        else:
            return term_1, term_2
        
        
class Baseline_mutation(Specific_Operator):
    """
    The general operator of mutation, which applies all off the mutation suboperators, which are selected in its self.suboperators['Mutation'] 
    to the population    
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the Specific_Operator class. 
        Suboperators, performing tasks of calculation of weights for each terms & fitness function calculation. 
        Additionally, it contains suboperators of different mutation types. Dictionary: keys - strings from 'Mutation', 'Coeff_calc', 'Fitness_eval'.
        values - corresponding operators (objects of Specific_Operator class).

    params : dict
        Inhereted from the Specific_Operator class. 
        Parameters of the operator; main parameters: 
            
            elitism - number of inidividuals with highest fitness values, spared from the mutation to preserve their quality;
                
            indiv_mutation_prob - probability of an individual in a population to be affected by a mutation;
            
            r_mutation - probability of a term in an equation, selected for mutation, to be affected by any mutation operator;
            
            type_probabilities - propabilities for selecting each mutation suboperator to affect the equation (In this operator, set by euristic, to be updated).
            
    Methods:
    -----------
    apply(population)
        return the new population, created with the specified operators and containing mutated population.    
    
    """

#    def indiv_mut(self, population):
#        

    def apply(self, population):
        """
        Return the new population, created with the specified operators and containing mutated population.
        
        Parameters:
        -----------
        population : list of Equation objects
            The population, to which the mutation operators would be applied.
            
        Returns:
        ----------
        population : list of Equation objects
            The input population, altered by mutation operators.
            
        """
        population = Population_Sort(population)
        for indiv_idx in range(self.params['elitism'], len(population)):
            if np.random.uniform(0, 1) <= self.params['indiv_mutation_prob']:
                equation = population[indiv_idx]

                for term_idx in range(equation.n_immutable, len(equation.structure)):
                    if np.random.uniform(0, 1) <= self.params['r_mutation'] and term_idx != equation.target_idx:
                        #if type(self.params['type_probabilities']) == type(None):
                        self.params['type_probabilities'] = [1 - 1/pow(equation.structure[term_idx].total_params, 2), 1/pow(equation.structure[term_idx].total_params, 2)]
                        if try_iterable(self.suboperators['Mutation']):
                            mut_operator = np.random.choice(self.suboperators['Mutation'], p=self.params['type_probabilities'])
                        else:
                            mut_operator = self.suboperators['Mutation']
                        if 'forbidden_tokens' in mut_operator.params.keys():
                            mut_operator.params['forbidden_tokens'] = [factor for factor in equation.structure[equation.target_idx].structure if factor.status['unique_for_right_part']]   # [factor.label for factor in equation.structure[].structure]
                        equation.structure[term_idx] = mut_operator.apply(term_idx, equation)
                        equation.check_split_correctness()
                self.suboperators['Coeff_calc'].apply(equation)
                self.suboperators['Fitness_eval'].apply(equation)
#                equation.Split_data()
                population[indiv_idx] = equation
        return population        

    
class Baseline_mutation_special(Specific_Operator):
    """
    The general operator of mutation, which applies all off the mutation suboperators, which are selected in its self.suboperators['Mutation'] 
    to the population    
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the Specific_Operator class. 
        Suboperators, performing tasks of calculation of weights for each terms & fitness function calculation. 
        Additionally, it contains suboperators of different mutation types. Dictionary: keys - strings from 'Mutation', 'Coeff_calc', 'Fitness_eval'.
        values - corresponding operators (objects of Specific_Operator class).

    params : dict
        Inhereted from the Specific_Operator class. 
        Parameters of the operator; main parameters: 
            
            elitism - number of inidividuals with highest fitness values, spared from the mutation to preserve their quality;
                
            indiv_mutation_prob - probability of an individual in a population to be affected by a mutation;
            
            r_mutation - probability of a term in an equation, selected for mutation, to be affected by any mutation operator;
            
            type_probabilities - propabilities for selecting each mutation suboperator to affect the equation (In this operator, set by euristic, to be updated).
            
    Methods:
    -----------
    apply(population)
        return the new population, created with the specified operators and containing mutated population.    
    
    """

#    def indiv_mut(self, population):
#        

    def apply(self, population):
        """
        Return the new population, created with the specified operators and containing mutated population.
        
        Parameters:
        -----------
        population : list of Equation objects
            The population, to which the mutation operators would be applied.
            
        Returns:
        ----------
        population : list of Equation objects
            The input population, altered by mutation operators.
            
        """
        population = Population_Sort(population)
        
        
        for indiv_idx in range(self.params['elitism'], len(population)):
            if np.random.uniform(0, 1) <= self.params['indiv_mutation_prob']:
                equation = deepcopy(population[indiv_idx])

                for term_idx in range(equation.n_immutable, len(equation.structure)):
                    if np.random.uniform(0, 1) <= self.params['r_mutation'] and term_idx != equation.target_idx:
                        #if type(self.params['type_probabilities']) == type(None):
                        self.params['type_probabilities'] = [1 - 1/pow(equation.structure[term_idx].total_params, 2), 1/pow(equation.structure[term_idx].total_params, 2)]
                        if try_iterable(self.suboperators['Mutation']):
                            mut_operator = np.random.choice(self.suboperators['Mutation'], p=self.params['type_probabilities'])
                        else:
                            mut_operator = self.suboperators['Mutation']
                        if 'forbidden_tokens' in mut_operator.params.keys():
                            mut_operator.params['forbidden_tokens'] = [factor for factor in equation.structure[equation.target_idx].structure if factor.status['unique_for_right_part']]   # [factor.label for factor in equation.structure[].structure]
                        equation.structure[term_idx] = mut_operator.apply(term_idx, equation)
                        equation.check_split_correctness()
                self.suboperators['Coeff_calc'].apply(equation)
                self.suboperators['Fitness_eval'].apply(equation)
#                equation.Split_data()
                population[indiv_idx] = equation
                
        if np.random.uniform(0, 1) <= self.params['indiv_mutation_prob']:
            equation = deepcopy(population[0])

            for term_idx in range(equation.n_immutable, len(equation.structure)):
                if np.random.uniform(0, 1) <= self.params['r_mutation'] and term_idx != equation.target_idx:
                    weight_idx = term_idx if term_idx < equation.target_idx else term_idx - 1
                    if equation.weights[weight_idx] == 0:
                    #if type(self.params['type_probabilities']) == type(None):
                        self.params['type_probabilities'] = [1 - 1/pow(equation.structure[term_idx].total_params, 2), 1/pow(equation.structure[term_idx].total_params, 2)]
                        if try_iterable(self.suboperators['Mutation']):
                            mut_operator = np.random.choice(self.suboperators['Mutation'], p=self.params['type_probabilities'])
                        else:
                            mut_operator = self.suboperators['Mutation']
                        if 'forbidden_tokens' in mut_operator.params.keys():
                            mut_operator.params['forbidden_tokens'] = [factor for factor in equation.structure[equation.target_idx].structure if factor.status['unique_for_right_part']]   # [factor.label for factor in equation.structure[].structure]
                        equation.structure[term_idx] = mut_operator.apply(term_idx, equation)
                        equation.check_split_correctness()
            self.suboperators['Coeff_calc'].apply(equation)
            self.suboperators['Fitness_eval'].apply(equation)
#                equation.Split_data()
            population.append(equation)
            
        return population            


class Term_mutation(Specific_Operator): # Добавить "запрещённые" токены 
    """
    Specific operator of the term mutation, where the term is replaced with a randomly created new one.
    """
    def apply(self, term_idx, equation):
        """
        Return a new term, randomly created to be unique from other terms of this particular equation.
        
        Parameters:
        -----------
        term_idx : integer
            The index of the mutating term in the equation.
            
        equation : Equation object
            The equation object, in which the term is present.
        
        Returns:
        ----------
        new_term : Term object
            The new, randomly created, term.
            
        """       
#        print('reference forbidden tokens:', self.params['forbidden_tokens'])
        new_term = Term(equation.pool, max_factors_in_term = equation.max_factors_in_term, forbidden_tokens = self.params['forbidden_tokens'])        #) #
        while not Check_Unqueness(new_term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
            new_term = Term(equation.pool, max_factors_in_term = equation.max_factors_in_term, forbidden_tokens = self.params['forbidden_tokens'])
#        print(equation.structure[term_idx].text_form, 'mutated into', new_term.text_form)
        new_term.use_cache()
        return new_term
        

#class Best_individual_mutation(Specific_Operator):
#    def apply(self, term_idx, equation):
#        new_term = Term(equation.tokens, max_factors_in_term = equation.max_factors_in_term, forbidden_tokens = self.params['forbidden_tokens'])        #) #
#        while not Check_Unqueness(new_term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
#            new_term = Term(equation.tokens, max_factors_in_term = equation.max_factors_in_term, forbidden_tokens = self.params['forbidden_tokens'])
##        print(equation.structure[term_idx].text_form, 'mutated into', new_term.text_form)
#        return new_term        


class Parameter_mutation_old(Specific_Operator):
    """
    Specific operator of the term mutation, where the term parameters are changed with a random increment.
    """
    def apply(self, term_idx, equation):
        """
        Specific operator of the term mutation, where the term parameters are changed with a random increment.
        
        Parameters:
        -----------
        term_idx : integer
            The index of the mutating term in the equation.
            
        equation : Equation object
            The equation object, in which the term is present.
        
        Returns:
        ----------
        new_term : Term object
            The new, created from the previous one with random parameters increment, term.
            
        """                
        unmutable_params = {'dim'}
        while True:
            term = equation.structure[term_idx] 
            for factor in term.structure:   # Возможное место ошибок
                parameter_selection = deepcopy(factor.params)
                token_family = [token_family for token_family in term.tokens if factor.label in token_family.tokens][0]
                for param, interval in token_family.token_params.items():
                    if param == 'power':
                        continue
                    if np.random.random() < self.params['r_param_mutation'] and param not in unmutable_params:
                        if interval[0] == interval[1]:
                            shift = 0
                            continue
                        if isinstance(interval[0], int):
                            shift = np.rint(np.random.normal(loc= 0, scale = self.params['multiplier']*(interval[1] - interval[0]))).astype(int) #
                        elif isinstance(interval[0], float):
                            shift = np.random.normal(loc= 0, scale = self.params['multiplier']*(interval[1] - interval[0]))
                        else:
                            raise ValueError('In current version of framework only integer and real values for parameters are supported') 
                        if self.params['strict_restrictions']:
                            parameter_selection[param] = np.min((np.max((parameter_selection[param] + shift, interval[0])), interval[1]))
                        else:
                            parameter_selection[param] = parameter_selection[param] + shift
                factor.Set_parameters(**parameter_selection)
            term.structure = Filter_powers(term.structure)        
            if Check_Unqueness(term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
                break
        term.reset_saved_state()
        return term


class Parameter_mutation(Specific_Operator):
    """
    Specific operator of the term mutation, where the term parameters are changed with a random increment.
    """
    def apply(self, term_idx, equation):
        """
        Specific operator of the term mutation, where the term parameters are changed with a random increment.
        
        Parameters:
        -----------
        term_idx : integer
            The index of the mutating term in the equation.
            
        equation : Equation object
            The equation object, in which the term is present.
        
        Returns:
        ----------
        new_term : Term object
            The new, created from the previous one with random parameters increment, term.
            
        """                
        unmutable_params = {'dim', 'power'}
        while True:
            term = equation.structure[term_idx] 
            for factor in term.structure:   # Возможное место ошибок
                parameter_selection = deepcopy(factor.params)
                for param_idx, param_properties in factor.params_description.items():
#                    if param_properties['name'] == 'power':
#                        continue
                    if np.random.random() < self.params['r_param_mutation'] and param_properties['name'] not in unmutable_params:
                        interval = param_properties['bounds']
                        if interval[0] == interval[1]:
                            shift = 0
                            continue
                        if isinstance(interval[0], int):
                            shift = np.rint(np.random.normal(loc= 0, scale = self.params['multiplier']*(interval[1] - interval[0]))).astype(int) #
                        elif isinstance(interval[0], float):
                            shift = np.random.normal(loc= 0, scale = self.params['multiplier']*(interval[1] - interval[0]))
                        else:
                            raise ValueError('In current version of framework only integer and real values for parameters are supported') 
                        if self.params['strict_restrictions']:
                            parameter_selection[param_idx] = np.min((np.max((parameter_selection[param_idx] + shift, interval[0])), interval[1]))
                        else:
                            parameter_selection[param_idx] = parameter_selection[param_idx] + shift
                factor.params = parameter_selection
            term.structure = Filter_powers(term.structure)        
            if Check_Unqueness(term, equation.structure[:term_idx] + equation.structure[term_idx+1:]):
                break
        term.reset_saved_state()
        return term

    
class Baseline_LASSO(Specific_Operator):
    """
    The operator, which applies LASSO regression to the equation object to obtain the real-valued coefficients from it.
    
    Notable attributes:
    -------------------
        
    params : dict
        Inhereted from the Specific_Operator class. 
        Parameters of the operator; main parameters: 
            
            sparcity - value of the sparcity constant in the LASSO operator;
            
    Methods:
    -----------
    apply(equation)
        calculate the coefficients of the equation, that will be stored in the equation.weights np.ndarray.    
        
    """
    def apply(self, equation):
        """
        Apply the LASSO operator to calculate the coefficients of the equation. The result is not returned, but stored in the equation.weights np.ndarray.
        
        Parameters:
        ------------
        equation : Equation object
            the equation object, to that the coefficients are obtained.
            
        Returns:
        ------------
        None
        """
#        equation.value() Настроить получение векторов значений фич и таргета
        estimator = Lasso(alpha = self.params['sparcity'], copy_X=True, fit_intercept=True, max_iter=1000,
                               normalize=False, positive=False, precompute=False, random_state=None,
                               selection='cyclic', tol=0.0001, warm_start=False)
#        print('running new lasso')
        _, target, features = equation.evaluate(normalize = True, return_val = False)

        estimator.fit(features, target)
        
#        if len(np.nonzero(estimator.coef_)[0]) == 1 and (1 - estimator.coef_[np.nonzero(estimator.coef_)[0][0]]) < 0.001:
#            np.save('file_features', features)
#            np.save('file_target', target)
##            print(equation.text_form)
#            print(estimator.coef_)
#            raise ValueError()        
        equation.weights_internal = estimator.coef_

        
#def Get_True_Coefficients(equation):
#    target_vals = equation.structure[equation.target_idx].value(False)
#    features_vals = []
#    nonzero_features_indexes = []
#    for i in range(len(equation.structure)):
#        if i == equation.target_idx:
#            continue
#        idx = i if i < equation.target_idx else i-1
#        if equation.weights[idx] != 0:
#            features_vals.append(equation.structure[i].value(False))
#            nonzero_features_indexes.append(idx)
#            
#    #print('Indexes of nonzero elements:', nonzero_features_indexes)
#    if len(features_vals) == 0:
#        return np.zeros(len(equation.structure)) #Bind_Params([(token.label, token.params) for token in target.structure]), [('0', 1)]
#    
#    features = features_vals[0]
#    if len(features_vals) > 1:
#        for i in range(1, len(features_vals)):
#            features = np.vstack([features, features_vals[i]])
#    features = np.vstack([features, np.ones(features_vals[0].shape)]) # Добавляем константную фичу
#    features = np.transpose(features)  
#    
#    estimator = LinearRegression(fit_intercept=False)
#    try:
#        estimator.fit(features, target_vals)
#    except ValueError:
#        features = features.reshape(-1, 1)
#        estimator.fit(features, target_vals)
#        
#    valueable_weights = estimator.coef_
#    weights = np.zeros(len(equation.structure))
##    print()
#    for weight_idx in range(len(weights)-1):
#        if weight_idx in nonzero_features_indexes:
#            weights[weight_idx] = valueable_weights[nonzero_features_indexes.index(weight_idx)]
#    weights[-1] = valueable_weights[-1]
#    
#    return weights            

class Baseline_fitness(Specific_Operator):
    """
    The operator, which calculates fitness function to the individual (equation).
    
    Notable attributes:
    -------------------
        
    params : dict
        Inhereted from the Specific_Operator class. 
        Parameters of the operator; main parameters: 
            
            penalty_coeff - penalty coefficient, to that the fitness function value of equation with no non-zero coefficients, is multiplied;
            
    Methods:
    -----------
    apply(equation)
        calculate the fitness function of the equation, that will be stored in the equation.fitness_value.    
        
    """
    def apply(self, equation):
        """
        Calculate the fitness function values. The result is not returned, but stored in the equation.fitness_value.
        
        Parameters:
        ------------
        equation : Equation object
            the equation object, to that the fitness function is obtained.
            
        Returns:
        ------------
        None
        """        
#        equation.value(normalize = False)
        
#        weights = Get_True_Coefficients(equation)   # Если заработает - настроить без лишних вычислений
#        print(type(equation), equation.weights_final)
        _, target, features = equation.evaluate(normalize = False, return_val = False)
        try:
            rl_error = np.linalg.norm(np.dot(features, equation.weights_final[:-1]) + 
                                  np.full(target.shape, equation.weights_final[-1]) - target, ord = 2) # + self.alpha * np.linalg.norm(self.weights, ord = 1)) 
#            print('error:', rl_error, np.linalg.norm(target, ord = 2), 'shape', [np.linalg.norm(features[..., idx]) for idx in range(features.shape[-1])], np.linalg.norm(np.dot(features, equation.weights_final[:-1])),  
#                                  np.linalg.norm(np.full(target.shape, equation.weights_final[-1])))
        except ValueError:
            raise ValueError('An error in getting weights ')
        if rl_error == 0:
            fitness_value = np.inf
            print('infinite fitness!', equation.text_form)
        else:
            fitness_value = 1 / (rl_error)
        if np.sum(equation.weights_final) == 0:
            fitness_value = fitness_value * self.params['penalty_coeff']
#        print(fitness_value)
        equation.fitness_value = fitness_value