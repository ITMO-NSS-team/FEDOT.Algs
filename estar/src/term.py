#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:39:55 2020

@author: mike_ubuntu
"""

import numpy as np
import collections
import copy

from src.supplementary import Encode_Gene, Decode_Gene


def Check_Unqueness(term, equation):
    if type(term) == Term:
        return not any([all(term.gene == equation_term.gene) for equation_term in equation])
    else:
        return not any([all(term == equation_term.gene) for equation_term in equation])

def normalize_ts_old(Input):    # Normalization of data time-frame
    Matrix = np.copy(Input)
    for i in np.arange(Matrix.shape[0]):
        norm  = np.abs(np.max(np.abs(Matrix[i, :])))
        if norm != 0:
            Matrix[i] = Matrix[i] / norm
        else:
            Matrix[i] = 1
    return Matrix


def normalize_ts(Input):    # Normalization of data time-frame
    Matrix = np.copy(Input)
    for i in np.arange(Matrix.shape[0]):
        #norm  = np.abs(np.max(np.abs(Matrix[i, :])))
        std = np.std(Matrix[i])
        if std != 0:
            Matrix[i] = (Matrix[i] - np.mean(Matrix[i])) / std
        else:
            Matrix[i] = 1
    return Matrix


class Term:
    def __init__(self, tokens, gene = None, token_params = collections.OrderedDict([('power', (0, 4))]), # = ['1', 'u']
                 init_random = False, label_dict = None, max_factors_in_term = 2, forbidden_tokens = None):

        """
        Class for the possible terms of the PDE, contating both packed symbolic form, and values on the grid;
        
        Attributes:
            gene : 1d - array of ints \r\n
            An array of 0 and 1, contating packed symbolic form of the equation. Shape: number_of_functions * max_power. Each subarray of max_power length 
            contains information about power of the corresponding function (power = sum of ones in the substring). Can be passed as the parameter into the 
            class during initiation;
            
            value : matrix of floats \r\n
            An array, containing value of the term in the grid in the studied space. It can be acquired from the self.Calculate_Value() method;
            
        Parameters:
            
            gene : 1d - array of integers \r\n
            Initiation of the gene for term. For further details look into the attributes;
            
            tokens_list : list of strings \r\n
            List of symbolic forms of the functions, that can be in the resulting equation;
            
            init_random : boolean, base value of False \r\n
            False, if the gene is created by passed label_dict, or gene. If True, than the gene is randomly generated, according to the parameters of max_power
            and max_factors_in_term;                    print('examining tokens:', forbidden_token, forbidden)       
            
            label_dict : dictionary \r\n
            Dictionary, containing information about the term: key - string of function symbolic form, value - power; 
            
            max_power : int, base value of 2 \r\n
            Maximum power of one function, that can exist in the term of the equation;
            
            max_factors_in_term : int, base value of 2 \r\n
            Maximum number of factors, that can exist in the equation; 
            
        """
        
        #assert init_random or ((gene or label_dict) and (not gene or not label_dict)), 'Gene initialization done incorrect'
        self.max_factors_in_term = max_factors_in_term            
        self.tokens = tokens;
        self.token_params = token_params; self.n_params = len(token_params.keys())
        self.label_dict = label_dict

        if init_random:
            self.Randomize_Gene(constant_token = self.tokens[0], forbidden = forbidden_tokens) 
        else:    
            if type(gene) == np.ndarray:
                self.gene = gene
                self.label_dict = Decode_Gene(self.gene, self.tokens, list(self.token_params.keys()), self.n_params)
            else:
                self.gene = Encode_Gene(self.label_dict, self.tokens, list(self.token_params.keys()), self.n_params)  #np.empty(shape = len(variable_list) * max_power)
    
#    def Determined_Gene()
    
    def Randomize_Gene(self, constant_token = '1', forbidden = None): # Разобраться с коидровкой
        self.label_dict = {}
        for token in self.tokens:
            term_params = {}
            for key, value in self.token_params.items():
                term_params[key] = 0
            self.label_dict[token] = term_params
            
        factor_num = np.random.randint(low = 1, high = self.max_factors_in_term + 1)
        
        self.label_dict[constant_token]['power'] = 1
        non_constant_tokens = list(self.label_dict.keys())
        non_constant_tokens.remove(constant_token)
        
        NoneType = type(None)
        if type(forbidden) != NoneType:
            forbidden = np.delete(forbidden, 0)
            for forbidden_token in forbidden:
                #print('examining tokens:', self.tokens[forbidden_token], non_constant_tokens)       
                try:
                    non_constant_tokens.remove(self.tokens[forbidden_token])             
                except ValueError:
                    continue        
            #print('after purge:', non_constant_tokens)

        for factor_idx in range(factor_num):
            while True:
                key = np.random.choice(non_constant_tokens)
#                print(self.label_dict)
                if self.label_dict[key]['power'] + 1 <= self.token_params['power'][1]:
                    self.label_dict[key]['power'] += 1 
                    break
                
        for token in list(self.label_dict.keys()):
            temp_keys = list(self.label_dict[token].keys())
            temp_keys.remove('power')
            for param in temp_keys:
                if isinstance(self.token_params[param][0], int):
                    self.label_dict[token][param] = np.random.randint(self.token_params[param][0], self.token_params[param][1]) 
                else:
                    self.label_dict[token][param] = self.token_params[param][0] + np.random.random()*(self.token_params[param][1] - self.token_params[param][0])

        self.gene = Encode_Gene(self.label_dict, self.tokens, list(self.token_params.keys()), self.n_params)

    def Remove_Dublicated_Factors(self, allowed_tokens, background_terms): # Переписать
        gene_cleared = np.copy(self.gene)
        factors_cleared = 0
        allowed = list(np.nonzero(allowed_tokens)[0])
        
        for idx in range(allowed_tokens.size):      # Filtration of forbidden tokens (that are present in the target for regression) from term 
            if self.gene[idx*self.n_params + list(self.token_params.keys()).index('power')] > 0 and not allowed_tokens[idx]:
                factors_cleared += int(self.gene[idx*self.n_params + list(self.token_params.keys()).index('power')])
                gene_cleared[idx*self.n_params + list(self.token_params.keys()).index('power')] = 0
        
        max_power_elements = [idx for idx in range(len(self.tokens)) if 
                              self.gene[idx*self.n_params + list(self.token_params.keys()).index('power')] == self.token_params['power'][1]]
        allowed = [factor for factor in allowed if not factor in max_power_elements]
        allowed.remove(0)
        clearing_iterations = 0 
        while True:
            gene_filled = np.copy(gene_cleared) 

            for i in range(factors_cleared):
                selected_idx = np.random.choice(allowed)
                gene_filled[selected_idx*self.n_params + list(self.token_params.keys()).index('power')] += 1
                if gene_filled[selected_idx*self.n_params + list(self.token_params.keys()).index('power')] == self.token_params['power'][1]:
                    allowed.remove(selected_idx)
            if Check_Unqueness(gene_filled, background_terms):
                self.gene = gene_filled
                break
            clearing_iterations += 1
            if clearing_iterations > 1000:
                print('background:')
                print([term.gene for term in background_terms])
                print([term.label_dict for term in background_terms])
                
                raise RuntimeError('Can not remove dublicated factors; can not build term from tokens', allowed, 
                                   ', previous try:', gene_filled, 'from ', gene_cleared, 'from', self.gene, 'with cleared ', factors_cleared, 'factors')            
        
    def Mutate_parameters(self, r_param_mutation, multiplier = 0.3, strict_restrictions = False): # Задать расстояние для мутации?
        gene_temp = copy.deepcopy(self.gene)
        for gene_idx in np.arange(gene_temp.size):
            if np.random.random() < r_param_mutation:
                #token_idx = int(gene_idx/self.n_params)
                param_idx = gene_idx % self.n_params
                param_name = list(self.token_params.keys()).index(param_idx)
                if param_name == 'power':
                    continue
                if isinstance(self.token_params[param_name][0], int):
                    shift = np.rint(np.random.normal(loc= 0, scale = multiplier*(self.token_params[param_name][1] - self.token_params[param_name][0])),
                                    dtype = np.int16)
                elif isinstance(self.token_params[param_name][0], float):
                    shift = np.random.normal(loc= 0, scale = multiplier*(self.token_params[param_name][1] - self.token_params[param_name][0]))
                else:
                    raise ValueError('In current version of framework only integer and real values for parameters are supported')
                if strict_restrictions:
                    gene_temp[gene_idx] = max(min(self.gene[gene_idx]+shift, self.token_params[param_name][1]), self.token_params[param_name][0])
                else:
                    gene_temp[gene_idx] += shift
        self.gene = gene_temp
        self.label_dict = Decode_Gene(self.gene, self.tokens, list(self.token_params.keys()), self.n_params)
    
    
    def Mutate_old_like(self, background_terms, allowed_factors, reverse_mutation_prob = 0.1):
        allowed = list(np.nonzero(allowed_factors)[0])        
        power_position = list(self.token_params.keys()).index('power')
        
        if np.sum([idx*self.n_params + power_position for idx in np.arange(len(self.tokens))], dtype = np.int8) == 0:
            iteration_idx = 0; max_attempts = 15
            while iteration_idx < max_attempts:
                mutated_gene = np.copy(self.gene)
                new_factor_idx = np.random.choice(allowed)
                mutated_gene[new_factor_idx*self.n_params + power_position] = 1
                if Check_Unqueness(mutated_gene, background_terms):
                    self.gene = mutated_gene
                    return
                iteration_idx += 1
                
            while True:
                mutated_gene = np.copy(self.gene)
                new_factor_idx = np.random.choice(allowed, size = 2)
                mutated_gene[new_factor_idx[0]*self.n_params + power_position] = 1
                mutated_gene[new_factor_idx[1]*self.n_params + power_position] = 1
                if Check_Unqueness(mutated_gene, background_terms):
                    self.gene = mutated_gene
                    return

        max_power_elements = [idx for idx in range(len(self.tokens)) if 
                              self.gene[idx*self.n_params + power_position] == self.token_params['power'][1]]
        lowest_power_elements = [idx for idx in range(len(self.tokens)) if 
                                 self.gene[idx*self.n_params + power_position] == self.token_params['power'][1]]
        
        iteration_idx = 0; max_attempts = 15
        total_power = np.sum([self.gene[idx*self.n_params + power_position] for idx in range(len(self.tokens))], dtype = np.int8)
        
        while True:
            mutated_gene = np.copy(self.gene)
            if np.random.uniform(0, 1) <= reverse_mutation_prob or iteration_idx > 15:
                mutation_type = np.random.choice(['Reduction', 'Increasing'])
                if mutation_type == 'Reduction' or total_power >= self.max_factors_in_term and not iteration_idx > 15:
                    red_factor_idx = np.random.choice([i for i in allowed if i not in lowest_power_elements])
                    mutated_gene[red_factor_idx*self.n_params + power_position] -= 1
                else:
                    incr_factor_idx = np.random.choice([i for i in allowed if i not in max_power_elements])
                    mutated_gene[incr_factor_idx*self.n_params + power_position] -= 1
            else:
                red_factor_idx = np.random.choice([i for i in allowed if i not in lowest_power_elements])
                mutated_gene[red_factor_idx*self.n_params + power_position] -= 1
                incr_factor_idx = np.random.choice([i for i in allowed if i not in max_power_elements])
                mutated_gene[incr_factor_idx*self.n_params + power_position] -= 1
            if Check_Unqueness(mutated_gene, background_terms):
                self.gene = mutated_gene
                return            
        
#    def Mutate_params()
        
    def Mutate_old(self, background_terms, allowed_factors, reverse_mutation_prob = 0.1):

        allowed = list(np.nonzero(allowed_factors)[0])
        allowed.remove(0)

        if int(np.sum(self.gene[1:])) == 0: # Case of constant term
            iteration_idx = 0; max_attempts = 15
            while iteration_idx < max_attempts:
                mutated_gene = np.copy(self.gene)
                new_factor_idx = np.random.choice(allowed)
                mutated_gene[new_factor_idx*self.max_power] = 1
                if Check_Unqueness(mutated_gene, background_terms):
                    self.gene = mutated_gene
                    return
                iteration_idx += 1
                
            while True:
                mutated_gene = np.copy(self.gene)
                new_factor_idx = np.random.choice(allowed, size = 2)
                mutated_gene[new_factor_idx[0] * self.max_power] = 1; mutated_gene[new_factor_idx[1] * self.max_power] = 1
                if Check_Unqueness(mutated_gene, background_terms):
                    self.gene = mutated_gene
                    return
        
        max_power_elements = [idx for idx in range(len(self.tokens)) if self.gene[idx*self.max_power + self.max_power - 1] == 1]
        zero_power_elements = [idx for idx in range(len(self.tokens)) if self.gene[idx*self.max_power] == 0]
        
        iteration_idx = 0; max_attempts = 15
        total_power = int(np.sum(self.gene[1:]))
        
        while True:
            mutated_gene = np.copy(self.gene)
            if np.random.uniform(0, 1) <= reverse_mutation_prob or iteration_idx > 15:
                mutation_type = np.random.choice(['Reduction', 'Increasing'])
                if mutation_type == 'Reduction' or total_power >= self.max_factors_in_term and not iteration_idx > 15:
                    red_factor_idx = np.random.choice([i for i in allowed if i not in zero_power_elements])
                    addendum = self.max_power - 1
                    while mutated_gene[red_factor_idx*self.max_power + addendum] == 0:
                        addendum -= 1
                    mutated_gene[red_factor_idx*self.max_power + addendum] = 0
                else:
                    incr_factor_idx = np.random.choice([i for i in allowed if i not in max_power_elements])
                    addendum = 0
                    while mutated_gene[incr_factor_idx*self.max_power + addendum] == 1:
                        addendum += 1
                    mutated_gene[incr_factor_idx*self.max_power + addendum] = 1 
            else:
                red_factor_idx = np.random.choice([i for i in allowed if i not in zero_power_elements])
                addendum = self.max_power - 1
                while mutated_gene[red_factor_idx*self.max_power + addendum] == 0:
                    addendum -= 1
                mutated_gene[red_factor_idx*self.max_power + addendum] = 0                
                incr_factor_idx = np.random.choice([i for i in allowed if i not in max_power_elements])
                addendum = 0
                while mutated_gene[incr_factor_idx*self.max_power + addendum] == 1:
                    addendum += 1
                mutated_gene[incr_factor_idx*self.max_power + addendum] = 1
            if Check_Unqueness(mutated_gene, background_terms):
                self.gene = mutated_gene
                return            
        
    def Evaluate(self, evaluator, normalize, eval_params):
        return evaluator(self.gene, normalize, eval_params)