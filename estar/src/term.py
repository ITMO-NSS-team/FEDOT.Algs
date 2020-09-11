#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:07:21 2020

@author: mike_ubuntu
"""

import numpy as np
import copy
import warnings
import time

from collections import OrderedDict
from functools import reduce

from src.factor import Factor
from src.supplementary import Filter_powers

def Check_Unqueness(obj, background):
    return not any([elem == obj for elem in background]) 
   
    
def normalize_ts(Input):
    Matrix = np.copy(Input)
    for i in np.arange(Matrix.shape[0]):
        std = np.std(Matrix[i])
        if std != 0:
            Matrix[i] = (Matrix[i] - np.mean(Matrix[i])) / std
        else:
            Matrix[i] = 1
    return Matrix


class Term:
    def __init__(self, tokens, passed_term = None, max_factors_in_term = 2, forbidden_tokens = None):#):

        """
;
            
        """
        
        self.tokens = tokens
        self.max_factors_in_term = max_factors_in_term
#        self.forbidden = forbidden_tokens
        
        if type(passed_term) == type(None):
            self.Randomize(forbidden_tokens)     
        else:
            self.Defined(passed_term)
            

    def Defined(self, passed_term):   
        self.gene = []

        if type(passed_term) == list or type(passed_term) == tuple:
            for i in range(len(passed_term)):
                token_family = [token_family for token_family in self.tokens if passed_term[i] in token_family.tokens][0]
                self.gene.append(Factor(passed_term[i], token_family))
                parameter_selection = copy.deepcopy(token_family.token_params)                    
                for param, interval in token_family.token_params.items():
                    if interval[0] == interval[1]:
                        parameter_selection[param] = interval[1]
                        continue
                    if param == 'power':
                        parameter_selection[param] = 1
                        continue
                    parameter_selection[param] = (np.random.randint(interval[0], interval[1]) 
                                              if isinstance(interval[0], int) else np.random.uniform(interval[0], interval[1]))
                self.gene[i].Set_parameters(**parameter_selection)
        else:
            token_family = [token_family for token_family in self.tokens if passed_term in token_family.tokens][0]
            self.gene.append(Factor(passed_term[i], token_family))
            parameter_selection = copy.deepcopy(token_family.token_params)   
            for param, interval in token_family.token_params.items():
                if interval[0] == interval[1]:
                    parameter_selection[param] = interval[1]
                    continue                    
                if param == 'power':
                    parameter_selection[param] = 1
                    continue
                parameter_selection[param] = (np.random.randint(interval[0], interval[1])
                                              if isinstance(interval[0], int) else np.random.uniform(interval[0], interval[1]))
            self.gene[0].Set_parameters(**parameter_selection)


    def Randomize(self, forbidden_factors = None):    
        #self.gene = []
        
        mandatory_tokens = [token_family for token_family in self.tokens if token_family.status['mandatory']]
            
        
        if len(mandatory_tokens) > self.max_factors_in_term:
            raise ValueError('Number of mandatory tokens for term exceeds allowed number of factors')
        factors_num = np.random.randint(len(mandatory_tokens), self.max_factors_in_term +1)
#        if type(None) != type(forbidden_factors): print(type(forbidden_factors), type(forbidden_factors[0]))
        
#        if type(forbidden_tokens) != type(None): 
#            self.occupied_tokens_labels.extend(forbidden_tokens)
#            print('forbidden: ', forbidden_tokens)
#            print('occupied: ', self.occupied_tokens_labels)
        
#        for token_family in mandatory_tokens:
#            new_tokens = copy.copy(token_family.tokens)
#            for token in token_family.tokens:
#                if token in self.occupied_tokens_labels:
#                    new_tokens.remove(token)
#            token_family.tokens = new_tokens

        while True:
            self.occupied_tokens_labels = []
            self.gene = []
            for mandatory_token_family_idx in range(len(mandatory_tokens)):
                selected_label = np.random.choice(mandatory_tokens[mandatory_token_family_idx].tokens)
                if mandatory_tokens[mandatory_token_family_idx].status['unique_specific_token']:      # 'single' - токен не может повторяться (даже с разными пар-ми) 
                    self.occupied_tokens_labels.append(selected_label)
                elif mandatory_tokens[mandatory_token_family_idx].status['unique_token_type']:  # 'unique_token' - в слагаемом не может быть 2 мн-теля выбранного типа
                    self.occupied_tokens_labels.extend([token_label for token_label in mandatory_tokens[mandatory_token_family_idx].tokens])
                
                self.gene.append(Factor(selected_label, mandatory_tokens[mandatory_token_family_idx]))
    
                parameter_selection = copy.deepcopy(mandatory_tokens[mandatory_token_family_idx].token_params)
                for param, interval in mandatory_tokens[mandatory_token_family_idx].token_params.items():
                    if interval[0] == interval[1]:
                        parameter_selection[param] = interval[1]
                        continue
                    if param == 'power':
                        parameter_selection[param] = 1
                    else:
                        parameter_selection[param] = (np.random.randint(interval[0], interval[1]) 
                                                  if isinstance(interval[0], int) else np.random.uniform(interval[0], interval[1]))
                self.gene[mandatory_token_family_idx].Set_parameters(**parameter_selection)            
                
            for i in np.arange(len(mandatory_tokens), factors_num):
                total_tokens_num = np.sum([len(token_family.tokens) for token_family in self.available_tokens], dtype = np.float32)
                selected_token = np.random.choice(self.available_tokens,
                                                 p = [len(token_family.tokens)/total_tokens_num for token_family in self.available_tokens])
                selected_label = np.random.choice(selected_token.tokens)
    
                if selected_token.status['unique_specific_token']:      # 'unique_token_type' - токен не может повторяться (даже с разными пар-ми)
                    self.occupied_tokens_labels.append(selected_label)
                elif selected_token.status['unique_token_type']:  # 'unique_specific_token' - в слагаемом не может быть 2 мн-теля выбранного типа
                    self.occupied_tokens_labels.extend([token_label for token_label in selected_token.tokens])
                
                self.gene.append(Factor(selected_label, selected_token))
    
                parameter_selection = copy.deepcopy(selected_token.token_params)
                for param, interval in selected_token.token_params.items():
                    if interval[0] == interval[1]:
                        parameter_selection[param] = interval[1]
                        continue
                    if param == 'power':
                        parameter_selection[param] = 1
                    else:
                        parameter_selection[param] = (np.random.randint(interval[0], interval[1]) 
                                                  if isinstance(interval[0], int) else np.random.uniform(interval[0], interval[1]))
                self.gene[i].Set_parameters(**parameter_selection)
            self.gene = Filter_powers(self.gene)
            if type(forbidden_factors) == type(None):
                break
            elif all([(Check_Unqueness(factor, forbidden_factors) or not factor.token.status['unique_for_right_part']) for factor in self.gene]):
                break
#        print('after random initialization:', self.text_form)


    def __eq__(self, other):
        res = (all([any([other_factor == self_factor for other_factor in other.gene]) for self_factor in self.gene]) and 
                all([any([other_factor == self_factor for self_factor in self.gene]) for other_factor in other.gene]))        
#        if res:
#            print(self.text_form, other.text_form)
        return res

    def Remove_Dublicated_Factors(self, target, background_terms):
        cleared_num = 0
        cleared_token_families = []
        gene_cleared = []
        for factor_idx in range(len(self.gene)):
            if all([(elem != self.gene[factor_idx] or not elem.token.status['unique_for_right_part']) for elem in target.gene]):
                gene_cleared.append(self.gene[factor_idx])
            else:
#                print('clearing due to presence in the target term:', self.gene[factor_idx].text_form)
                cleared_token_families.append(self.gene[factor_idx].token.type)
                cleared_num += 1
#            if self.gene[factor_idx].label in forbidden_terms:
#                cleared_token_families.append([family.type for family in self.tokens if self.gene[factor_idx].label in family.tokens][0])
#                cleared_num += 1
#            else:
#                gene_cleared.append(self.gene[factor_idx])

#        default_occupied = copy.copy(self.occupied_tokens_labels)        
#        for term in forbidden_terms:
#            if term not in self.occupied_tokens_labels: self.occupied_tokens_labels.append(term)
            
        token_selection = self.available_tokens
#        print('token_selection during RDF:', [token.tokens for token in token_selection])
        filling_try = 0
        while True:
            filling_try += 1

            cleared_token_families_copy = copy.copy(cleared_token_families)
            if filling_try == 10:
                cleared_num += 1
                cleared_token_families_copy.append(np.random.choice(cleared_token_families))
                
            if filling_try == 100:
                #print('forbidden:', [token for token in forbidden_terms])
                print('background:', [token.text_form for token in background_terms])
                print('self:', self.text_form, cleared_num)
                warnings.warn('Algorithm is having issues with creation of unique terms: reduce length of equation or increase token pool')
                
            gene_filled = copy.deepcopy(gene_cleared)
            iter_selection_pool = copy.deepcopy(token_selection)
            for i in np.arange(cleared_num):
                while True:
                    iter_selection_pool_copy = copy.copy(iter_selection_pool)
                    cleared_token_families_copy_2 = copy.copy(cleared_token_families_copy)
                    
                    total_tokens_num = np.sum([len(token_family.tokens) for token_family in [token_family_loc for token_family_loc in iter_selection_pool_copy if token_family_loc.type in cleared_token_families_copy]], dtype = np.float32)
                    try:
                        selected_token = np.random.choice([token_family_loc for token_family_loc in iter_selection_pool_copy if token_family_loc.type in cleared_token_families_copy_2],
                                                         p = [len(token_family.tokens)/total_tokens_num for token_family in
                                                              [token_family_loc for token_family_loc in iter_selection_pool_copy if token_family_loc.type in cleared_token_families_copy_2]]) 
                    except ValueError:
                        print('cleared tokens:', cleared_token_families_copy_2, cleared_token_families)
                        print()
                        print('current pool:', [token_family_loc for token_family_loc in iter_selection_pool_copy if token_family_loc.type in cleared_token_families_copy_2])
                        print('occupied tokens:', self.occupied_tokens_labels)
                        print('available:', [token_family.tokens for token_family in token_selection])
                        raise ValueError('The framework had issues with creating filtering the tokens: the selection pool is emptied.') from None
                    
                    try:
                        selected_label = np.random.choice(selected_token.tokens)
                    except ValueError:
                        print('selected label:', selected_token.type, selected_token.tokens)
                        print('cleared tokens:', cleared_token_families_copy_2, cleared_token_families)
                        print('current pool:', [token_family_loc.tokens for token_family_loc in iter_selection_pool_copy if token_family_loc.type in cleared_token_families_copy_2])
                        print('occupied tokens:', self.occupied_tokens_labels)
                        print('available:', [token_family.tokens for token_family in token_selection])
                        print('self:', self.text_form)
                        for term in background_terms:
                            print(term.text_form)                    
                        raise ValueError('The framework had issues with selecting the token. Probable cause: lack of initialized tokens') from None
                        
                    cleared_token_families_copy_2.remove(selected_token.type)
                    iter_selection_pool_copy[iter_selection_pool_copy.index(selected_token)].tokens.remove(selected_label)
                    
                    temp_token = Factor(selected_label, selected_token)
                    parameter_selection = copy.deepcopy(selected_token.token_params)
                    for param, interval in selected_token.token_params.items():
                        if interval[0] == interval[1]:
                            parameter_selection[param] = interval[1]
                            continue                    
                        if param == 'power':
                            parameter_selection[param] = 1
                            continue
                        parameter_selection[param] = (np.random.randint(interval[0], interval[1])
                                                      if isinstance(interval[0], int) else np.random.uniform(interval[0], interval[1]))
                    temp_token.Set_parameters(**parameter_selection)
#                    if all([(Check_Unqueness(factor, target.gene) or factor.token.status['unique_for_right_part']) for factor in new_term.gene]):
                    if (Check_Unqueness(temp_token, target.gene) or not temp_token.token.status['unique_for_right_part']):
                        gene_filled.append(temp_token)
                        break
             
            new_term = copy.deepcopy(self)
            new_term.gene = gene_filled
            if Check_Unqueness(new_term, background_terms):# and all([(Check_Unqueness(factor, target.gene) or factor.token.status['unique_for_right_part']) for factor in new_term.gene]):
                del new_term
                break 
#        self.occupied_tokens_labels = default_occupied      
       
        self.gene = gene_filled
#        print('clearing in accordance the ')
#        print(target.text_form)
#        print(self.text_form)
                
    def Evaluate(self, normalize = True):
        value = reduce(lambda x, y: x*y, [token() for token in self.gene])
#        print(value.shape, value.ndim)
        if normalize and np.ndim(value) != 1:
            value = normalize_ts(value)    
        elif normalize and np.ndim(value) == 1 and np.std(value) != 0:
            value = (value - np.mean(value))/np.std(value)
        elif normalize and np.ndim(value) == 1 and np.std(value) == 0:
            value = (value - np.mean(value))   
        value = value.reshape(value.size)
        return value    

    def Reset_occupied_tokens(self):    # Дописать
        occupied_tokens_new = []
        for factor in self.gene:
            for token_family in self.tokens:
                if factor.label in token_family.tokens and token_family.status['unique_token_type']:
                    occupied_tokens_new.extend([token for token in token_family.tokens])
                elif factor.label in token_family.tokens and token_family.status['unique_specific_token']:
                    occupied_tokens_new.append(factor.label)
            #[token_family.status['unique_token_type'] for token_family in self.tokens if factor.label in token_family.tokens][0]:
#                occupied_tokens_new.extend([[token for token in token_family.tokens] for token_family in self.tokens]) #! 
#                continue
#            if [token_family.status['unique_specific_token'] for token_family in self.tokens if factor.label in token_family.tokens][0]:
#                occupied_tokens_new.append(factor.label)
        self.occupied_tokens_labels = occupied_tokens_new

    @property
    def available_tokens(self):
        available_tokens = []
        for token in self.tokens:
            if not all([label in self.occupied_tokens_labels for label in token.tokens]):
                token_new = copy.deepcopy(token)
                token_new.tokens = [label for label in token.tokens if label not in self.occupied_tokens_labels]
                available_tokens.append(token_new)
        return available_tokens

    @property
    def total_params(self):
        return max(sum([len(factor.params) - 1 for factor in self.gene]), 1)
    
    @property
    def text_form(self):
        form = ''
        for token_idx in range(len(self.gene)):
            form += self.gene[token_idx].text_form
            if token_idx < len(self.gene) - 1:
                form += ' * '
        return form