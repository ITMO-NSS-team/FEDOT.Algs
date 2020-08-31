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

def Check_Unqueness(term, background):
    return not any([eq_term == term for eq_term in background]) 

   
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
    def __init__(self, tokens, passed_term = None, max_factors_in_term = 2):#, forbidden_tokens = None):

        """
;
            
        """
        
        self.tokens = tokens
        self.max_factors_in_term = max_factors_in_term
#        self.forbidden = forbidden_tokens
        
        if type(passed_term) == type(None):
            self.Randomize()     
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


    def Randomize(self):    
        self.gene = []
        mandatory_tokens = [token_family for token_family in self.tokens if token_family.status['mandatory']]
        
        if len(mandatory_tokens) > self.max_factors_in_term:
            raise ValueError('Number of mandatory tokens for term exceeds allowed number of factors')
        factors_num = np.random.randint(len(mandatory_tokens), self.max_factors_in_term +1)
        
        self.occupied_tokens_labels = []
#        permitted_tokens = copy.deepcopy(self.tokens)
        for mandatory_token_family_idx in range(len(mandatory_tokens)):
#            selected_type = mandatory_tokens[mandatory_token_type_idx].type
            selected_label = np.random.choice(mandatory_tokens[mandatory_token_family_idx].tokens)
            if mandatory_tokens[mandatory_token_family_idx].status['unique_specific_token']:      # 'single' - токен не может повторяться (даже с разными пар-ми)
#                permitted_tokens[permitted_tokens.index(mandatory_tokens[mandatory_token_type_idx])].token.remove(selected_label) 
                self.occupied_tokens_labels.append(selected_label)
            elif mandatory_tokens[mandatory_token_family_idx].status['unique_token_type']:  # 'unique_token' - в слагаемом не может быть 2 мн-теля выбранного типа
#                del permitted_tokens[permitted_tokens.index(mandatory_tokens[mandatory_token_type_idx])]
                self.occupied_tokens_labels.extend([token_label for token_label in mandatory_tokens[mandatory_token_family_idx].tokens])
#                print('had', self.occupied_tokens_labels)
#                print('added', [token_label for token_label in mandatory_tokens[mandatory_token_family_idx].tokens])
#                print('now', self.occupied_tokens_labels)
            
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
#            print(selected_label, selected_token.status['unique_specific_token'], selected_token.status['unique_token_type'])

            if selected_token.status['unique_specific_token']:      # 'unique_token_type' - токен не может повторяться (даже с разными пар-ми)
                self.occupied_tokens_labels.append(selected_label)
#                permitted_tokens[permitted_tokens.index(selected_token)].token.remove(selected_label) 
            elif selected_token.status['unique_token_type']:  # 'unique_specific_token' - в слагаемом не может быть 2 мн-теля выбранного типа
                self.occupied_tokens_labels.extend([token_label for token_label in selected_token.tokens])
#                print(selected_label, selected_token.status['unique_token_type'])
#                print('had', self.occupied_tokens_labels)
#                print('added', [token_label for token_label in selected_token.tokens])
#                print('now', self.occupied_tokens_labels)

#                del permitted_tokens[permitted_tokens.index(selected_token)]
            
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



    def __eq__(self, other):
        res = (all([any([other_factor == self_factor for other_factor in other.gene]) for self_factor in self.gene]) and 
                all([any([other_factor == self_factor for self_factor in self.gene]) for other_factor in other.gene]))        
#        if res:
#            print(self.text_form, other.text_form)
        return res

    def Remove_Dublicated_Factors(self, forbidden_terms, background_terms):
        cleared_num = 0
#        print('before deleting:', self.text_form)
        gene_cleared = [] #copy.deepcopy(self.gene)
        for factor_idx in range(len(self.gene)):        # Переделать удаление в построение
            if self.gene[factor_idx].label in forbidden_terms:#set(reduce(lambda x, y: x.union(y), forbidden_terms.values())):
#                print('deleting', self.gene[factor_idx].label)
                cleared_num += 1
                #gene_cleared.remove(factor)
            else:
                gene_cleared.append(self.gene[factor_idx])
#        print('after deleting:', [factor.label for factor in gene_cleared])
                
#        print('before merging')
#        print('self.occupied... ', self.occupied_tokens_labels)
#        print('forbidden terms', forbidden_terms)
        for term in forbidden_terms:
            if term not in self.occupied_tokens_labels: self.occupied_tokens_labels.append(term)
#        self.occupied_tokens_labels = forbidden_terms
#        print([token.tokens for token in self.available_tokens])
        token_selection = self.available_tokens
#        print(cleared_num, forbidden_terms, [token.tokens for token in token_selection])
        
#        token_selection = copy.deepcopy(self.tokens)
#        for token_type_idx in range(len(token_selection)):
#            for token_idx in range(len(token_selection[token_type_idx].tokens)):
#                if token_selection[token_type_idx].tokens[token_idx] in forbidden_terms:
#                    del token_selection[token_type_idx].tokens[token_idx]

#        print('after merging')
#        print('self.occupied... ', self.occupied_tokens_labels)
#        print('available', [token_family.tokens for token_family in token_selection])
        
        filling_try = 0
        while True:
            filling_try += 1
            if filling_try == 10:
                cleared_num += 1
                
            if filling_try == 100:
                print('forbidden:', [token for token in forbidden_terms])
                print('background:', [token.text_form for token in background_terms])
                print('self:', self.text_form, cleared_num)
                warnings.warn('Algorithm is having issues with creation of unique terms: reduce length of equation or increase token pool')
                
            gene_filled = copy.deepcopy(gene_cleared)
            iter_selection_pool = copy.deepcopy(token_selection)
            for i in np.arange(cleared_num):
                total_tokens_num = np.sum([len(token_family.tokens) for token_family in iter_selection_pool], dtype = np.float32)
                selected_token = np.random.choice(iter_selection_pool,
                                                 p = [len(token_family.tokens)/total_tokens_num for token_family in iter_selection_pool]) 
                selected_label = np.random.choice(selected_token.tokens)
                iter_selection_pool[iter_selection_pool.index(selected_token)].tokens.remove(selected_label)
                
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
                                
                gene_filled.append(temp_token)
             
            new_term = copy.deepcopy(self)
            new_term.gene = gene_filled
            if Check_Unqueness(new_term, background_terms):
                del new_term
                break 
#        print('forbidden terms:', forbidden_terms)
#        print('before', self.text_form)
#        print('after', new_term.text_form)
        self.gene = gene_filled
        
                
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