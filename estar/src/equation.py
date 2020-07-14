#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:43:10 2020

@author: mike_ubuntu
"""

import numpy as np
import copy
from functools import reduce
from sklearn.linear_model import LinearRegression, Lasso, Ridge 
from abc import ABC, abstractmethod

from src.term import Term
from src.term import Check_Unqueness

class Equation(object):
    def __init__(self, tokens, basic_terms, terms_number = 6, max_factors_in_term = 2): 

        """

        Class for the single equation for the dynamic system.
            
        attributes:
            terms : list of Term objects \r\n
            List, containing all terms of the equation; first 2 terms are reserved for constant value and the input function;
        
            target_idx : int \r\n
            Index of the target term, selected in the Split phase;
        
            target : 1-d array of float \r\n
            values of the Term object, reshaped into 1-d array, designated as target for application in sparse regression;
            
            features : matrix of float \r\n
            matrix, composed of terms, not included in target, value columns, designated as features for application in sparse regression;
        
            fitness_value : float \r\n
            Inverse value of squared error for the selected target 2function and features and discovered weights; 
        
            estimator : sklearn estimator of selected type \r\n
        
        parameters:

            Matrix of derivatives: first axis through various orders/coordinates in order: ['1', 'f', all derivatives by one coordinate axis
            in increasing order, ...]; second axis: time, further - spatial coordinates;

            tokens : list of strings \r\n
            Symbolic forms of functions, including derivatives;

            terms_number : int, base value of 6 \r\n
            Maximum number of terms in the discovered equation; 

            max_factors_in_term : int, base value of 2\r\n
            Maximum number of factors, that can form a term (e.g. with 2: df/dx_1 * df/dx_2)

        """

        self.n_immutable = len(basic_terms)
        self.tokens = tokens
        self.terms = []
        self.terms_number = terms_number; self.max_factors_in_term = max_factors_in_term
        if (terms_number < self.n_immutable): 
            raise Exception('Number of terms ({}) is too low to contain all mandatory ones'.format(terms_number))        
            
        self.terms.extend([Term(self.tokens, passed_term = label, max_factors_in_term = self.max_factors_in_term) for label in basic_terms])

        for i in range(len(basic_terms), terms_number):
            check_test = 0
            while True:
                check_test += 1 
                new_term = Term(self.tokens, max_factors_in_term = self.max_factors_in_term, passed_term = None)
                if Check_Unqueness(new_term, self.terms):
                    break
            self.terms.append(new_term)


    def __eq__(self, other):
        if all([any([self_term == other_term for other_term in other.terms]) for self_term in self.terms]):
            return True
        else:
            return False


    def Evaluate_equation(self, normalize = True):
        self.target = self.terms[self.target_idx].Evaluate(normalize)
        
        for feat_idx in range(len(self.terms)):
            if feat_idx == 0:
                self.features = self.terms[feat_idx].Evaluate(normalize)
            elif feat_idx != 0 and self.target_idx != feat_idx:
                temp = self.terms[feat_idx].Evaluate(normalize)
                self.features = np.vstack([self.features, temp])
            else:
                continue
        self.features = np.transpose(self.features)
        

    def Split_data(self, target_idx = None): 
        
        '''
        
        Separation of target term from features & removal of factors, that are in target, from features
        
        '''
        
        self.target_idx = target_idx if type(target_idx) != type(None) else np.random.randint(low = 1, high = len(self.terms)-1)
        
                           
        for feat_idx in range(len(self.terms)):
            if feat_idx == 0:
                continue
            elif feat_idx != 0 and self.target_idx != feat_idx:
                self.terms[feat_idx].Remove_Dublicated_Factors(self.forbidden_token_labels, self.terms[:feat_idx]+self.terms[feat_idx+1:])
            else:
                continue        
     
        
#    def Show_terms(self):
#        for term in self.terms:
#            print([(factor.label, factor.params) for factor in term.gene])             
        
    @property 
    def forbidden_token_labels(self):
        target_symbolic = [factor.label for factor in self.terms[self.target_idx].gene]
        forbidden_tokens = set()

        for token_family in self.tokens:
            for token in token_family.tokens:
                if token in target_symbolic and token_family.status['unique_for_right_part']:
                    forbidden_tokens.add(token)        
#        print(forbidden_tokens)
        return forbidden_tokens
        
    @property
    def text_form(self):
        form = ''
        for term_idx in range(len(self.terms)):
            if term_idx != self.target_idx:
                form += str(self.weights[term_idx]) if term_idx < self.target_idx else str(self.weights[term_idx-1])
                form += ' * ' + self.terms[term_idx].text_form + ' + '
#            if term_idx < len(self.terms) - 1:
        form += 'const = ' + self.terms[self.target_idx].text_form
        return form
    