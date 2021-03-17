#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:33:34 2020

@author: mike_ubuntu
"""

import numpy as np
import copy

def memory_assesment():
    try:
        h=hpy()
    except NameError:
        from guppy import hpy
        h=hpy()
    print(h.heap())
    del h

def factor_params_to_str(factor, set_default_power = False):
    param_label= tuple(factor.params)            
    return (factor.label, param_label)

def form_label(x, y):
    return x + ' * ' + y.cache_label if len(x) > 0 else x + y.cache_label

def Detect_Similar_Terms(base_equation_1, base_equation_2):
    equation_1 = copy.deepcopy(base_equation_1); equation_2 = copy.deepcopy(base_equation_2)
    same_terms_from_eq1 = []
    same_terms_from_eq2 = []    
    
    similar_terms_from_eq1 = []
    similar_terms_from_eq2 = []
    
    different_terms_from_eq1 = []
    different_terms_from_eq2 = []    
    
    for eq1_term in base_equation_1.structure:     
        for eq2_term in base_equation_2.structure:
            if eq1_term == eq2_term:
                same_terms_from_eq1.append(eq1_term); same_terms_from_eq2.append(eq2_term);
                equation_1.structure.remove(eq1_term); equation_2.structure.remove(eq2_term); break
            elif set([token.label for token in eq1_term.structure]) == set([token.label for token in eq2_term.structure]) and len(eq1_term.structure) == len(eq2_term.structure):
                similar_terms_from_eq1.append(eq1_term); similar_terms_from_eq2.append(eq2_term); 
                try:
                    equation_1.structure.remove(eq1_term)
                    equation_2.structure.remove(eq2_term); 
                except ValueError:
                    print(eq1_term.text_form, [factor.text_form for factor in eq1_term.structure])
                    print([(term == eq1_term, term.text_form, [factor.text_form for factor in eq1_term.structure], (term.structure[0] == eq1_term.structure[0]), (all([any([other_elem == self_elem for other_elem in eq1_term.structure]) for self_elem in term.structure]) and 
                all([any([other_elem == self_elem for self_elem in term.structure]) for other_elem in eq1_term.structure]) and 
                len(term.structure) == len(eq1_term.structure))) for term in equation_1.structure])
                    print([(term == eq1_term, term.text_form, [factor.text_form for factor in eq1_term.structure]) for term in base_equation_1.structure])
                    raise ValueError
                break

    for term_idx in np.arange(len(equation_1.structure)):
        different_terms_from_eq1.append(equation_1.structure[term_idx]); different_terms_from_eq2.append(equation_2.structure[term_idx])
    return [same_terms_from_eq1, similar_terms_from_eq1, different_terms_from_eq1], [same_terms_from_eq2, similar_terms_from_eq2, different_terms_from_eq2]
        

def filter_powers(gene):    # Разобраться и переделать
    gene_filtered = []
    for token_idx in range(len(gene)):
        total_power = gene.count(gene[token_idx])
        powered_token = copy.deepcopy(gene[token_idx])
        
        power_idx = np.inf
        for param_idx, param_info in powered_token.params_description.items():
            if param_info['name'] == 'power': 
                power_idx = param_idx
                break
        powered_token.params[power_idx] = total_power
        if powered_token not in gene_filtered:
            gene_filtered.append(powered_token)
    return gene_filtered

def Bind_Params(zipped_params):
    param_dict = {}
    for token_props in zipped_params:
        param_dict[token_props[0]] = token_props[1]
    return param_dict

def Slice_Data_3D(matrix, part = 4, part_tuple = None):     # Input matrix slicing for separate domain calculation
    if part_tuple:
        for i in range(part_tuple[0]):
            for j in range(part_tuple[1]):
                yield matrix[:, i*int(matrix.shape[1]/float(part_tuple[0])):(i+1)*int(matrix.shape[1]/float(part_tuple[0])), 
                             j*int(matrix.shape[2]/float(part_tuple[1])):(j+1)*int(matrix.shape[2]/float(part_tuple[1]))], i, j   
    part_dim = int(np.sqrt(part))
    for i in range(part_dim):
        for j in range(part_dim):
            yield matrix[:, i*int(matrix.shape[1]/float(part_dim)):(i+1)*int(matrix.shape[1]/float(part_dim)), 
                         j*int(matrix.shape[2]/float(part_dim)):(j+1)*int(matrix.shape[2]/float(part_dim))], i, j

def Define_Derivatives(var_name = 'u', dimensionality = 1, max_order = 2):
    var_names = [var_name,]
    for var_idx in range(dimensionality):
        for order in range(max_order):
            if order == 0:
                var_names.append('d'+ var_name + '/dx'+str(var_idx+1))
            else:
                var_names.append('d^'+str(order+1) + var_name + '/dx'+str(var_idx+1)+'^'+str(order+1))
    return var_names    

def Create_Var_Matrices(U_input, max_order = 3):
    var_names = ['u']

    for var_idx in range(U_input.ndim):
        for order in range(max_order):
            if order == 0:
                var_names.append('du/dx'+str(var_idx+1))
            else:
                var_names.append('d^'+str(order+1)+'u/dx'+str(var_idx+1)+'^'+str(order+1))

    variables = np.ones((len(var_names),) + U_input.shape)      
    return variables, tuple(var_names)


def Prepare_Data_matrixes(raw_matrix, dim_info):
    resulting_matrix = np.reshape(raw_matrix, dim_info)
    return resulting_matrix 


def Decode_Gene(gene, token_names, parameter_labels, n_params = 1):
    term_dict = {}
    for token_idx in range(0, gene.shape[0], n_params):
        term_params = {}#coll.OrderedDict()
        for param_idx in range(0, n_params):
            term_params[parameter_labels[param_idx]] = gene[token_idx*n_params + param_idx]    
        term_dict[token_names[int(token_idx/n_params)]] = term_params
    return term_dict


def Encode_Gene(label_dict, token_names, parameter_labels, n_params = 1):
    gene = np.zeros(shape = len(token_names) * n_params)

    for i in range(len(token_names)):
        if token_names[i] in label_dict:
            #print(token_names, label_dict[token_names[i]])
            for key, value in label_dict[token_names[i]].items():
                gene[i*n_params + parameter_labels.index(key)] = value
    return gene

def Population_Sort(input_population):
    pop_sorted = [x for x, _ in sorted(zip(input_population, [individual.fitness_value for individual in input_population]), key=lambda pair: pair[1])]
    return list(reversed(pop_sorted))