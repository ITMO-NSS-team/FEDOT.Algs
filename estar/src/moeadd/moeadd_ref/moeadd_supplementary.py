#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:04:57 2020

@author: mike_ubuntu
"""

from copy import deepcopy
import numpy as np
    

def check_dominance(target, compared_with) -> bool:
    flag = False
    for obj_fun_idx in range(len(target.obj_fun)):
        if target.obj_fun[obj_fun_idx] <= compared_with.obj_fun[obj_fun_idx]:
            if target.obj_fun[obj_fun_idx] < compared_with.obj_fun[obj_fun_idx]:
                flag = True
        else:
            return False
    return flag

    # return (all([target.obj_fun[obj_fun_idx] <= compared_with.obj_fun[obj_fun_idx] for obj_fun_idx in np.arange(target.obj_fun.size)]) and
    #         any([target.obj_fun[obj_fun_idx] < compared_with.obj_fun[obj_fun_idx] for obj_fun_idx in np.arange(target.obj_fun.size)]))
    
def NDL_update(new_solution, levels) -> list:   # efficient_NDL_update
    moving_set = {new_solution}
    new_levels = levels
    for level_idx in np.arange(len(levels)):
        moving_set_new = set()
        for ms_idx, moving_set_elem in enumerate(moving_set):
            if np.any([check_dominance(solution, moving_set_elem) for solution in new_levels[level_idx]]):
                moving_set_new.add(moving_set_elem)
            elif (not np.any([check_dominance(solution, moving_set_elem) for solution in new_levels[level_idx]]) and 
                  not np.any([check_dominance(moving_set_elem, solution) for solution in new_levels[level_idx]])):
                new_levels[level_idx].append(moving_set_elem)#; completed_levels = True
            elif np.all([check_dominance(moving_set_elem, solution) for solution in levels[level_idx]]):
                temp_levels = new_levels[level_idx:]
                new_levels[level_idx:] = []
                new_levels.append([moving_set_elem,]); new_levels.extend(temp_levels)#; completed_levels = True
            else:
                dominated_level_elems = [level_elem for level_elem in new_levels[level_idx] if check_dominance(moving_set_elem, level_elem)]
                non_dominated_level_elems = [level_elem for level_elem in new_levels[level_idx] if not check_dominance(moving_set_elem, level_elem)]
                non_dominated_level_elems.append(moving_set_elem)
                new_levels[level_idx] = non_dominated_level_elems

                for element in dominated_level_elems:
                    moving_set_new.add(element)
        moving_set = moving_set_new
        if not len(moving_set):
            break
    if len(moving_set):
        new_levels.append(list(moving_set))
    if len(new_levels[len(new_levels)-1]) == 0:
        _ = new_levels.pop()
    return new_levels
            
    
def fast_non_dominated_sorting(population) -> list:
    levels = []; ranks = np.empty(len(population))
    domination_count = np.zeros(len(population)) # Число элементов, доминирующих над i-ым кандидиатом
    dominated_solutions = [[] for elem_idx in np.arange(len(population))] # Индексы элементов, над которыми доминирует i-ый кандидиат
    current_level_idxs = []
    for main_elem_idx in np.arange(len(population)):
        for compared_elem_idx in np.arange(len(population)):
            if main_elem_idx == compared_elem_idx:
                continue
            if check_dominance(population[compared_elem_idx], population[main_elem_idx]):
                domination_count[main_elem_idx] += 1 
            elif check_dominance(population[main_elem_idx], population[compared_elem_idx]):
                dominated_solutions[main_elem_idx].append(compared_elem_idx)
        if domination_count[main_elem_idx] == 0:
            current_level_idxs.append(main_elem_idx); ranks[main_elem_idx] = 0
    levels.append([population[elem_idx] for elem_idx in current_level_idxs])
    
    level_idx = 0
    while len(current_level_idxs) > 0:
        new_level_idxs = []
        for main_elem_idx in current_level_idxs:
            for dominated_elem_idx in dominated_solutions[main_elem_idx]:
                domination_count[dominated_elem_idx] -= 1
                if domination_count[dominated_elem_idx] == 0:
                    ranks[dominated_elem_idx] = level_idx + 1
                    new_level_idxs.append(dominated_elem_idx)
    
        if len(new_level_idxs): levels.append([population[elem_idx] for elem_idx in new_level_idxs])
        level_idx += 1
        current_level_idxs = new_level_idxs
    return levels

    
def slow_non_dominated_sorting(population) -> list:
    locked_idxs = []
    levels = []; levels_elems = 0
    while len(population) > levels_elems:
        processed_idxs = []
        for main_elem_idx in np.arange(len(population)):
            if not main_elem_idx in locked_idxs:
                dominated = False
                for compared_elem_idx in np.arange(len(population)):
                    if main_elem_idx == compared_elem_idx or compared_elem_idx in locked_idxs:
                        continue
                    if check_dominance(population[compared_elem_idx], population[main_elem_idx]):
                        dominated = True
                if not dominated:
                    processed_idxs.append(main_elem_idx)
        locked_idxs.extend(processed_idxs); levels_elems += len(processed_idxs)
        levels.append([population[elem_idx] for elem_idx in processed_idxs])
    return levels
 
    
def acute_angle(vector_a, vector_b) -> float:
    return np.arccos(np.dot(vector_a, vector_b)/(np.sqrt(np.dot(vector_a, vector_a))*np.sqrt(np.dot(vector_b, vector_b))))


class Constraint(object):
    def __init__(self, *args):
        pass
    
    def __call__(self, *args):
        pass
    
    
class Inequality(Constraint):
    def __init__(self, g):
        '''
            
        Inequality assumed in format g(x) >= 0
        
        '''
        self._g = g
        
    def __call__(self, x) -> float:
        return - self._g(x) if self._g(x) < 0 else 0
    

class Equality(Constraint):
    def __init__(self, h):
        '''
            
        Equality assumed in format h(x) = 0
        
        '''
        self._h = h
        
    def __call__(self, x) -> float:
        return np.abs(self._h(x))    