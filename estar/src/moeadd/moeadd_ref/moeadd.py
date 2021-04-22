#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:29:18 2020

@author: mike_ubuntu
"""
import time

import numpy as np
from copy import deepcopy, copy
from functools import reduce
from moea_dd.src.moeadd_supplementary import fast_non_dominated_sorting,\
    slow_non_dominated_sorting, NDL_update, Equality, Inequality, acute_angle


class moeadd_solution(object):

    def __init__(self, x: np.ndarray, obj_funs: list):
        self.vals = x # генотип, в случая индивида - набор токенов + коэф. регуляризации
        self.obj_funs = obj_funs # функции-критерии - каждая делает преобразование генотипа в число

        self._obj_fun = None
        self._domain = None

        self.precomputed_value = False
        self.precomputed_domain = False
    
    @property
    def obj_fun(self) -> np.ndarray:
        if self.precomputed_value: 
            return self._obj_fun
        else:
            # в сумме формируется флоат-вектор, в который отображается индивид - многомерный фитнесс, фенотип.
            # формируется путем конкантенации чисел от каждого критерия
            self._obj_fun = np.fromiter(map(lambda obj_fun: obj_fun(self.vals), self.obj_funs), dtype=float)
            self.precomputed_value = True
            return self._obj_fun

    def get_domain(self, weights) -> int:
        if self.precomputed_domain:
            return self._domain
        else:
            self._domain = get_domain_idx(self, weights)
            self.precomputed_domain = True
            return self._domain
    
    def __eq__(self, other):

        if isinstance(other, type(self)):
            return self.vals == other.vals
        return False

    def __call__(self):
        return self.obj_fun
    
    # def __hash__(self):
    #     raise NotImplementedError('The hash needs to be defined in the subclass')

def get_domain_idx(solution, weights) -> int:
    if type(solution) == np.ndarray:
        return np.fromiter(map(lambda x: acute_angle(x, solution), weights), dtype=float).argmin()
    elif type(solution.obj_fun) == np.ndarray:
        return np.fromiter(map(lambda x: acute_angle(x, solution.obj_fun), weights), dtype=float).argmin()
    else:
        raise ValueError('Can not detect the vector of objective function for individ')
    

def penalty_based_intersection(sol_obj, weight, ideal_obj, penalty_factor = 1) -> float:
    d_1 = np.dot((sol_obj.obj_fun - ideal_obj), weight) / np.linalg.norm(weight)
    d_2 = np.linalg.norm(sol_obj.obj_fun - (ideal_obj + d_1 * weight/np.linalg.norm(weight)))
    return d_1 + penalty_factor * d_2


def population_to_sectors(population, weights): # Много жрёт
    solution_selection = lambda weight_idx: [solution for solution in population if solution.get_domain(weights) == weight_idx]
    return list(map(solution_selection, np.arange(len(weights))))    


def clear_list_of_lists(inp_list) -> list:
    return [elem for elem in inp_list if len(elem) > 0]

    
class pareto_levels(object):
    def __init__(self, population, sorting_method = fast_non_dominated_sorting, update_method = NDL_update):
        self._sorting_method = sorting_method
        self.population = population
        self._update_method = update_method
        self.levels = self._sorting_method(self.population)
        
    def sort(self):
        self.levels = self._sorting_method(self.population)
    
    def update(self, point):
        self.levels = self._update_method(point, self.levels)
        self.population.append(point)

    def delete_point(self, point):  # Разобраться с удалением.  Потенциально ошибка
#        print('deleting', point.vals)
        new_levels = []
        print('New pareto')
        for level in self.levels:
            print('New level processing')
            # temp = deepcopy(level)
            temp = []
            for element in level:
                if element is not point:
                    print('found point')
                    temp.append(element)
            if not len(temp) == 0:
                new_levels.append(temp) # Точка находится в нескольких уровнях

#        print(point, point.vals, type(point), '\n')
#        print('population vals:', [individ.vals for individ in self.population], '\n')
#        print('population objects:', [individ for individ in self.population], '\n')
        population_cleared = []

        for elem in self.population:
            if elem is not point:
                population_cleared.append(elem)
        if len(population_cleared) != sum([len(level) for level in new_levels]):
            print('initial population', [solution.vals for solution in self.population],'\n')
            print('cleared population', [solution.vals for solution in population_cleared],'\n')
            print(point.vals)
            raise Exception('Deleted something extra')

            # new_ind = deepcopy(point)
            # new_ind.vals.structure = []
            # new_ind.vals.fitness = None
            # new_ind.vals.change_all_fixes(False)
            population_cleared.append(point)
        self.levels = new_levels
        self.population = population_cleared
#        self.population.remove(point)
            

def locate_pareto_worst(levels, weights, best_obj, penalty_factor = 1.):
    domain_solutions = population_to_sectors(levels.population, weights)
    most_crowded_count = np.max([len(domain) for domain in domain_solutions]); crowded_domains = [domain_idx for domain_idx in np.arange(len(weights)) if 
                                                                           len(domain_solutions[domain_idx]) == most_crowded_count]
    if len(crowded_domains) == 1:
        most_crowded_domain = crowded_domains[0]
    else:
        PBI = lambda domain_idx: np.sum([penalty_based_intersection(sol_obj, weights[domain_idx], best_obj, penalty_factor) for sol_obj in domain_solutions[domain_idx]])
        PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
        most_crowded_domain = crowded_domains[np.argmax(PBIS)]
        
    worst_NDL_section = []
    domain_solution_NDL_idxs = np.empty(most_crowded_count)
    for solution_idx, solution in enumerate(domain_solutions[most_crowded_domain]):
        domain_solution_NDL_idxs[solution_idx] = [level_idx for level_idx in np.arange(len(levels.levels)) 
                                                    if np.any([solution == level_solution for level_solution in levels.levels[level_idx]])][0]
        
    max_level = np.max(domain_solution_NDL_idxs)
    worst_NDL_section = [domain_solutions[most_crowded_domain][sol_idx] for sol_idx in np.arange(len(domain_solutions[most_crowded_domain])) 
                        if domain_solution_NDL_idxs[sol_idx] == max_level]
    PBIS = np.fromiter(map(lambda solution: penalty_based_intersection(solution, weights[most_crowded_domain], best_obj, penalty_factor), worst_NDL_section), dtype = float)
    return worst_NDL_section[np.argmax(PBIS)]        


class moeadd_optimizer(object):
    '''
    
    Solving multiobjective optimization problem (minimizing set of functions)
    
    '''
    def __init__(self, pop_constructor, weights_num, pop_size, optimized_functionals, solution_params, delta, neighbors_number, 
                 NDS_method = fast_non_dominated_sorting, NDL_update = NDL_update):
        population = []
        for solution_idx in range(pop_size):
            while True:
                temp_solution = pop_constructor.create(solution_params)
                # if not np.any([temp_solution == solution for solution in population]):
                if temp_solution not in population:
                    population.append(temp_solution)
                    break
        self.pareto_levels = pareto_levels(population, sorting_method=NDS_method, update_method=NDL_update)
        
        self.opt_functionals = optimized_functionals
        self.weights = []
        weights_size = len(optimized_functionals) #np.empty((pop_size, len(optimized_functionals)))
        for weights_idx in range(weights_num):
            while True:
                temp_weights = self.weights_generation(weights_size, delta)
                if temp_weights not in self.weights:
                    self.weights.append(temp_weights)
                    break
        self.weights = np.array(self.weights)

        self.neighborhood_lists = []
        for weights_idx in range(weights_num):
            self.neighborhood_lists.append([elem_idx for elem_idx, _ in sorted(
                    list(zip(np.arange(weights_num), [np.linalg.norm(self.weights[weights_idx, :] - self.weights[weights_idx_inner, :]) for weights_idx_inner in np.arange(weights_num)])), 
                    key = lambda pair: pair[1])][:neighbors_number+1]) # срез листа - задаёт регион "близости"

        self.best_obj = None

        
    @staticmethod
    def weights_generation(weights_num, delta) -> list:
        weights = np.empty(weights_num)
        assert 1./delta == round(1./delta) # check, if 1/delta is integer number
        m = np.zeros(weights_num)
        for weight_idx in np.arange(weights_num):
            weights[weight_idx] = np.random.choice([div_idx * delta for div_idx in np.arange(1./delta + 1 - np.sum(m[:weight_idx + 1]))])
            m[weight_idx] = weights[weight_idx]/delta
        weights[-1] = 1 - np.sum(weights[:-1])
        assert (weights[-1] <= 1 and weights[-1] >= 0)
        return list(weights) # Переделать, т.к. костыль
    
        
    def pass_best_objectives(self, *args) -> None:
        assert len(args) == len(self.opt_functionals)
        self.best_obj = np.empty(len(self.opt_functionals))
        for arg_idx, arg in enumerate(args):
            self.best_obj[arg_idx] = arg if isinstance(arg, int) or isinstance(arg, float) else arg() # Переделать под больше elif'ов
    
    
    
    def set_evolutionary(self, operator) -> None:
        # добавить возможность теста оператора
        self.evolutionary_operator = operator
    
    
    @staticmethod
    def mating_selection(weight_idx, weights, neighborhood_vectors, population, neighborhood_selector, neighborhood_selector_params, delta) -> list:
        # parents_number = int(len(population)/4.) # Странное упрощение
        parents_number = 4
        if np.random.uniform() < delta: # особый выбор
            selected_regions_idxs = neighborhood_selector(neighborhood_vectors[weight_idx], *neighborhood_selector_params)
            candidate_solution_domains = list(map(lambda x: x.get_domain(weights), [candidate for candidate in population]))

            solution_mask = [(population[solution_idx].get_domain(weights) in selected_regions_idxs) for solution_idx in candidate_solution_domains]
            available_in_proximity = sum(solution_mask)
            parent_idxs = np.random.choice([idx for idx in np.arange(len(population)) if solution_mask[idx]], 
                                            size = min(available_in_proximity, parents_number), 
                                            replace = False)
            if available_in_proximity < parents_number:
                parent_idxs_additional = np.random.choice([idx for idx in np.arange(len(population)) if not solution_mask[idx]], 
                                            size = parents_number - available_in_proximity, 
                                            replace = False)
                parent_idxs_temp = np.empty(shape = parent_idxs.size + parent_idxs_additional.size)
                parent_idxs_temp[:parent_idxs.size] = parent_idxs; parent_idxs_temp[parent_idxs.size:] = parent_idxs_additional
                parent_idxs = parent_idxs_temp
        else: # либо просто выбирает из всех
            parent_idxs = np.random.choice(np.arange(len(population)), size=parents_number, replace=False)
        return parent_idxs
    
    
    def update_population(self, offspring, PBI_penalty):
        '''
        Update population to get the pareto-nondomiated levels with the worst element removed. 
        Here, "worst" means the individ with highest PBI value (penalty-based boundary intersection)
        '''
#        domain = get_domain_idx(offspring, self.weights)        
        
        self.pareto_levels.update(offspring)  #levels_updated = NDL_update(offspring, levels)
        if len(self.pareto_levels.levels) == 1:
            worst_solution = locate_pareto_worst(self.pareto_levels, self.weights, self.best_obj, PBI_penalty)
        else:
            if self.pareto_levels.levels[len(self.pareto_levels.levels) - 1] == 1:
                domain_solutions = population_to_sectors(self.pareto_levels.population, self.weights)
                reference_solution = self.pareto_levels.levels[len(self.pareto_levels.levels) - 1][0]
                reference_solution_domain = [idx for idx in np.arange(domain_solutions) if reference_solution in domain_solutions[idx]]
                if len(domain_solutions[reference_solution_domain] == 1):
                    worst_solution = locate_pareto_worst(self.pareto_levels.levels, self.weights, self.best_obj, PBI_penalty)                            
                else:
                    worst_solution = reference_solution
            else:
                last_level_by_domains = population_to_sectors(self.pareto_levels.levels[len(self.pareto_levels.levels)-1], self.weights)
                most_crowded_count = np.max([len(domain) for domain in last_level_by_domains]); 
                crowded_domains = [domain_idx for domain_idx in np.arange(len(self.weights)) if len(last_level_by_domains[domain_idx]) == most_crowded_count]

                if len(crowded_domains) == 1:
                    most_crowded_domain = crowded_domains[0]
                else:
                    PBI = lambda domain_idx: np.sum([penalty_based_intersection(sol_obj, self.weights[domain_idx], self.best_obj, PBI_penalty) 
                                                        for sol_obj in last_level_by_domains[domain_idx]])
                    PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
                    most_crowded_domain = crowded_domains[np.argmax(PBIS)]
                    
                if len(last_level_by_domains[most_crowded_domain]) == 1:
                    worst_solution = locate_pareto_worst(self.pareto_levels, self.weights, self.best_obj, PBI_penalty)
                else:
                    PBIS = np.fromiter(map(lambda solution: penalty_based_intersection(solution, self.weights[most_crowded_domain], self.best_obj, PBI_penalty),
                                               last_level_by_domains[most_crowded_domain]), dtype = float)
                    worst_solution = last_level_by_domains[most_crowded_domain][np.argmax(PBIS)]                    
        
        self.pareto_levels.delete_point(worst_solution)
        
        
    def optimize(self, neighborhood_selector, delta, neighborhood_selector_params, epochs, PBI_penalty):
        assert not type(self.best_obj) == type(None)
        for epoch_idx in np.arange(epochs):
            for weight_idx in np.arange(len(self.weights)):
                print('\n\n\n')
                print(epoch_idx, weight_idx)
                print('\n\n\n')
                # time.sleep(2)
                parent_idxs = self.mating_selection(weight_idx, self.weights, self.neighborhood_lists, self.pareto_levels.population,
                                               neighborhood_selector, neighborhood_selector_params, delta)
                offsprings = self.evolutionary_operator.crossover([self.pareto_levels.population[int(idx)] for idx in parent_idxs]) # В объекте эволюционного оператора выделять кроссовер
                # try:
                for offspring_idx, offspring in enumerate(offsprings):
                    while True:
                        temp_offspring = self.evolutionary_operator.mutation(offspring)
                        if not np.any([temp_offspring == solution for solution in self.pareto_levels.population]):
                        # if temp_offspring not in self.pareto_levels.population:
                            break
                    self.update_population(temp_offspring, PBI_penalty)
                # except TypeError:
                #     while True:
                #         temp_offspring = self.evolutionary_operator.mutation(offsprings)
                #         if not np.any([temp_offspring == solution for solution in self.pareto_levels.population]):
                #             break
                #     self.update_population(temp_offspring, PBI_penalty)
            if len(self.pareto_levels.levels) == 1:
                break
                    
                    
                    
class moeadd_optimizer_constrained(moeadd_optimizer):

    def set_constraints(self, *args) -> None:
        self.constraints = args


    def constaint_violation(self, solution) -> float:
        summ = 0
        x = solution.vals
        for constraint in self.constraints:
            summ += constraint(x)
        return summ
        # return np.sum(np.fromiter(map(lambda constr: constr(individ.vals), self.constraints), dtype = float))


    def tournament_selection(self, candidate_1, candidate_2):
        if self.constaint_violation(candidate_1) < self.constaint_violation(candidate_2):
            return candidate_1
        elif self.constaint_violation(candidate_1) > self.constaint_violation(candidate_2):
            return candidate_2
        else:
            return np.random.choice((candidate_1, candidate_2))


    def update_population(self, offspring, PBI_penalty):
        self.pareto_levels.update(offspring)
        cv_values = np.zeros(len(self.pareto_levels.population))
        for sol_idx, solution in enumerate(self.pareto_levels.population):
            cv_val = self.constaint_violation(solution)
            if cv_val > 0:
                cv_values[sol_idx] = cv_val 
        if sum(cv_values) == 0:
            if len(self.pareto_levels.levels) == 1:
                worst_solution = locate_pareto_worst(self.pareto_levels, self.weights, self.best_obj, PBI_penalty)
            else:
                if self.pareto_levels.levels[len(self.pareto_levels.levels) - 1] == 1:
                    domain_solutions = population_to_sectors(self.pareto_levels.population, self.weights)
                    reference_solution = self.pareto_levels.levels[len(self.pareto_levels.levels) - 1][0]
                    reference_solution_domain = [idx for idx in np.arange(domain_solutions) if reference_solution in domain_solutions[idx]]
                    if len(domain_solutions[reference_solution_domain] == 1):
                        worst_solution = locate_pareto_worst(self.pareto_levels.levels, self.weights, self.best_obj, PBI_penalty)                            
                    else:
                        worst_solution = reference_solution
                else:
                    last_level_by_domains = population_to_sectors(self.pareto_levels.levels[len(self.pareto_levels.levels)-1], self.weights)
                    most_crowded_count = np.max([len(domain) for domain in last_level_by_domains]); 
                    crowded_domains = [domain_idx for domain_idx in np.arange(len(self.weights)) if len(last_level_by_domains[domain_idx]) == most_crowded_count]
    
                    if len(crowded_domains) == 1:
                        most_crowded_domain = crowded_domains[0]
                    else:
                        PBI = lambda domain_idx: np.sum([penalty_based_intersection(sol_obj, self.weights[domain_idx], self.best_obj, PBI_penalty) 
                                                            for sol_obj in last_level_by_domains[domain_idx]])
                        PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
                        most_crowded_domain = crowded_domains[np.argmax(PBIS)]
                        
                    if len(last_level_by_domains[most_crowded_domain]) == 1:
                        worst_solution = locate_pareto_worst(self.pareto_levels, self.weights, self.best_obj, PBI_penalty)                            
                    else:
#                        print('the most crowded domain', most_crowded_domain)
                        PBIS = np.fromiter(map(lambda solution: penalty_based_intersection(solution, self.weights[most_crowded_domain], self.best_obj, PBI_penalty), 
                                               last_level_by_domains[most_crowded_domain]), dtype = float)
#                        print('PBIS', PBIS, last_level_by_domains)
                        worst_solution = last_level_by_domains[most_crowded_domain][np.argmax(PBIS)]                    
        else:
            infeasible = [solution for solution, _ in sorted(list(zip(self.pareto_levels.population, cv_values)), key = lambda pair: pair[1])]
            infeasible.reverse()
#            print(np.nonzero(cv_values))
            infeasible = infeasible[:np.nonzero(cv_values)[0].size]
            deleted = False
            domain_solutions = population_to_sectors(self.pareto_levels.population, self.weights)
            
            for infeasable_element in infeasible:
                domain_idx = [domain_idx for domain_idx, domain in enumerate(domain_solutions) if infeasable_element in domain][0]
                if len(domain_solutions[domain_idx]) > 1:
                    deleted = True
                    worst_solution = infeasable_element
                    break
            if not deleted:
                worst_solution = infeasible[0]
        
        self.pareto_levels.delete_point(worst_solution)

            
    def optimize(self, neighborhood_selector, delta, neighborhood_selector_params, epochs, PBI_penalty):
        assert not type(self.best_obj) == type(None)
        self.train_hist = []
        for epoch_idx in np.arange(epochs):
            for weight_idx in np.arange(len(self.weights)):
                print(epoch_idx, weight_idx)
                obj_fun = np.array([solution.obj_fun for solution in self.pareto_levels.population])
                self.train_hist.append(np.mean(obj_fun, axis=0))
                parent_idxs = self.mating_selection(weight_idx, self.weights, self.neighborhood_lists, self.pareto_levels.population,
                                               neighborhood_selector, neighborhood_selector_params, delta)
                if len(parent_idxs) % 2:
                    parent_idxs = parent_idxs[:-1]
                np.random.shuffle(parent_idxs) 
                parents_selected = [self.tournament_selection(self.pareto_levels.population[int(parent_idxs[2*p_metaidx])], 
                                        self.pareto_levels.population[int(parent_idxs[2*p_metaidx+1])]) for 
                                        p_metaidx in np.arange(int(len(parent_idxs)/2.))]
                
                offsprings = self.evolutionary_operator.crossover(parents_selected) # В объекте эволюционного оператора выделять кроссовер
                try:                
                    for offspring_idx, offspring in enumerate(offsprings):
                        while True:
                            temp_offspring = self.evolutionary_operator.mutation(offspring)
                            if not np.any([temp_offspring == solution for solution in self.pareto_levels.population]):
                                break
                        self.update_population(temp_offspring, PBI_penalty)
                except TypeError:
                    while True:
                        temp_offspring = self.evolutionary_operator.mutation(offsprings)
                        if not np.any([temp_offspring == solution for solution in self.pareto_levels.population]):
                            break
                    self.update_population(temp_offspring, PBI_penalty)
            if len(self.pareto_levels.levels) == 1:
                break