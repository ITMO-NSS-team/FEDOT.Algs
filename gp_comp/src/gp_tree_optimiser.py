from copy import deepcopy
from random import choice, randint
from typing import (
    List,
    Callable,
    Optional,
    Any
)

import numpy as np
import datetime
from gp_comp.src.selection import tournament_selection
from gp_comp.src.crossover import standard_crossover
from gp_comp.src.mutation import standard_mutation
from gp_comp.src.gp_node import GPNode
from gp_comp.src.timer import CompositionTimer
from dataclasses import dataclass


@dataclass
class ComposerRequirements:
    primary: List[Any]
    secondary: List[Any]
    max_lead_time: Optional[datetime.timedelta] = datetime.timedelta(minutes=10)
    max_depth: Optional[int] = None
    max_arity: Optional[int] = None

@dataclass
class GPComposerRequirements(ComposerRequirements):
    pop_size: Optional[int] = 50
    num_of_generations: Optional[int] = 50
    crossover_prob: Optional[float] = None
    mutation_prob: Optional[float] = None


class GPChainOptimiser:
    def __init__(self, initial_chain, requirements, primary_node_func: Callable, secondary_node_func: Callable):
        self.requirements = requirements
        self.primary_node_func = primary_node_func
        self.secondary_node_func = secondary_node_func
        self.best_individual = None
        self.best_fitness = None

        if initial_chain and type(initial_chain) != list:
            self.population = [initial_chain] * requirements.pop_size
        else:
            self.population = initial_chain or self._make_population(self.requirements.pop_size)

    def optimise(self, metric_function_for_nodes):

        with CompositionTimer() as t:

            history = []
            self.fitness = [round(metric_function_for_nodes(tree_root), 3) for tree_root in self.population]
            [history.append((self.population[ind_num], self.fitness[ind_num])) for ind_num in
             range(self.requirements.pop_size)]

            for generation_num in range(self.requirements.num_of_generations - 1):
                print(f'GP generation num: {generation_num}')
                best_ind_num = np.argmin(self.fitness)
                self.best_individual = deepcopy(self.population[best_ind_num])
                self.best_fitness = self.fitness[best_ind_num]
                selected_individuals = tournament_selection(self.fitness, self.population)

                for ind_num in range(self.requirements.pop_size):

                    if ind_num == self.requirements.pop_size - 1:
                        self.population[ind_num] = deepcopy(self.best_individual)
                        self.fitness[ind_num] = self.best_fitness
                        history.append((self.population[ind_num], self.fitness[ind_num]))
                        break

                    self.population[ind_num] = standard_crossover(*selected_individuals[ind_num],
                                                                  crossover_prob=self.requirements.crossover_prob,
                                                                  max_depth=self.requirements.max_depth)

                    self.population[ind_num] = standard_mutation(root_node=self.population[ind_num],
                                                                 secondary=self.requirements.secondary,
                                                                 primary=self.requirements.primary,
                                                                 secondary_node_func=self.secondary_node_func,
                                                                 primary_node_func=self.primary_node_func,
                                                                 mutation_prob=self.requirements.mutation_prob)

                    self.fitness[ind_num] = round(metric_function_for_nodes(self.population[ind_num]), 3)

                    history.append((self.population[ind_num], self.fitness[ind_num]))

                if t.is_max_time_reached(self.requirements.max_lead_time, generation_num):
                    break

        return self.population[np.argmin(self.fitness)], history

    def _make_population(self, pop_size: int) -> List[GPNode]:
        return [self._random_tree() for _ in range(pop_size)]

    def _random_tree(self) -> GPNode:
        root = GPNode(chain_node=self.secondary_node_func(model=choice(self.requirements.secondary)))
        new_tree = root
        self._tree_growth(node_parent=root)
        return new_tree

    def _tree_growth(self, node_parent: Any):
        offspring_size = randint(2, self.requirements.max_arity)
        offspring_nodes = []
        for offspring_node in range(offspring_size):
            if node_parent.height >= self.requirements.max_depth - 1 or (
                    node_parent.height < self.requirements.max_depth - 1
                    and randint(0, 1)):

                new_node = GPNode(
                    chain_node=self.primary_node_func(model=choice(self.requirements.primary), input_data=None),
                    node_to=node_parent)
                offspring_nodes.append(new_node)
            else:
                new_node = GPNode(chain_node=self.secondary_node_func(model=choice(self.requirements.secondary)),
                                  node_to=node_parent)
                self._tree_growth(new_node)
                offspring_nodes.append(new_node)
        node_parent.nodes_from = offspring_nodes
