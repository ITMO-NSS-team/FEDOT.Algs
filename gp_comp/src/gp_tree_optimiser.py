import numpy as np
from copy import deepcopy
from core.composer.chain import Chain
from core.composer.gp_composer.gp_node import GP_Node
from core.composer.tree_drawing import Tree_Drawing
from core.models.data import Data
from core.models.model import Model
from random import choice, randint
from typing import (
    List,
    Callable,
    Optional
)


class GPChainOptimiser:
    def __init__(self, initial_chain, requirements, primary_node_func: Callable, secondary_node_func: Callable):
        self.requirements = requirements
        self.__primary_node_func = primary_node_func
        self.__secondary_node_func = secondary_node_func
        if initial_chain and type(initial_chain) != list:
            self.population = [initial_chain] * requirements.pop_size
        else:
            self.population = initial_chain or self._make_population(self.requirements.pop_size)

        Tree_Drawing().draw_branch(node=self.population[1], jpeg="tree.png")

    def optimise(self, metric_function_for_nodes) -> GP_Node:
        for generation_num in range(self.requirements.num_of_generations):
            print("GP generation num:\n", generation_num)
            self.fitness = [round(metric_function_for_nodes(tree_root), 3) for tree_root in self.population]
            if not self.requirements.minimization:
                self.the_best_ind = self.population[np.argsort(self.fitness)[len(self.fitness) - 1]]
            else:
                self.the_best_ind = self.population[np.argsort(self.fitness)[0]]

            selected_indexes = GPChainOptimiser._tournament_selection(fitnesses=self.fitness,
                                                                      minimization=self.requirements.minimization,
                                                                      group_size=5)
            new_population = []
            for ind_num in range(self.requirements.pop_size - 1):
                new_population.append(
                    GPChainOptimiser._standard_crossover(tree1=self.population[selected_indexes[ind_num][0]],
                                                         tree2=self.population[selected_indexes[ind_num][1]],
                                                         max_depth=self.requirements.max_depth, pair_num=ind_num,
                                                         pop_num=generation_num))

        return self.population[0]

    def _make_population(self, pop_size) -> List[GP_Node]:
        return [self._random_tree() for _ in range(pop_size)]

    def _random_tree(self) -> GP_Node:
        root = self.__secondary_node_func(choice(self.requirements.secondary_requirements))
        self._tree_growth(node_parent=root)
        return root

    def _tree_growth(self, node_parent):
        offspring_size = randint(2, self.requirements.max_arity)
        node_offspring = []
        for offspring_node in range(offspring_size):
            if node_parent.get_depth_up() >= self.requirements.max_depth or (
                    node_parent.get_depth_up() < self.requirements.max_depth and
                    self.requirements.max_depth and randint(
                0, 1)):

                new_node = self.__primary_node_func(choice(self.requirements.primary_requirements),
                                                    nodes_to=node_parent, input_data=None)
                node_offspring.append(new_node)
            else:
                new_node = self.__secondary_node_func(choice(self.requirements.secondary_requirements),
                                                      nodes_to=node_parent)
                self._tree_growth(new_node)
                node_offspring.append(new_node)
        node_parent.nodes_from = node_offspring

    @staticmethod
    def _tournament_selection(fitnesses, minimization=False, group_size=5):
        selected = []
        pair_num = 0
        for j in range(len(fitnesses) * 2):
            if not j % 2:
                selected.append([])
                if j > 1:
                    pair_num += 1

            tournir = [randint(0, len(fitnesses) - 1) for _ in range(group_size)]
            fitnessobjfromtour = [fitnesses[tournir[i]] for i in range(group_size)]

            if minimization:
                selected[pair_num].append(tournir[np.argmin(fitnessobjfromtour)])
            else:
                selected[pair_num].append(tournir[np.argmax(fitnessobjfromtour)])

        return selected

    @staticmethod
    def _standard_crossover(tree1, tree2, max_depth, pair_num=None, pop_num=None):
        if tree1 is tree2:
            return deepcopy(tree1)
        tree1_copy = deepcopy(tree1)
        rnlayer = randint(0, tree1_copy.get_depth_down() - 1)
        rnselflayer = randint(0, tree2.get_depth_down() - 1)
        if rnlayer == 0 and rnselflayer == 0:
            return deepcopy(tree2)

        changednode = choice(tree1_copy.get_nodes_from_layer(rnlayer))
        nodeforchange = choice(tree2.get_nodes_from_layer(rnselflayer))

        Tree_Drawing().draw_branch(node=tree1,
                                   jpeg=f'p1_pair{pair_num}_pop{pop_num}_rnlayer{rnlayer}({changednode.eval_strategy.model.__class__.__name__}).png')
        Tree_Drawing().draw_branch(node=tree2,
                                   jpeg=f'p2_pair{pair_num}_pop{pop_num}_rnselflayer{rnselflayer}({nodeforchange.eval_strategy.model.__class__.__name__}).png')

        if rnlayer == 0:
            return tree1_copy

        if changednode.get_depth_up() + nodeforchange.get_depth_down() - nodeforchange.get_depth_up() < max_depth + 1:
            print(changednode.get_depth_up())
            print(nodeforchange.get_depth_down())
            print(nodeforchange.get_depth_up())
            changednode.swap_nodes(nodeforchange)
            Tree_Drawing().draw_branch(node=tree1_copy, jpeg=f'result_pair{pair_num}_pop{pop_num}.png')
            return tree1_copy
        else:
            return tree1_copy
