import random
from copy import deepcopy
from random import randint, choice

import numpy as np

from gp_comp.src.treedrawing import TreeDrawing


def tournament_selection(fitnesses, minimization=True, group_size=5):
    selected = []
    pair_num = 0
    for j in range(len(fitnesses) * 2):
        if not j % 2:
            selected.append([])
            if j > 1:
                pair_num += 1

        tournir = [randint(0, len(fitnesses) - 1) for _ in range(group_size)]
        fitness_obj_from_tour = [fitnesses[tournir[i]] for i in range(group_size)]

        if minimization:
            selected[pair_num].append(tournir[np.argmin(fitness_obj_from_tour)])
        else:
            selected[pair_num].append(tournir[np.argmax(fitness_obj_from_tour)])

    return selected


def standard_crossover(tree1, tree2, max_depth, crossover_prob, pair_num=None, pop_num=None):
    if tree1 is tree2 or random.random() > crossover_prob:
        return deepcopy(tree1)
    tree1_copy = deepcopy(tree1)
    rn_layer = randint(0, tree1_copy.get_depth_down() - 1)
    rn_self_layer = randint(0, tree2.get_depth_down() - 1)
    if rn_layer == 0 and rn_self_layer == 0:
        return deepcopy(tree2)

    changed_node = choice(tree1_copy.get_nodes_from_layer(rn_layer))
    node_for_change = choice(tree2.get_nodes_from_layer(rn_self_layer))

    TreeDrawing().draw_branch(node=tree1,
                              jpeg=f'p1_pair{pair_num}_pop{pop_num}_rnlayer{rn_layer}'
                                   f'({changed_node.eval_strategy.model.__class__.__name__}).png')
    TreeDrawing().draw_branch(node=tree2,
                              jpeg=f'p2_pair{pair_num}_pop{pop_num}_rnselflayer{rn_self_layer}'
                                   f'({node_for_change.eval_strategy.model.__class__.__name__}).png')

    if rn_layer == 0:
        return tree1_copy

    if changed_node.get_depth_up() + node_for_change.get_depth_down() - node_for_change.get_depth_up() < max_depth + 1:
        print(changed_node.get_depth_up())
        print(node_for_change.get_depth_down())
        print(node_for_change.get_depth_up())
        changed_node.swap_nodes(node_for_change)
        TreeDrawing().draw_branch(node=tree1_copy, jpeg=f'result_pair{pair_num}_pop{pop_num}.png')
        return tree1_copy
    else:
        return tree1_copy


def standard_mutation(root_node, secondary_requirements, primary_requirements, probability=None):
    if not probability:
        probability = 1.0 / root_node.get_depth_down()

    TreeDrawing().draw_branch(node=root_node, jpeg=f'tree(mut).png')

    def _node_mutate(node):
        if node.nodes_from:
            if random.random() < probability:
                node.eval_strategy.model = random.choice(secondary_requirements)
            for child in node.nodes_from:
                _node_mutate(child)
        else:
            if random.random() < probability:
                node.eval_strategy.model = random.choice(primary_requirements)

    result = deepcopy(root_node)
    _node_mutate(node=result)
    TreeDrawing().draw_branch(node=result, jpeg=f'tree after mut.png')

    return result
