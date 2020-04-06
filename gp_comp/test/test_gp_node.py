from copy import deepcopy
from abc import ABC
from gp_comp.src.gp_node import GPNode
from gp_comp.src.gp_node import swap_nodes
from typing import (Any, List, Optional)


class TestEvaluationStrategy:
    def __init__(self, model: str):
        self.model = model

class TestNode(ABC):
    def __init__(self, eval_strategy: TestEvaluationStrategy, nodes_from: Optional[List['TestNode']] = None):
        self.nodes_from = nodes_from
        self.eval_strategy = eval_strategy
        if self.nodes_from is None:
            self.nodes_from = []


class TestNodeGenerator:
    @staticmethod
    def node(model: str, nodes_from: Optional[List[TestNode]] = None) -> TestNode:
        eval_strategy = TestEvaluationStrategy(model=model)
        return TestNode(nodes_from=nodes_from,
                        eval_strategy=eval_strategy)


def tree_to_chain(tree_root: GPNode) -> List[Any]:
    nodes_list = []
    nodes = flat_nodes_tree(deepcopy(tree_root))
    for node in nodes:
        if node.nodes_from:
            for i in range(len(node.nodes_from)):
                node.nodes_from[i] = node.nodes_from[i].chain_node
        nodes_list.append(node.chain_node)
    return nodes_list


def flat_nodes_tree(node):
    if node.nodes_from:
        nodes = []
        for children in node.nodes_from:
            nodes += flat_nodes_tree(children)
        return [node] + nodes
    else:
        return [node]


def tree_first():
    root_of_tree = GPNode(chain_node=TestNodeGenerator.node(model="XGBoost"))
    root_child_first, root_child_second = [GPNode(chain_node=TestNodeGenerator.node(model=model),
                                                  node_to=root_of_tree) for model in ("XGBoost", "MLP")]
    for last_node_child in (root_child_first, root_child_second):
        for requirement_model in ("KNN", "LDA"):
            new_node = GPNode(chain_node=TestNodeGenerator.node(model=requirement_model),
                              node_to=last_node_child)
            last_node_child.nodes_from.append(new_node)
        root_of_tree.nodes_from.append(last_node_child)
    return root_of_tree


def tree_second():
    root_of_tree = GPNode(chain_node=TestNodeGenerator.node("XGBoost"))
    root_child_first, root_child_second = [
        GPNode(chain_node=TestNodeGenerator.node(model="XGBoost"), node_to=root_of_tree) for _ in
        (range(2))]

    new_node = GPNode(chain_node=TestNodeGenerator.node(model = "LogRegression"), node_to=root_child_first)
    root_child_first.nodes_from.append(new_node)

    new_node = GPNode(TestNodeGenerator.node(model = "XGBoost"), node_to=root_child_first)

    for model_type in ("KNN", "LDA"):
        new_node.nodes_from.append(GPNode(TestNodeGenerator.node(model=model_type), node_to=new_node))

    root_child_first.nodes_from.append(new_node)
    root_of_tree.nodes_from.append(root_child_first)

    for model_type in ("LogRegression", "LDA"):
        root_child_second.nodes_from.append(
            GPNode(TestNodeGenerator.node(model=model_type), node_to=root_child_second))

    root_of_tree.nodes_from.append(root_child_second)
    return root_of_tree


def test_node_depth_and_height():
    last_node = tree_first()

    tree_root_depth = last_node.depth
    tree_secondary_node_depth = last_node.nodes_from[0].depth
    tree_primary_node_depth = last_node.nodes_from[0].nodes_from[0].depth

    assert all([tree_root_depth == 2, tree_secondary_node_depth == 1, tree_primary_node_depth == 0])

    tree_secondary_node_height = last_node.nodes_from[0].height
    tree_primary_node_height = last_node.nodes_from[0].nodes_from[0].height
    tree_root_height = last_node.height

    assert all([tree_secondary_node_height == 1, tree_primary_node_height == 2, tree_root_height == 0])


def test_swap_nodes():
    root_of_tree_first = tree_first()

    root_of_tree_second = tree_second()

    height_in_tree = 1

    # nodes_from_height function check
    nodes_set_tree_first = root_of_tree_first.nodes_from_height(height_in_tree)
    assert len(nodes_set_tree_first) == 2
    nodes_set_tree_second = root_of_tree_second.nodes_from_height(height_in_tree)
    assert len(nodes_set_tree_second) == 2

    tree_first_node = nodes_set_tree_first[1]
    tree_second_node = nodes_set_tree_second[0]

    assert tree_first_node.eval_strategy.model == "MLP"
    assert tree_second_node.eval_strategy.model == "XGBoost"

    swap_nodes(tree_first_node, tree_second_node)

    nodes_list = tree_to_chain(root_of_tree_first)
    assert len(nodes_list) == 9

    correct_nodes = ["XGBoost", "XGBoost", "KNN", "LDA", "XGBoost", "LogRegression", "XGBoost", "KNN", "LDA"]
    # swap_nodes function check
    assert all([model_after_swap.eval_strategy.model == correct_model for model_after_swap, correct_model in
                zip(nodes_list, correct_nodes)])
