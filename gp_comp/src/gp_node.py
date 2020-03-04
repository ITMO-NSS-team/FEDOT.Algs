from copy import deepcopy
from typing import (List, Optional)

from gp_comp.example.classes.data import InputData
from gp_comp.example.classes.model import Model
from gp_comp.example.classes.node import Node, NodeGenerator


class GP_Node:
    def __init__(self, chain_node, nodes_to: Optional[List['Node']]):
        self._chain_node = chain_node
        self.nodes_to = nodes_to

    @property
    def nodes_from(self):
        return self._chain_node.nodes_from

    @nodes_from.setter
    def nodes_from(self, nodes):
        self._chain_node.nodes_from = nodes

    @property
    def eval_strategy(self):
        return self._chain_node.eval_strategy

    def get_depth_up(self):
        if self.nodes_to:
            depth = self.nodes_to.get_depth_up() + 1
            return depth
        else:
            return 0

    def get_depth_down(self):
        if not self.nodes_from:
            return 0
        else:
            return 1 + max([next_node.get_depth_down() for next_node in self.nodes_from])

    def get_nodes_from_layer(self, selected_depth):
        if self.nodes_from:
            if self.get_depth_up() == selected_depth:
                return [self]
            else:
                nodes = []
                for child in self.nodes_from:
                    nodes += child.get_nodes_from_layer(selected_depth)
                if nodes:
                    return nodes
                else:
                    return []
        else:
            return []

    def swap_nodes(self, other):
        newnode = deepcopy(other)
        newnode.nodes_to = self.nodes_to
        self.nodes_to.nodes_from[self.nodes_to.nodes_from.index(self)] = newnode
        self = newnode


class GP_NodeGenerator:
    @staticmethod
    def primary_node(model: Model, input_data: InputData, nodes_to: Optional[Node] = None) -> GP_Node:
        chain_node = NodeGenerator.primary_node(model=model, input_data=input_data)
        return GP_Node(chain_node=chain_node, nodes_to=nodes_to)

    @staticmethod
    def secondary_node(model: Model, nodes_to: Optional[Node] = None) -> GP_Node:
        chain_node = NodeGenerator.secondary_node(model=model)
        return GP_Node(chain_node=chain_node, nodes_to=nodes_to)
