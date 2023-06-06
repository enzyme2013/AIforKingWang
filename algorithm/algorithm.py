from abc import abstractmethod
from algorithm import utils
from algorithm.node import Node


class Algorithm:
    def solve(self,  target_state, start_state=None):
        start_node = None
        if start_state is None:
            start_node = Node(utils.random_state())
        else:
            start_node = Node(start_state)
        target_node = Node(target_state)
        print(f"try to solve from:\n{start_node} to \n{target_node}")
        lastNode = self.alogrithm_func(start_node, target_node)
        paths = [lastNode]
        while lastNode.parent:
            paths.append(lastNode.parent)
            lastNode = lastNode.parent
        paths.reverse()
        return paths

    @staticmethod
    def alogrithm_func(start_node, target_node):
        return target_node
