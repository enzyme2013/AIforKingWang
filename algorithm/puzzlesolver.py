from algorithm import utils
from algorithm.node import Node


class PuzzleSolver:
    def solve(self, target_state, start_state=None):
        if start_state is None:
            start_node = Node(utils.random_state())
        else:
            start_node = Node(start_state)
        target_node = Node(target_state)
        lastNode = self.algorithm_func(start_node, target_node)
        paths = [lastNode]
        while lastNode.parent:
            paths.append(lastNode.parent)
            lastNode = lastNode.parent
        paths.reverse()
        return paths

    def algorithm_func(self, start_node, target_node):
        return target_node
