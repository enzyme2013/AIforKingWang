from puzzle.algorithm import utils
from puzzle.algorithm.node import Node
from puzzle.algorithm.utils import get_inv_count, get_inv_count_str


def is_solvable(start_state, target_state):
    a = get_inv_count(start_state)
    b = get_inv_count(target_state)
    return (a & 0x01) == (b & 0x01)


def is_solvable_str(start_state, target_state):
    a = get_inv_count_str(start_state)
    b = get_inv_count_str(target_state)
    return (a & 0x01) == (b & 0x01)


class PuzzleSolver:
    def solve_by_str(self, start_str, target_str="123456780"):
        return self.solve(utils.str_2_state(start_str), utils.str_2_state(target_str))

    def solve(self, start_state, target_state):
        if not is_solvable(start_state, target_state):
            return None
        if start_state is None:
            start_node = Node(utils.random_state())
        else:
            start_node = Node(start_state)
        target_node = Node(target_state)
        if tuple(start_state) == tuple(target_state):
            return None
        lastNode = self.algorithm_func(start_node, target_node)
        paths = [lastNode]
        while lastNode.parent:
            paths.append(lastNode.parent)
            lastNode = lastNode.parent
        paths.reverse()
        return paths

    def algorithm_func(self, start_node, target_node):
        return target_node


# print(is_solvable_str("834267501","123456780"))