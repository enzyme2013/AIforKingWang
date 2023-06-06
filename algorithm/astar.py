from collections import deque

from algorithm import Algorithm, utils
from algorithm.node import Node


class AStar(Algorithm):
    open_list: list = None
    closed_list: list = None

    def __int__(self):
        super()

    @staticmethod
    def alogrithm_func(start_node, target_node):
        frontier = [start_node]
        visited = set()
        while frontier:
            cur_node = frontier.pop()
            visited.add(tuple(cur_node.state))
            if cur_node.equal(target_node):
                return cur_node
            neighbors = cur_node.neighbors()
            for node in neighbors:
                if tuple(node.state) not in visited:
                    frontier.append(node)
                    visited.add(tuple(node.state))

        return None