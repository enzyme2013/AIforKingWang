from collections import deque

from algorithm import Algorithm, utils
from algorithm.node import Node


def heuristic(cur_node, target_node):
    neighbors = cur_node.neighbors()
    sorted_list = sorted(neighbors, key=lambda neighbor: neighbor.manhattan(target_node))
    sorted_list.reverse()
    return sorted_list


# def heuristic2(cur_node, target_node):
#     neighbors = cur_node.neighbors()
#     sorted_list = sorted(neighbors, key=lambda neighbor: neighbor.manhattan2(target_node))
#     sorted_list.reverse()
#     return sorted_list



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
            sorted_list = heuristic(cur_node,target_node)
            for node in sorted_list:
                if tuple(node.state) not in visited:
                    frontier.append(node)
                    visited.add(tuple(node.state))
        return None
