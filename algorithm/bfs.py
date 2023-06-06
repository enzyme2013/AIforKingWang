from collections import deque

from algorithm import Algorithm, utils
from algorithm.node import Node


class BFS(Algorithm):

    def __int__(self, init_state=None):
        super(DFS, init_state)

    def solve(self, target_state, start_state=None):
        start_node = None
        if start_state is None:
            start_node = Node(utils.random_state())
        else:
            start_node = Node(start_state)
        target_node = Node(target_state)
        print(f"try to solve from:\n{start_node} to \n{target_node}")
        lastNode = self.dfs(start_node, target_node)
        paths = [lastNode]
        while lastNode.parent:
            paths.insert(0,lastNode.parent)
            lastNode = lastNode.parent
        return paths

    @staticmethod
    def dfs(start_node, target_node):
        frontier = deque()
        frontier.append(start_node)
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

    def step(self):
        pass


