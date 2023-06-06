from algorithm import Algorithm, utils
from node import Node


class DFS(Algorithm):

    def __int__(self, init_state=None):
        super(DFS, init_state)

    def solve(self, target_node, start_node=None):
        if start_node is None:
            start_node = Node(utils.random_state())
        print(f"try to solve from:\n{start_node} to \n{target_node}")
        lastNode = self.dfs(start_node, target_node)
        paths = [lastNode]
        while lastNode.parent:
            paths.insert(0,lastNode.parent)
            lastNode = lastNode.parent
        return paths

    @staticmethod
    def dfs(start_node, target_node):
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

    def step(self):
        pass


dfs = DFS()
paths = dfs.solve(Node([1, 2, 3, 8, 0, 4, 7, 6, 5]))
# , Node([1,2,3,8,4,0,7,6,5]))
print('=========================')
for n in paths:
    print(n)
print(f"total steps:{len(paths)}")
