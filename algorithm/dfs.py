from algorithm.algorithm import Algorithm


class DFS(Algorithm):
    visited = set()

    def solve(self):
        pass

    def dfs(self, visited, node):
        if node not in visited:
            if is_target(node):
                return node
            visited.add(node)
            for ch in node.children:
                self.dfs(visited, ch)

    def step(self):
        pass
