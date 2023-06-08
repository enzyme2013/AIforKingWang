from algorithm import PuzzleSolver


class DFS(PuzzleSolver):

    def algorithm_func(self, start_node, target_node):
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


