from algorithm import Algorithm


def heuristic(algorithm_type, cur_node, target_node):
    neighbors = cur_node.neighbors()
    if algorithm_type == 1:
        sorted_list = sorted(neighbors, key=lambda neighbor: neighbor.hamming_distance(target_node))
    elif algorithm_type == 2:
        sorted_list = sorted(neighbors, key=lambda neighbor: neighbor.manhattan_distance(target_node))
    elif algorithm_type == 3:
        sorted_list = sorted(neighbors, key=lambda neighbor: neighbor.euclidean_distance(target_node))
    elif algorithm_type == 4:
        sorted_list = sorted(neighbors, key=lambda neighbor: neighbor.miss_row_col(target_node))
    else:
        sorted_list = sorted(neighbors, key=lambda neighbor: neighbor.manhattan(target_node))
    sorted_list.reverse()
    return sorted_list


class AStar(Algorithm):
    algorithm_type = 1

    def __int__(self):
        super()

    def set_type(self,alg_type):
        self.algorithm_type = alg_type

    def algorithm_func(self, start_node, target_node):
        frontier = [start_node]
        visited = set()
        while frontier:
            cur_node = frontier.pop()
            visited.add(tuple(cur_node.state))
            if cur_node.equal(target_node):
                return cur_node
            sorted_list = heuristic(self.algorithm_type, cur_node, target_node)
            for node in sorted_list:
                if tuple(node.state) not in visited:
                    frontier.append(node)
                    visited.add(tuple(node.state))
        return None
