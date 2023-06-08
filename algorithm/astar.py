from algorithm import PuzzleSolver


def hamming_distance(node, target):
    h = 0
    for i in range(0, 9):
        if node.state[i] != target.state[i]:
            h += 1
    return h


def manhattan_distance(node, target):
    h = 0
    for i in range(0, 9):
        target_num = target.state[i]
        index = node.state.index(target_num)
        h += 10 * (abs(int(index / 3) - int(i / 3)) + abs(index % 3 - i % 3))
    return h


def euclidean_distance(node, target):
    h = 0
    for i in range(0, 9):
        target_num = target.state[i]
        index = node.state.index(target_num)
        h += (int(index / 3) - int(i / 3) ^ 2) + (index % 3 - i % 3) ^ 2
    return h


def miss_row_col(node, target):
    h = 0
    for i in range(0, 9):
        target_num = target.state[i]
        index = node.state.index(target_num)
        if index / 3 != i / 3:
            h += 1
        if index % 3 != i % 3:
            h += 1
    return h


def gaschnig_heuristics(node, target):
    res = 0
    candidate = list(node.state)
    solved = list(target.state)
    while candidate != solved:
        zi = candidate.index(0)
        if solved[zi] != 0:
            sv = solved[zi]
            ci = candidate.index(sv)
            candidate[ci], candidate[zi] = candidate[zi], candidate[ci]
        else:
            for i in range(9):
                if solved[i] != candidate[i]:
                    candidate[i], candidate[zi] = candidate[zi], candidate[i]
                    break
        res += 1
    return res


def linear_conflicts(node, target, size=3):
    def count_conflicts(row, target_row, size=3, ans=0):
        counts = [0 for x in range(size)]
        for i, tile_1 in enumerate(row):
            if tile_1 in target_row and tile_1 != 0:
                solved_i = target_row.index(tile_1)
                for j, tile_2 in enumerate(row):
                    if tile_2 in target_row and tile_2 != 0 and i != j:
                        solved_j = target_row.index(tile_2)
                        if solved_i > solved_j and i < j:
                            counts[i] += 1
                        if solved_i < solved_j and i > j:
                            counts[i] += 1
        if max(counts) == 0:
            return ans * 2
        else:
            i = counts.index(max(counts))
            row[i] = -1
            ans += 1
            return count_conflicts(row, target_row, size, ans)

    res = manhattan_distance(node, target)
    candidate_rows = [[] for y in range(size)]
    candidate_columns = [[] for x in range(size)]
    solved_rows = [[] for y in range(size)]
    solved_columns = [[] for x in range(size)]
    for y in range(size):
        for x in range(size):
            idx = (y * size) + x
            candidate_rows[y].append(node.state[idx])
            candidate_columns[x].append(node.state[idx])
            solved_rows[y].append(target.state[idx])
            solved_columns[x].append(target.state[idx])
    for i in range(size):
        res += count_conflicts(candidate_rows[i], solved_rows[i], size)
    for i in range(size):
        res += count_conflicts(candidate_columns[i], solved_columns[i], size)
    return res


def pattern_distance(self, target):
    return 0


def manhattan2(self, target):
    h = 0
    for i in range(0, 9):
        target_num = target.state[i]
        index = self.state.index(target_num)
        h += 10 * (abs(index / 3 - i / 3) + abs(index % 3 - i % 3))
    return h


def heuristic(algorithm_type, cur_node, target_node):
    neighbors = cur_node.neighbors()
    if algorithm_type == 1:
        sorted_list = sorted(neighbors, key=lambda neighbor: hamming_distance(neighbor,target_node))
    elif algorithm_type == 2:
        sorted_list = sorted(neighbors, key=lambda neighbor: manhattan_distance(neighbor,target_node))
    elif algorithm_type == 3:
        sorted_list = sorted(neighbors, key=lambda neighbor: euclidean_distance(neighbor,target_node))
    elif algorithm_type == 4:
        sorted_list = sorted(neighbors, key=lambda neighbor: miss_row_col(neighbor,target_node))
    elif algorithm_type == 5:
        sorted_list = sorted(neighbors, key=lambda neighbor: gaschnig_heuristics(neighbor, target_node))
    elif algorithm_type == 6:
        sorted_list = sorted(neighbors, key=lambda neighbor: linear_conflicts(neighbor, target_node))
    elif algorithm_type == 7:
        sorted_list = sorted(neighbors, key=lambda neighbor: pattern_distance(neighbor, target_node))
    else:
        sorted_list = sorted(neighbors, key=lambda neighbor: manhattan_distance(neighbor,target_node))
    sorted_list.reverse()
    return sorted_list


class AStar(PuzzleSolver):
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
