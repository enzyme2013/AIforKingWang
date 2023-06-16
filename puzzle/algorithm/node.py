import numpy as np


def neighbors(state):
    zero_index = state.index(0)
    adjusted = []
    adjusted = []
    if zero_index - 3 >= 0:
        adjusted.append(zero_index - 3)
    if zero_index + 3 < 9:
        adjusted.append(zero_index + 3)
    if zero_index > 0 and int((zero_index - 1) / 3) == int(zero_index / 3):
        adjusted.append(zero_index - 1)
    if int((zero_index + 1) / 3) == int(zero_index / 3):
        adjusted.append(zero_index + 1)
    result = []
    for swap_pos in adjusted:
        _list = state.copy()
        _list[zero_index] = _list[swap_pos]
        _list[swap_pos] = 0
        result.append(_list)
    return result


class Node:
    parent = None
    state = []
    g = 0

    # h = 0

    # children = []

    def __init__(self, state):
        self.state = state

    def __str__(self):
        return f"{self.state[:3]}\n{self.state[3:6]}\n{self.state[6:]}\n---------"

    def neighbors(self):
        _list = neighbors(self.state)
        result = []
        for st in _list:
            is_parent = False
            if self.parent:
                if tuple(self.parent.state) == tuple(st):
                    is_parent = True
            if not is_parent:
                n = Node(st)
                n.parent = self
                n.g = self.g + 1
                result.append(n)
        return result

    # def kurtosis(self, target):
    #     h = []
    #     for i in range(0, 9):
    #         target_num = target.state[i]
    #         index = self.state.index(target_num)
    #         h.append(abs(int(index / 3) - int(i / 3)) + abs(index % 3 - i % 3))
    #     if len(h) == 0:
    #         print("DEBUG")
    #     r = skew(h, axis=0, bias=True)
    #     return r

    def hamming_distance(self, array):
        return 0

    def equal(self, node):
        return np.array_equal(self.state, node.state)

    def __hash__(self):
        r = hash(tuple(self.state))
        return r

# no = Node([1, 2, 3, 4, 5, 6, 7, 8, 0])
# print(no)
# print(no.__hash__())
# no2 = Node(no.state)
# no2.parent = no
# print(no2.__hash__())
