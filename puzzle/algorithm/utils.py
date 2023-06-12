import numpy as np
import random


def random_state():
    state = []
    for i in range(0, 9):
        state.append(i)
    random.shuffle(state)
    return state


def random_state_str():
    return "".join(str(t) for t in random_state())


def is_target_node(self, _node, target_state):
    if _node is None:
        return False
    if target_state is None:
        return False
    return np.array_equal(_node.state, target_state)


def str_2_state(_str):
    return list(map(int, list(_str)))
    # return list(map(int,list(_str)))


def get_inv_count(arr):
    inv_count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if (arr[i] > arr[j]) and arr[j] != 0:
                inv_count += 1
    return inv_count


def get_inv_count_str(_str):
    return get_inv_count(str_2_state(_str))


