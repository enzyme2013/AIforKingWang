import numpy as np
import random

def random_state():
    state = []
    for i in range(0, 9):
        state.append(i)
    random.shuffle(state)
    return state


def is_target_node(self, _node, target_state):
    if _node is None:
        return False
    if target_state is None:
        return False
    return np.array_equal(_node.state, target_state)
