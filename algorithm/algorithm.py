import random
from abc import abstractmethod

TARGET_STATE = [1, 2, 3, 8, 0, 4, 6, 5, 7]

def gen_state():
    state = []
    for i in range(0, 9):
        state.append(i)
    random.shuffle(state)
    return state


def print_state(state):
    print(f"{state[0]} {state[1]} {state[2]}\n{state[3]} {state[4]} {state[5]}\n{state[6]} {state[7]} {state[8]}\n")


def find_children(state):
    zero_index = state.index(0)
    adjusted = []
    if zero_index - 3 >= 0:
        adjusted.append(zero_index - 3)
    if zero_index + 3 < 9:
        adjusted.append(zero_index + 3)
    if int((zero_index - 1) / 3) == int(zero_index / 3):
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
    children = []

    def __int__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def is_target(self):
        return self.sate == TARGET_STATE


class Algorithm:
    start_state = []

    open_list = []
    closed_list = []

    def __int__(self):
        self.start_state = gen_state()

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def step(self):
        pass


_state = gen_state()
print_state(_state)
children = find_children(_state)
for ch in children:
    print_state(ch)
node = Node(TARGET_STATE)
print(node.is_target())
