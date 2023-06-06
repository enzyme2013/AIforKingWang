from abc import abstractmethod
import utils
from node import Node


class Algorithm:
    start_state:Node = None
    open_list = []
    closed_list = []

    def __int__(self, init_state=None):
        if init_state:
            self.start_state = Node(init_state)
        else:
            self.start_state = Node(utils.random_state())

    def solve(self, target_state, start_node=None):
        pass

    @abstractmethod
    def step(self):
        pass



