from abc import abstractmethod
from algorithm import utils
from algorithm.node import Node


class Algorithm:
    # start_state:Node = None
    # open_list = []
    # closed_list = []

    # def __int__(self, init_state=None):

    def solve(self, start_state, target_state):
        pass

    @abstractmethod
    def step(self):
        pass
