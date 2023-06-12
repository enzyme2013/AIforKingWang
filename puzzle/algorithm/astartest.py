from puzzle.algorithm import AStar
from puzzle.algorithm.utils import random_state

bfs = AStar()
paths = bfs.solve([1, 2, 3, 8, 0, 4, 7, 6, 5])
# , Node([1,2,3,8,4,0,7,6,5]))
print('=========================')
for n in paths:
    print(n)
print(f"total steps:{len(paths)}")