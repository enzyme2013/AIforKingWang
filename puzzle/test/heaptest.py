# heap  - LIFO - DFS -- array.pop()
# stack - FIFO - BFS -- array.pop(0)
import numpy as np

from puzzle.algorithm.puzzlesolver import is_solvable, is_solvable_str
from puzzle.algorithm.utils import get_inv_count_str

a = [1, 2, 3, 4, 5]
print(a)

print(a.pop())
print(a)

print(a.pop(0))
print(a)

a = [1,2,3,8,0,4,7,6,5]
index = np.array(range(9))
for x , y in enumerate(a):
    index[y] = x
print(index)

_str = "1234567"
a = list(_str)
print(a)
print(list(map(int,list(_str))))

print(get_inv_count_str("123456780"))
print(get_inv_count_str("456870132"))

print(is_solvable_str("806541372","12345680"))
print(is_solvable_str("123456870","12345680"))