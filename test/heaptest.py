# heap  - LIFO - DFS -- array.pop()
# stack - FIFO - BFS -- array.pop(0)
import numpy as np

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