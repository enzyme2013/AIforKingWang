from algorithm import DFS
from algorithm import AStar

dfs = DFS()
astar = AStar()

start = [1,2,3,4,5,6,7,8,0]
target = [1,2,3,8,0,4,7,6,5]

path_dfs = dfs.solve(target,start)
path_astar = astar.solve(target, start)

print(f"dfs:{len(path_dfs)} astar:{len(path_astar)}")