from algorithm import AStar
from algorithm.utils import random_state


def test_alg(alg, start_state, end_state):
    paths = alg.solve(start_state, end_state)
    return len(paths)


def test_algs(algs):
    start_state = random_state()
    end_state = random_state()
    r = []
    for alg in algs:
        r.append(test_alg(alg, start_state, end_state))
    return r


def gen_AStar(alg_type):
    astar = AStar()
    astar.set_type(alg_type)
    return astar


algs = [gen_AStar(1), gen_AStar(2), gen_AStar(3), gen_AStar(4)]
for i in range(10):
    r = test_algs(algs)
    print(r)

# [41798, 1742, 140222, 46]
# [84763, 1953, 113211, 1433]
# [137693, 5009, 108509, 3235]
# [35453, 5225, 76879, 2463]
# [137486, 182, 108876, 626]
# [41843, 877, 118161, 6213]
# [82279, 17429, 101407, 5633]
# [132987, 4659, 133311, 831]
# [47801, 1573, 90083, 2993]
# [86672, 78, 57532, 7604]
