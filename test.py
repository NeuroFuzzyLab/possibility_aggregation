from possibilities import PossibiltyDistribution as PD
import numpy as np
from itertools import permutations
import pytest 
from random import sample, randint, random

tests = [
    [[1, 2, 3], [3, 2, 1], [1, 1, 1], [0, 0, 0]],
    [[3, 2, 1], [3, 1, 2], [3, 1, 1], [1, 0, 0]],
    [[1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0]],
    [[1, 2, 1], [2, 1, 1], [1, 1, 1], [0, 0, 0]],
    [[1, 2, 1], [1, 2, 2], [1, 1, 1], [1, 3, 2]],
    [[1, 0, 2], [2, 1, 2], [2, 1, 2], [1, 0, 2]]
]

tests_with_permutations = []
for test in tests:
    all_permutations = list(zip(*[permutations(dist) for dist in test]))
    tests_with_permutations += [test] + all_permutations



@pytest.mark.sup
@pytest.mark.parametrize("pi_1,pi_2,sup", 
    [tuple(test[i] for i in range(3)) for test in tests_with_permutations])
def test_sup(pi_1, pi_2, sup):
    assert PD(pi_1) | PD(pi_2) == PD(sup)
    
    
@pytest.mark.inf
@pytest.mark.parametrize("pi_1,pi_2,inf", 
    [(test[0], test[1], test[3]) for test in tests_with_permutations])
def test_inf(pi_1, pi_2, inf):
    assert PD(pi_1) ^ PD(pi_2) == PD(inf)
