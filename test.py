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
    
    
    








# def barplot(pi_1, pi_2, sup=None, inf=None):
    # x = np.arange(len(pi_1))
    # pi_1 = pi_1.normalize()  
    # pi_2 = pi_2.normalize()
    # sup = sup.normalize()
    # inf = inf.normalize()
    
    # fig = plt.figure(figsize=(7, 7))
    # gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1],
                           # hspace=0.5)
    
    # ax0 = fig.add_subplot(gs[0, :])
    # ax0.bar(x, sup, color='r')
    # ax0.set_title('supremum')
    # ax0.set_aspect("2")


    # ax1 = fig.add_subplot(gs[1, 0])
    # ax1.bar(x, pi_1, color='b')
    # ax1.set_title('distribution 1')

    # ax2 = fig.add_subplot(gs[1, 1])
    # ax2.bar(x, pi_2, color='g')
    # ax2.set_title('distribution 2')

    # ax3 = fig.add_subplot(gs[2, :])
    # ax3.bar(x, inf, color='r')
    # ax3.set_title('infimum')
    # ax3.set_aspect("2")
    
    # plt.show()
    
    
    
# pi_1 = PossibiltyDistribution([1, 2, 3])
# pi_2 = PossibiltyDistribution([3, 2, 1])

# inf = pi_1 ^ pi_2
# sup = pi_1 | pi_2

# barplot(pi_1, pi_2, sup, inf)