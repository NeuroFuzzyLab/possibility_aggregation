import numpy as np
from functools import cmp_to_key

def find_points_satisfying_lemma_3(pi_1, pi_2):
    p_1, p_2 = pi_1.possibilities, pi_2.possibilities
    cond_1 = (p_1[:, None] <= p_1) & (p_2[:, None] > p_2)
    cond_2 = (p_1[:, None] < p_1) & (p_2[:, None] >= p_2)
    index = np.transpose(np.array(np.where(cond_1 | cond_2)))
    if not np.any(index):
        return None
    return index[0]


def apply_lemma_1(pi_1, pi_2):
    mask = pi_1.support.mask
    alpha = min(pi_1[mask]) / max(pi_2[~mask]) * 0.9
    pi_1[~mask] = alpha * pi_2[~mask]
    return pi_1, pi_2


def apply_lemma_2(pi_1, pi_2):
    supp_1, supp_2 = pi_1.support, pi_2.support
    alpha1 = max(pi_1[(supp_1 - supp_2).mask])
    alpha2 = max(pi_2[(supp_2 - supp_1).mask])
    union_mask = (supp_1 | supp_2).mask
    pi_1[union_mask] = max(max(pi_1[union_mask]), alpha1)
    pi_1[~union_mask] = 0
    pi_2[union_mask] = max(max(pi_2[union_mask]), alpha2)
    pi_2[~union_mask] = 0
    return pi_1, pi_2


def apply_lemma_3(pi_1, pi_2, index):
    i, j = index
    alpha_1x = min(pi_1[i], pi_1[j])
    alpha_2x = max(pi_1[i], pi_1[j])
    alpha_1y = min(pi_2[i], pi_2[j])
    alpha_2y = max(pi_2[i], pi_2[j])
    pi_1[(pi_1 >= alpha_1x) & (pi_1 <= alpha_2x)] = alpha_2x
    pi_2[(pi_2 >= alpha_1y) & (pi_2 <= alpha_2y)] = alpha_2y
    return pi_1, pi_2


def compute_supremum(pi_1, pi_2):
    pi_1, pi_2 = pi_1.copy(), pi_2.copy()
    # print("-"*30)
    # print(pi_1, pi_2)
    if pi_1.support != pi_2.support:
        # print("here")
        if pi_1.support < pi_2.support:
            pi_1, pi_2 = apply_lemma_1(pi_1, pi_2)
        elif pi_1.support > pi_2.support:
            pi_2, pi_1 = apply_lemma_1(pi_2, pi_1)
        else:
            pi_1, pi_2 = apply_lemma_2(pi_1, pi_2)
    
    # print(pi_1, pi_2)
    
    points_satisfying_lemma_3 = find_points_satisfying_lemma_3(pi_1, pi_2) 
    while points_satisfying_lemma_3 is not None:
        pi_1, pi_2 = apply_lemma_3(pi_1, pi_2, points_satisfying_lemma_3)
        points_satisfying_lemma_3 = find_points_satisfying_lemma_3(pi_1, pi_2)
    
    # print("*", pi_1[0] == 1.0, pi_1[1] == 0., pi_1[2] == 1.0)
    return pi_1, pi_2
    
 
def cmp(a, b):
    return 1 if (a[1] >= b[1]) and (a[2] >= b[2]) else -1


def find_points_satisfying_lemma_4(pi_1, pi_2):
    p_1, p_2 = pi_1.possibilities, pi_2.possibilities
    cond = (p_1[:, None] < p_1) & (p_2[:, None] > p_2)
    index = np.transpose(np.array(np.where(cond)))
    if not np.any(index):
        return None
    return index[0]


def apply_lemma_4(pi_1, pi_2, index):
    i, j = index
    alpha1 = max(pi_1[i], pi_1[j])
    alpha2 = max(pi_2[i], pi_2[j])
    m = np.apply_along_axis(np.any, 0, np.eye(len(pi_1), dtype=bool)[index]) # ??????
    pi_1[m | (pi_1 < alpha1)] = 0
    pi_2[m | (pi_2 < alpha2)] = 0
    return pi_1, pi_2


def apply_lemma_5(pi_1, pi_2):
    pi_2[~pi_1.support.mask] = 0
    return pi_1, pi_2


def apply_lemma_6(pi_1, pi_2):
    p_1, p_2 = pi_1.possibilities, pi_2.possibilities
    to_sort = [(i, x, y) for i, (x, y) in enumerate(zip(p_1, p_2))]
    indexes, sorted_p_1, sorted_p_2 = np.array(sorted(to_sort, key=cmp_to_key(cmp))).T
    indexes = indexes.astype(int)
    wop1 = (sorted_p_1 == np.unique(sorted_p_1)[::-1, np.newaxis])
    wop2 = (sorted_p_2 == np.unique(sorted_p_2)[::-1, np.newaxis])
    index = np.unique(np.transpose(np.delete(np.array(np.where(wop1[:, None] & wop2)), 2, 0)), axis = 0)
    possibilities = np.zeros(len(pi_1), dtype = float)
    for i in range(len(index)):
        possibilities[indexes[(wop1[:, None] & wop2)[index[i][0]][index[i][1]]]] = (len(index) - i)/len(index)
    possibilities[np.where((p_1 == 0) | (p_2 == 0))] = 0
    return PossibiltyDistribution(possibilities)


def compute_infimum(pi_1, pi_2):
    pi_1, pi_2 = pi_1.copy(), pi_2.copy()

    points__satisfying_lemma_4 = find_points_satisfying_lemma_4(pi_1, pi_2)
    while points__satisfying_lemma_4 is not None:
        pi_1, pi_2 = apply_lemma_4(pi_1, pi_2, points__satisfying_lemma_4)
        points__satisfying_lemma_4 = find_points_satisfying_lemma_4(pi_1, pi_2)
    
    if pi_1.support != pi_2.support:
        if pi_1.support < pi_2.support:
            pi_1, pi_2 = apply_lemma_5(pi_1, pi_2)
        elif pi_1.support > pi_2.support:
            pi_2, pi_1 = apply_lemma_5(pi_2, pi_1)
    
    pi = apply_lemma_6(pi_1, pi_2)
    return pi

def set_subtraction(mask_1, mask_2):
    """
    returns mask of  
    set_1 - set_2, 
    where set_i is represented by mask_i.
    """
    return np.logical_not(mask_1 & mask_2) & mask_1

def is_subset(mask_1, mask_2):
    """
    returns True if set_1 is subset of set_2 and False otherwise
    set_i is represented by mask_i
    """
    return np.array_equal(mask_1 | mask_2, mask_2)


class Support:
    def __init__(self, mask):
        self.mask = mask
    
    def __lt__(self, other):
        return is_subset(self.mask, other.mask)
    
    def __gt__(self, other):
        return other < self
    
    def __sub__(self, other):
        # print(self, other, set_subtraction(self.mask, other.maks))
        return Support(set_subtraction(self.mask, other.mask))
    
    def __or__(self, other):
        return Support(self.mask | other.mask)
    
    def __and__(self, other):
        return Support(self.mask & other.mask)
    
    def __eq__(self, other):
        return np.array_equal(self.mask, other.mask)
    
    def __ne__(self, other):
        return not (self == other)

    def __invert__(self):
        return Support(~self.mask)
    
    def __str__(self):
        return self.mask.__str__()
    
    def __repr__(self):
        return self.mask.__repr__()


class PossibiltyDistribution:
    def __init__(self, possibilities, mask=None):
        possibilities = np.array(possibilities)
        if mask is None:
            self.support = Support(possibilities > 0)
        else:
            mask = np.array(mask)
            self.support = Support(mask)
            possibilities *= mask
        if possibilities.max() != 0:
            possibilities = possibilities / possibilities.max()
        self.possibilities = possibilities
        
    def normalize(self):
        # print(self.possibilities)
        values = np.unique(self.possibilities)
        if 0 not in values:
            values = np.concatenate([np.array([0]), values])
        target_values = np.linspace(start=0, stop=1, num=values.size)
        values_to_target_values = {value:target_value for (value, target_value) in zip(values, target_values)}
        new_possibilities = np.vectorize(values_to_target_values.get)(self.possibilities)
        return PossibiltyDistribution(new_possibilities)
    
    
    def __eq__(self, other):
        if self.support != other.support:
            return False
        
        if find_points_satisfying_lemma_3(self, other) is not None:
            return False
        
        return np.array_equal(self.normalize().possibilities, other.normalize().possibilities)
    
    def __ne__(self, other):
        return not self == other
    
    def __getitem__(self, key):
        return self.possibilities.__getitem__(key)
    
    def __setitem__(self, key, value):
        self.possibilities.__setitem__(key, value)
        self.support = Support(self.possibilities > 0)
        
    def __len__(self):
        return self.possibilities.__len__()
        
    def __str__(self):
        normalized = self.normalize()
        return normalized.possibilities.__str__()
    
    def __repr__(self):
        return self.possibilities.__str__()
    
    def __lt__(self, other):
        if isinstance(other, PossibiltyDistribution):
            return NotImplemented
        else:
            return self.possibilities < other
    
    def __le__(self, other):
        if isinstance(other, PossibiltyDistribution):
            return self < other
        else:
            return self.possibilities <= other
        
    def __gt__(self, other):
        if isinstance(other, PossibiltyDistribution):
            return other < self
        else:
            return self.possibilities > other
    
    def __ge__(self, other):
        if isinstance(other, PossibiltyDistribution):
            return other <= self
        else:
            return self.possibilities >= other
    
    def __xor__(self, other):
        return compute_infimum(self, other)
        
    def __or__(self, other):
        return compute_supremum(self, other)[0]
        
    def copy(self):
        return PossibiltyDistribution(self.possibilities)