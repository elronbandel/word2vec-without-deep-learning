from collections import defaultdict, Counter
from math import sqrt
import numpy as np
class SparseVector(Counter):
    def non_zero_indexes(self):
        return sorted(self.keys())

    def __truediv__(self, other):

        if isinstance(other, SparseVector):
            return SparseVector(dict({(k, v / other[k]) for k,v in self.items()}))

    def __mul__(self, other):
        if not isinstance(other, SparseVector):
            return SparseVector(dict({(k, v * other) for k, v in self.items()}))

    def to_numpy(self, size=0):
        k = np.fromiter(self.keys(), dtype=int)
        size = max(k.max() + 1, size)
        res = np.zeros((size,))
        res[k] = np.fromiter(self.values(), dtype=float)
        return res



class SparseMatrix:
    def __init__(self):
        self._m = defaultdict(SparseVector)
        self._is = defaultdict(set)
        self._js = defaultdict(set)

    def __iter__(self):
        return self._m.__iter__()

    def items(self):
        return self._m.items()

    def __getitem__(self, item):
        try:
            i, j = item
            return self._m[i][j]
        except:
            return self._m[item]

    def __setitem__(self, key, value):
        i, j = key
        self._m[i][j] = value
        if value != 0:
            self._is[j].add(i)
            self._js[i].add(j)
        else:
            self._is[j].remove(i)
            self._js[i].remove(j)

    def non_zero_cols_at(self, i):
        return self._js[i]

    def non_zero_rows_at(self, j):
        return self._is[j]

    def __str__(self):
        return str(self._m)

    def to_numpy(self):
        vecs = [v.to_numpy() for v in self._m.values()]
        res = np.zeros((len(vecs), max(map(len, vecs))))
        for k, vec in zip(self._m.keys(), vecs):
            res[k, :vec.shape[0]] = vec
        return res




#dot product between sparse matrix and sparse vector
def dot(M, U):
    DP = SparseVector()
    for i in U.non_zero_indexes():
        for j in M.non_zero_rows_at(i):
            DP[j] += M[j, i] * U[i]
    return DP



def norm(V):
    if isinstance(V, SparseVector):
        return sqrt(sum(map(lambda x: x*x, V.values())))
    if isinstance(V, SparseMatrix):
        return SparseVector(dict({(k, norm(v)) for k, v in V.items()}))

def cosine(M, U):
    return dot(M, U) / (norm(M) * norm(U))