import numpy as np
from copy import deepcopy as copy
from warnings import catch_warnings

__all__ = ['orthogonalizationEngine']

class orthogonalizationEngine:
    def __init__(self, energy_matrix = None):
        self.energy_matrix = energy_matrix

    def norm(self, a):
        if self.energy_matrix is None: return np.linalg.norm(a, axis = 0)
        return np.abs(np.sum((self.energy_matrix @ a) * a.conj(), axis = 0)) ** .5

    def inner(self, a, b):
        if self.energy_matrix is None: return b.conj().T @ a
        return b.conj().T @ (self.energy_matrix @ a)

    def normalize(self, A):
        """
        Normalize column-wise by norm.
        
        Args:
            A: matrix to be normalized;

        Returns:
            Resulting normalized matrix, column-wise norm.
        """
        r = self.norm(A)
        return A / r, r

    def GS(self, a, Q, n = -1):
        """
        Compute 1 Gram-Schmidt step with given projector.
        
        Args:
            a: vector to be projected;
            Q: orthogonal projection matrix;
            n: number of columns of Q to be considered;

        Returns:
            Resulting normalized vector, coefficients of a wrt the updated 
                basis, whether computation is ill-conditioned.
        """
        if n == -1: n = Q.shape[1]
        r = np.zeros(n + 1, dtype = complex)
        if n > 0:
            Q = Q[:, : n]
            for j in range(2): # twice is enough!
                nu = self.inner(a, Q)
                a = a - Q @ nu
                r[:-1] = r[:-1] + nu.flatten()
        r[-1] = self.norm(a)
        ill_cond = False
        with catch_warnings(record = True) as w:
            snr = np.abs(r[-1]) / np.linalg.norm(r)
            if len(w) > 0 or snr < np.finfo(np.complex).eps * len(r):
                ill_cond = True
                r[-1] = 1.
            a = a / r[-1]
        return a, r, ill_cond

    def generalizedQR(self, A, Q0 = None, only_R = False, genTrials = 10):
        """
        Compute generalized QR decomposition of a matrix through Householder
            method; see Trefethen, IMA J.N.A., 2010.
        
        Args:
            A: matrix to be decomposed;
            Q0(optional): initial orthogonal guess for Q; defaults to random;
            only_R(optional): whether to skip reconstruction of Q; defaults to
                False.
            genTrials(optional): number of trials of generation of linearly
                independent vector; defaults to 10.

        Returns:
            Resulting (orthogonal and )upper-triangular factor(s).
        """
        Nh, N = A.shape
        B = copy(A)
        V = np.zeros(A.shape, dtype = complex)
        R = np.zeros((N, N), dtype = complex)
        Q = copy(V) if Q0 is None else copy(Q0)
        for k in range(N):
            a = B[:, k]
            R[k, k] = self.norm(a)
            if Q0 is None and k < Nh:
                for _ in range(genTrials):
                    Q[:, k], _, illC = self.GS(np.random.randn(Nh), Q, k)
                    if not illC: break
            else:
                illC = k >= Nh
            if illC:
                if Q0 is not None or k < Nh: Q[k] = 0.
            else:
                alpha = self.inner(a, Q[:, k])
                if np.isclose(np.abs(alpha), 0., atol = 1e-15): s = 1.
                else: s = - alpha / np.abs(alpha)
                Q[:, k] = s * Q[:, k]
            V[:, k], _, _ = self.GS(R[k, k] * Q[:, k] - a, Q, k)
            J = np.arange(k + 1, N)
            vtB = self.inner(B[:, J], V[:, k])
            B[:, J] = B[:, J] - 2 * np.outer(V[:, k], vtB)
            if not illC:
                R[k, J] = self.inner(B[:, J], Q[:, k])
                B[:, J] = B[:, J] - np.outer(Q[:, k], R[k, J])
        if only_R: return R
        for k in range(min(Nh, N) - 1, -1, -1):
            J = np.arange(k, min(Nh, N))
            vtQ = self.inner(Q[:, J], V[:, k])
            Q[:, J] = Q[:, J] - 2 * np.outer(V[:, k], vtQ)
        return Q, R

