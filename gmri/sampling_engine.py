import numpy as np
from .orthogonalization_engine import orthogonalizationEngine

__all__ = ['samplingEngine']

class samplingEngine:
    def __init__(self, solve_single, energy_matrix = 1.):
        self.solveSingle = solve_single
        self.orthoEngine = orthogonalizationEngine(energy_matrix)
        self.load(samples_ortho = None, Rfactor = np.empty((0, 0)),
                  nsamples = 0)

    def load(self, **kwargs):
        for name, value in kwargs.items():
            super().__setattr__(name, value)

    def nextSample(self, z):
        u = self.solveSingle(z)
        self.samples = np.append(self.samples, u.reshape(-1, 1), axis = 1)
        self.zs = np.append(self.zs, z)
        u, r, _ = self.orthoEngine.GS(u, self.samples_ortho, self.nsamples)
        self.Rfactor = np.pad(self.Rfactor, ((0, 1), (0, 1)), 'constant')
        self.Rfactor[:, -1] = r
        self.samples_ortho = np.append(self.samples_ortho, u.reshape(-1, 1),
                                       axis = 1)
        self.nsamples += 1
        return self.samples[:, -1]

    def iterSample(self, zs):
        n = len(zs)
        self.zs = np.array(zs)
        u = self.solveSingle(zs[0])
        self.samples = np.empty((len(u), n), dtype = u.dtype)
        self.samples[:, [0]] = u
        for j in range(1, len(zs)):
            self.samples[:, [j]] = self.solveSingle(zs[j])
        self.nsamples = n
        self.samples_ortho, self.Rfactor = self.orthoEngine.generalizedQR(
                                                                  self.samples)
        return self.samples

