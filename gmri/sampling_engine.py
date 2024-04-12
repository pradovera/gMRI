import numpy as np
from .orthogonalization_engine import orthogonalizationEngine

__all__ = ['samplingEngine']

class samplingEngine:
    def __init__(self, solve_single, energy_matrix = 1.):
        """
        Initialize samplingEngine.
        
        Args:
            solve_single: callable for getting new samples;
            energy_matrix(optional): energy matrix for inner products; defaults
                to identity.
        """
        self.solveSingle = solve_single
        self.orthoEngine = orthogonalizationEngine(energy_matrix)
        self.load(samples_ortho = None, Rfactor = np.empty((0, 0)),
                  nsamples = 0)

    def load(self, **kwargs):
        """
        Load samplingEngine.
        
        Args:
            **kwargs: keyword arguments to be loaded.
        """
        for name, value in kwargs.items():
            super().__setattr__(name, value)

    def nextSample(self, z):
        """
        Compute new sample and add it to list.
        
        Args:
            z: location of new sample;

        Returns:
            New sample as vector.
        """
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
        """
        Compute a collection of samples and use them to initialize the
            sample list.
        
        Args:
            zs: location of samples (array or list);

        Returns:
            Resulting snapshot matrix, with samples as columns.
        """
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

