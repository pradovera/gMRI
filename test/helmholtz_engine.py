import numpy as np
import scipy.sparse as scsp
import scipy.sparse.linalg as scspla

class solverHelmholtz1DFiniteDifferences:
    def __init__(self, domain_length:float, n_mesh:int,
                 random_seed : int = 42):
        """
        Initialize solverHelmholtzFiniteDifferences for a FD discretization
            of a 1D Helmholtz problem with Dirichlet BCs.
        
        Args:
            domain_length: length of 1D interval serving as Helmholtz domain;
            n_mesh: mesh size;
            random_seed(optional): random seed for generating RHS; defaults to
                42.
        """
        h_minus2 = ((n_mesh + 1) / domain_length) ** 2
        self.stiffness = h_minus2 * scsp.diags([2 * np.ones(n_mesh)]
                                             + [- np.ones(n_mesh)] * 2,
                                               [0, 1, -1], format = "csr")
        self.mass = scsp.eye(n_mesh, format = "csr")
        np.random.seed(random_seed)
        self.rhs = np.random.randn(n_mesh)

    def energyMatrix(self, w):
        """
        Get weighted H1 energy matrix.
        
        Args:
            w: weight;

        Returns:
            Energy matrix.
        """
        return self.stiffness + w * self.mass

    def solve(self, k2):
        """
        Solve Helmholtz at given squared frequency.
        
        Args:
            k2: squared frequency;

        Returns:
            Solution as single-column matrix.
        """
        return scspla.spsolve(self.energyMatrix(- k2), self.rhs).reshape(-1, 1)

