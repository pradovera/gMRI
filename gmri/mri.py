import numpy as np
from scipy.linalg import eigvals
from .sampling_engine import samplingEngine

__all__ = ['barycentricRationalFunction', 'MRI', 'gMRI']

def removeClose(zs, z, tol = 1e-12):
    """
    Purge array by removing elements close to prescribed value.
    
    Args:
        zs: array to be purged;
        z: given value (scalar or vector);
        tol(optional): tolerance to define "closeness"; defaults to 1e-12;

    Returns:
        Purged vector.
    """
    return zs[np.all(np.abs(zs - z) > tol, 1)]

class barycentricRationalFunction:
    def __init__(self, sampler, energy_matrix, supp, coeffs, vals):
        """
        Initialize barycentricRationalFunction.
        
        Args:
            sampler, energy_matrix: arguments of samplingEngine;
            supp, coeffs, vals: arguments of self.setBarycentric.
        """
        self.sampler = samplingEngine(sampler, energy_matrix)
        self.setBarycentric(supp, coeffs, vals)
    
    def setBarycentric(self, supp, coeffs, vals):
        """
        Set all parameters of barycentric form.
        
        Args:
            supp: support points (vector);
            coeffs: barycentric denominator coefficients (vector);
            vals: barycentric numerator coefficients, to be multiplied by
                coeffs at evaluation time (vector or matrix).
        """
        self.supp = supp.reshape(1, -1)
        self.coeffs = coeffs.flatten()
        self.vals = vals

    def __call__(self, x, mult = "snaps_ortho", tol = 1e-12,
                 only_den = False, ders = [0] * 2):
        """
        Evaluate barycentric form.
        
        Args:
            x: locations where to evaluate (scalar or vector);
            mult(optional): what to left-multiply the evaluations by; allowed
                values are "snaps_ortho" or None;
            tol(optional): tolerance for stable evaluation; defaults to 1e-12;
            only_den(optional): whether to evaluate only the barycentric
                denominator; defaults to False;
            ders(optional): how many derivatives to compute of numerator and 
                denominator; defaults to [0, 0], i.e., no derivatives;

        Returns:
            Evaluations.
        """
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim != 2 or x.shape[1] != 1: x = np.reshape(x, (-1, 1))
        dx = x - self.supp
        xInfinite = np.abs(dx) < tol
        dx[np.any(xInfinite, 1), :] = np.inf
        dxm1 = dx ** -1
        if not only_den: # evaluate P
            # correct for almost-zero denominators
            if ders[0] <= ders[1]: dxm1[xInfinite] = 1.
            num = (dxm1 ** (1 + ders[0]) @ self.vals).T
            # correct for almost-zero denominators
            if ders[0] < ders[1]: dxm1[xInfinite] = 0.
        # evaluate Q
        if ders[0] >= ders[1]: dxm1[xInfinite] = 1.
        den = dxm1 ** (1 + ders[1]) @ self.coeffs
        if only_den: return den
        val = num / den * (1 - 2 * (sum(ders) % 2))
        if mult == "snaps_ortho": val = self.sampler.samples_ortho.dot(val)
        return val
    
    def poles(self, return_residues = False, mult = "snaps_ortho"):
        """
        Compute poles of barycentric form.
        
        Args:
            return_residues(optional): whether to return residues as well;
                defaults to False;
            mult(optional): what to left-multiply the residues by; allowed
                values are "snaps_ortho" or None;

        Returns:
            Poles (vector) and, if requested, residues too (vector or matrix).
        """
        arrow = np.diag(np.append(0., self.supp[0]) + 0.j)
        arrow[0, 1 :], arrow[1 :, 0] = self.coeffs, 1.
        active = np.diag(np.append(0., np.ones(self.supp.shape[1])))
        roots = eigvals(arrow, active)
        poles = roots[np.logical_not(np.logical_or(np.isinf(roots),
                                                   np.isnan(roots)))]
        poles = np.sort(poles)
        if return_residues:
            residues = self(poles, ders = [0, 1])
            if mult == "snaps_ortho":
                residues = self.sampler.samples_ortho.dot(residues)
            return poles, residues
        return poles

class MRI(barycentricRationalFunction):
    def __init__(self, sampler, energy_matrix, supp,
                 **starting_sampler_data):
        """
        Initialize minimal rational interpolant.
        
        Args:
            sampler, energy_matrix: arguments of samplingEngine;
            supp: support points (vector);
            starting_sampler_data(optional): keyword arguments for loading
                precomputed samples in samplingEngine.
        """
        self.sampler = samplingEngine(sampler, energy_matrix)
        if len(starting_sampler_data) > 0:
            self.sampler.load(**starting_sampler_data)
        else:
            self.sampler.iterSample(supp)
        self.build(self.sampler.zs)
    
    def build(self, supp):
        """
        Build barycentric form using MRI, based on the sampled data.
        
        Args:
            supp: support points (vector).
        """
        coeffs = np.linalg.svd(self.sampler.Rfactor)[2][-1, :].conj()
        self.setBarycentric(supp, coeffs, (self.sampler.Rfactor * coeffs).T)
    
class gMRI(MRI):
    def __init__(self, sampler, energy_matrix, test_points, tol, nmax = 1000,
                 track_indicator = False, **starting_sampler_data):
        """
        Initialize greedy MRI.
        
        Args:
            sampler, energy_matrix: arguments of samplingEngine;
            test_points: potential support points (vector);
            tol: greedy tolerance;
            nmax(optional): maximum number of greedy iterations; defaults to
                1e3;
            track_indicator(optional): whether to keep track of the greedy
                error indicator's evolution; defaults to False;
            starting_sampler_data(optional): keyword arguments for loading
                precomputed samples in samplingEngine.
        """
        self.sampler = samplingEngine(sampler, energy_matrix)
        
        test_points = np.array(test_points).reshape(-1, 1)
        if len(starting_sampler_data) > 0:
            zs0 = np.array(starting_sampler_data["zs"]).reshape(1, -1)
            test_points = removeClose(test_points, zs0)
            self.sampler.load(**starting_sampler_data)
        else:
            zs0 = np.array([[test_points[0, 0], test_points[-1, 0]]])
            test_points = test_points[1:-1, :]
            self.sampler.iterSample(zs0[0])
        self.build(zs0)
        if track_indicator: self.indicator = []
        while zs0.shape[1] <= nmax:
            # identify next sample point
            Qvals = np.abs(self(test_points, only_den = 1))
            idx = np.argmin(Qvals)
            z = test_points[idx, 0]
            # sample at next sample point
            app = self(z)[:, 0]
            ex = self.sampler.nextSample(z)
            # compute relative error
            err = (self.sampler.orthoEngine.norm(app - ex)
                 / self.sampler.orthoEngine.norm(ex))
            if track_indicator:
                self.indicator += [(test_points, err * Qvals[idx] / Qvals)]
            print("{} samples, error at {:.4e} is {:.4e}".format(zs0.shape[1],
                                                                 z, err))
            test_points = np.delete(test_points, idx, axis = 0)
            zs0 = np.append(zs0, [[z]], axis = 1)
            self.build(zs0)
            if err <= tol: break
