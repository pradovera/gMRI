import numpy as np
from scipy.linalg import eigvals
from .sampling_engine import samplingEngine

__all__ = ['barycentricRationalFunction', 'barycentricRationalFunctionMulti',
           'MRI', 'gMRI']

EPS = 1e-12
class StabilityError(Exception): pass

def removeClose(zs, z, tol = EPS):
    """
    Purge array by removing elements close to prescribed value.
    
    Args:
        zs: array to be purged;
        z: given value (scalar or vector);
        tol(optional): tolerance to define "closeness"; defaults to EPS;

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

    def __call__(self, x, mult = "snaps_ortho", tol = EPS,
                 only_den = False, ders = [0] * 2):
        """
        Evaluate barycentric form.
        
        Args:
            x: locations where to evaluate (scalar or vector);
            mult(optional): what to left-multiply the evaluations by; allowed
                values are "snaps_ortho" or None;
            tol(optional): tolerance for stable evaluation; defaults to EPS;
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
            return poles, residues.T
        return poles

class barycentricRationalFunctionMulti:
    def __init__(self, apps):
        """
        Initialize barycentricRationalFunctionMulti.
        
        Args:
            apps: list of barycentricRationalFunction.
        """
        assert len(apps) > 0, "input must be non-empty list"
        self.apps = apps
        self.setSupportGrid()
    
    def setSupportGrid(self):
        """
        Set grid of support intervals.
        """
        if len(self.apps) == 1:
            self.grid = np.empty(0)
            return
        supps = [app.supp for app in self.apps]
        mins = np.array([np.min(supp) for supp in supps])
        maxs = np.array([np.max(supp) for supp in supps])
        idx_sort, idx_sort_max = np.argsort(mins), np.argsort(maxs)
        mins, maxs = mins[idx_sort], maxs[idx_sort_max]
        if np.any(idx_sort - idx_sort_max) or np.any(maxs[: -1] > mins[1 :]):
            raise Exception("cannot sort support intervals due to overlaps")
        self.apps = [self.apps[j] for j in idx_sort]
        self.grid = .5 * (maxs[: -1] + mins[1 :])

    def __call__(self, x, *args, **kwargs):
        """
        Evaluate barycentric form.
        
        Args:
            x: locations where to evaluate (scalar or vector);
            *args, **kwargs(optional): arguments of
                barycentricRationalFunction().__call__;

        Returns:
            Evaluations (as 2d array with samples as columns if possible, else
                         as list).
        """
        if len(self.grid) == 0: return self.apps[0](x, *args, **kwargs)
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim != 2 or x.shape[1] != 1: x = np.reshape(x, (-1, 1))
        if len(x) == 0: return np.empty((0, 0))
        val = None

        # find index of support interval
        idx = sum([x > gp for gp in self.grid], 0).flatten()
        for j, app in enumerate(self.apps):
            is_j = idx == j
            xj = x[is_j]
            if len(xj) > 0:
                valj = app(xj, *args, **kwargs)
                if val is None: # initialize val as 2d array
                    val = np.empty((len(valj), len(x)), dtype = valj.dtype)
                elif isinstance(val, np.ndarray) and len(valj) != len(val):
                    val = list(val.T) # convert val to list of length len(x)
                if isinstance(val, np.ndarray):
                    val[:, is_j] = valj
                else:
                    for i, k in enumerate(np.where(is_j)[0]):
                        val[k] = valj[:, i]
        return val
    
    def poles(self, *args, **kwargs):
        """
        Compute poles of barycentric forms.
        
        Args:
            *args, **kwargs(optional): arguments of
                barycentricRationalFunction().poles;

        Returns:
            Poles (vector) and, if requested, residues too (matrix or list).
        """
        if len(self.grid) == 0: return self.apps[0].poles(*args, **kwargs)
        vals = [app.poles(*args, **kwargs) for app in self.apps]
        if isinstance(vals[0], tuple):
            poles_raw, residues_raw = [[v[j] for v in vals] for j in range(2)]
            poles = np.concatenate(poles_raw)
            if np.any([r.shape[1] - residues_raw[0].shape[1]
                                                       for r in residues_raw]):
                # shape of residues is incompatible: return as list
                residues = sum([list(r) for r in residues_raw], [])
            else:
                residues = np.concatenate(residues_raw, axis = 0)
            return poles, residues
        return np.concatenate(vals)

class MRI(barycentricRationalFunction):
    def __init__(self, sampler, energy_matrix, supp, eps_stab = None,
                 **starting_sampler_data):
        """
        Initialize minimal rational interpolant.
        
        Args:
            sampler, energy_matrix: arguments of samplingEngine;
            supp: support points (vector);
            eps_stab(optional): tolerance for SVD stability check; defaults to
                None, i.e., no stability check;
            starting_sampler_data(optional): keyword arguments for loading
                precomputed samples in samplingEngine.
        """
        self.sampler = samplingEngine(sampler, energy_matrix)
        if len(starting_sampler_data) > 0:
            self.sampler.load(**starting_sampler_data)
        else:
            self.sampler.iterSample(supp)
        self.eps_stab = eps_stab
    
    def build(self, supp):
        """
        Build barycentric form using MRI, based on the sampled data.
        
        Args:
            supp: support points (vector).
        """
        _, sigma, Vh = np.linalg.svd(self.sampler.Rfactor)
        coeffs = Vh[-1, :].conj()
        self.setBarycentric(supp, coeffs, (self.sampler.Rfactor * coeffs).T)
        if (self.eps_stab is not None
        and sigma[-2] - sigma[-1] < self.eps_stab * sigma[0]):
            raise StabilityError("MRI stability check fail")
    
class gMRI(MRI):
    def __init__(self, sampler, energy_matrix, eps_stab = None,
                 **starting_sampler_data):
        """
        Initialize greedy MRI.
        
        Args:
            sampler, energy_matrix: arguments of samplingEngine;
            eps_stab(optional): tolerance for SVD stability check; defaults to
                None, i.e., no stability check;
            starting_sampler_data(optional): keyword arguments for loading
                precomputed samples in samplingEngine.
        """
        self.sampler = samplingEngine(sampler, energy_matrix)
        if len(starting_sampler_data) > 0:
            self.sampler.load(**starting_sampler_data)
        self.eps_stab = eps_stab

    def build(self, test_points, tol, nmax = 1000, track_indicator = False):
        """
        Build barycentric form using greedy MRI.
        
        Args:
            test_points: potential support points (vector);
            tol: greedy tolerance;
            nmax(optional): maximum number of greedy iterations; defaults to
                1e3;
            track_indicator(optional): whether to keep track of the greedy
                error indicator's evolution; defaults to False;
        """
        if track_indicator: self.indicator = []
        test_points = np.array(test_points).reshape(-1, 1)
        if self.sampler.nsamples > 0:
            zs0 = self.sampler.zs.reshape(1, -1)
            test_points = removeClose(test_points, zs0)
        else:
            zs0 = np.array([[test_points[0, 0], test_points[-1, 0]]])
            test_points = test_points[1:-1, :]
            self.sampler.iterSample(zs0[0])
        super().build(zs0)

        while len(test_points) and zs0.shape[1] <= nmax:
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
            super().build(zs0)
            if err <= tol: break
        else:
            if zs0.shape[1] > nmax:
                raise StabilityError("gMRI ran out of iterations")
