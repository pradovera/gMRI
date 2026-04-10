import numpy as np
from scipy.linalg import eigvals
from .sampling_engine import samplingEngine
from .bisection import convexHullWithDistance

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

    @property
    def support(self):
        return self.supp[0]

    @property
    def nsupport(self):
        return self.supp.shape[1]

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
        if not only_den:  # evaluate P
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
        arrow = np.diag(np.append(0., self.support) + 0.j)
        arrow[0, 1 :], arrow[1 :, 0] = self.coeffs, 1.
        active = np.diag(np.append(0., np.ones(self.nsupport)))
        roots = eigvals(arrow, active)
        poles = roots[np.logical_not(np.logical_or(np.isinf(roots),
                                                   np.isnan(roots)))]
        poles = np.sort(poles)
        if return_residues:
            residues = self(poles, mult, ders = [0, 1])
            return poles, residues.T
        return poles


def mergePolesResidues(vals):
    has_also_residues = isinstance(vals[0], tuple)
    if not has_also_residues: return np.concatenate(vals)

    poles_raw, residues_raw = [[v[j] for v in vals] for j in range(2)]
    poles = np.concatenate(poles_raw)
    if np.any([r.shape[1] - residues_raw[0].shape[1] for r in residues_raw]):
        # shape of residues is incompatible: return as list
        residues = sum([list(r) for r in residues_raw], [])
    else:
        residues = np.concatenate(residues_raw, axis = 0)
    return poles, residues


class barycentricRationalFunctionMulti:
    def __init__(self, apps, is_1d = False, **ConvexHullkwargs):
        """
        Initialize barycentricRationalFunctionMulti.

        Args:
            apps: list of barycentricRationalFunction.
        """
        self.napps = len(apps)
        assert self.napps > 0, "input must be non-empty list"
        self.apps = apps
        self.hulls = None
        self.setSupportGrid(is_1d = is_1d, **ConvexHullkwargs)

    def setSupportGrid(self, is_1d = False, **ConvexHullkwargs):
        """
        Set grid of support convex hulls.
        """
        if len(self.apps) == 1: return
        self.hulls = [convexHullWithDistance(app.support, is_1d = is_1d,
                                             **ConvexHullkwargs) for app in self.apps]

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
        # if self.hulls is None: return self.apps[0](x, *args, **kwargs)
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim > 1: x = x[:]
        if len(x) == 0: return np.empty((0, 0))

        if self.hulls is None:
            idx = np.zeros(len(x), dtype = int)
        else:
            dists = np.array([h.distance(x) for h in self.hulls])
            idx = np.argmin(dists, axis = 0)

        val = None
        # find index of support interval
        for j, app in enumerate(self.apps):
            is_j = idx == j
            xj = x[is_j]
            if len(xj) > 0:
                valj = app(xj, *args, **kwargs)
                if val is None:  # initialize val as 2d array
                    val = np.empty((len(valj), len(x)), dtype = valj.dtype)
                elif isinstance(val, np.ndarray) and len(valj) != len(val):
                    val = list(val.T)  # convert val to list of length len(x)
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
        # if self.hulls is None: return self.apps[0].poles(*args, **kwargs)
        return mergePolesResidues([app.poles(*args, **kwargs) for app in self.apps])

    def polesConsolidated(self, *args, **kwargs):
        """
        Compute consolidated poles of barycentric forms. Poles are trusted
            only if nearest to support region of corresponding app.

        Args:
            *args, **kwargs(optional): arguments of
                barycentricRationalFunction().poles;

        Returns:
            Poles (vector) and, if requested, residues too (matrix or list).
        """
        if self.hulls is None: return self.poles(*args, **kwargs)
        vals = [app.poles(*args, **kwargs) for app in self.apps]
        has_also_residues = isinstance(vals[0], tuple)
        for j, val in enumerate(vals):
            poles = val[0] if has_also_residues else val

            dists = np.array([h.distance(poles) for h in self.hulls])
            idx_keep = np.argmin(dists, axis = 0) == j

            if has_also_residues:
                vals[j] = [v[idx_keep] for v in val]
            else:
                vals[j] = val[idx_keep]
        return mergePolesResidues(vals)


class MRI(barycentricRationalFunction):
    def __init__(self, sampler, energy_matrix, supp, eps_stab = None,
                 starting_sampler_data = None):
        """
        Initialize minimal rational interpolant.

        Args:
            sampler, energy_matrix: arguments of samplingEngine;
            supp: support points (vector);
            eps_stab(optional): tolerance for SVD stability check; defaults to
                None, i.e., no stability check;
            starting_sampler_data(optional): dict with keyword arguments for
                loading precomputed samples in samplingEngine.
        """
        self.sampler = samplingEngine(sampler, energy_matrix)
        if starting_sampler_data is not None and len(starting_sampler_data):
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
                 starting_sampler_data = None):
        """
        Initialize greedy MRI.

        Args:
            sampler, energy_matrix: arguments of samplingEngine;
            eps_stab(optional): tolerance for SVD stability check; defaults to
                None, i.e., no stability check;
            starting_sampler_data(optional): dict of keyword arguments for
                loading precomputed samples in samplingEngine.
        """
        self.sampler = samplingEngine(sampler, energy_matrix)
        if starting_sampler_data is not None and len(starting_sampler_data):
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
