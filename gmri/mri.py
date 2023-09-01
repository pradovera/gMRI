import numpy as np
from scipy.linalg import eigvals
from .sampling_engine import samplingEngine

__all__ = ['baseMRI', 'MRI', 'gMRI', 'buildMRI', 'buildgMRI']

def removeClose(zs, zs0, tol = 1e-12):
    return zs[np.all(np.abs(zs - zs0) > tol, 1)]

class baseMRI:
    def __init__(self, sampler, energy_matrix, supp, coeffs, vals):
        self.sampler = samplingEngine(sampler, energy_matrix)
        self.setBarycentric(supp, coeffs, vals)
    
    def setBarycentric(self, supp, coeffs, vals):
        self.supp = supp.reshape(1, -1) # support points
        self.coeffs = coeffs.flatten() # barycentric (denominator) coefficients
        self.vals = vals # support (numerator) values

    def __call__(self, x, mult = "snaps_ortho", tol = 1e-12,
                 only_den = False, ders = [0] * 2):
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

class MRI(baseMRI):
    def __init__(self, sampler, energy_matrix, zs,
                 starting_sampler_data = None):
        self.sampler = samplingEngine(sampler, energy_matrix)
        if starting_sampler_data is None:
            self.sampler.iterSample(zs)
        else:
            self.sampler.load(**starting_sampler_data)
        self.build(self.sampler.zs)
    
    def build(self, supp):
        coeffs = np.linalg.svd(self.sampler.Rfactor)[2][-1, :].conj()
        self.setBarycentric(supp, coeffs, (self.sampler.Rfactor * coeffs).T)
    
class gMRI(MRI):
    def __init__(self, sampler, energy_matrix, zs, tol, nmax = 1000,
                 track_indicator = False, starting_sampler_data = None):
        self.sampler = samplingEngine(sampler, energy_matrix)
        
        zs = np.array(zs).reshape(-1, 1)
        if starting_sampler_data is None:
            zs0 = np.array([[zs[0, 0], zs[-1, 0]]])
            zs = zs[1:-1, :]
            self.sampler.iterSample(zs0[0])
        else:
            zs0 = np.array(starting_sampler_data["zs"]).reshape(1, -1)
            zs = removeClose(zs, zs0)
            self.sampler.load(**starting_sampler_data)
        self.build(zs0)
        if track_indicator: self.indicator = []
        while zs0.shape[1] <= nmax:
            # identify next sample point
            Qvals = np.abs(self(zs, only_den = 1))
            idx = np.argmin(Qvals)
            z = zs[idx, 0]
            # sample at next sample point
            app = self(z)[:, 0]
            ex = self.sampler.nextSample(z)
            # compute relative error
            err = (self.sampler.orthoEngine.norm(app - ex)
                 / self.sampler.orthoEngine.norm(ex))
            if track_indicator:
                self.indicator += [(zs, err * Qvals[idx] / Qvals)]
            print("{} samples, error at {:.4e} is {:.4e}".format(zs0.shape[1],
                                                                 z, err))
            zs = np.delete(zs, idx, axis = 0)
            zs0 = np.append(zs0, [[z]], axis = 1)
            self.build(zs0)
            if err <= tol: break

def buildMRI(sampler, energy_matrix, zs, starting_sampler_data = None,
             subdivisions = 1):
    zs = np.array(zs).flatten()
    n = len(zs)
    idx_split = int(np.round(np.arange(0, n, subdivisions + 1)))
    zs_eff = [zs[idx_split[i] : idx_split[i + 1]] for i in range(subdivisions)]
    mris = [MRI(sampler, energy_matrix, z, starting_sampler_data)
                                                               for z in zs_eff]
    return mris

def buildgMRI(sampler, energy_matrix, zs, tol, nmax = 1000,
              track_indicator = False, starting_sampler_data = None,
              bisections = 0):
    if bisections: track_indicator = False
    mris = [gMRI(sampler, energy_matrix, zs, tol, nmax, track_indicator,
                 starting_sampler_data)]
    for layer in range(bisections):
        new_mris = []
        # halve each of previous
        for mri in mris:
            zsamples = mri.supp[0]
            idxs_sort = np.argsort(zsamples)
            idxs_split = [idxs_sort[: len(idxs_sort) // 2 + 1],
                          idxs_sort[len(idxs_sort) // 2 :]]
            print("\nsplitting at {}".format(zsamples[idxs_split[1][0]]))
            for idxs in idxs_split:
                zsamples_ = zsamples[idxs]
                samples_ = mri.sampler.samples[:, idxs]
                samples_ortho_, Rfactor_ = (
                               mri.sampler.orthoEngine.generalizedQR(samples_))
                sampler_data = {"nsamples": len(zsamples_), "zs": zsamples_,
                                "samples": samples_, "Rfactor": Rfactor_,
                                "samples_ortho": samples_ortho_}
                zs_ = zs[np.logical_and(zs > zsamples_[0], zs < zsamples_[-1])]
                new_mris += [gMRI(sampler, energy_matrix, zs_, tol, nmax,
                                  track_indicator, sampler_data)]
        mris = new_mris
    return mris
