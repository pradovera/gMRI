import numpy as np
from .mri import MRI, gMRI

__all__ = ['buildMRI', 'buildgMRI']

def buildMRI(sampler, energy_matrix, zs, subdivisions = 1,
             **starting_sampler_data):
    """
    Build MRI with potential subdivisions of parameter range.
    
    Args:
        sampler, energy_matrix: first two arguments of MRI;
        zs: support points (vector);
        subdivisions(optional): number of subdivisions of parameter range;
            defaults to 1, i.e., no subdivisions;
        starting_sampler_data(optional): keyword arguments for loading
            precomputed samples in samplingEngine;

    Returns:
        List of trained MRIs.
    """
    zs = np.array(zs).flatten()
    n = len(zs)
    idx_split = np.round(np.linspace(0, n, subdivisions + 1)).astype(int)
    zs_eff = [zs[idx_split[i] : idx_split[i + 1]] for i in range(subdivisions)]
    mris = [MRI(sampler, energy_matrix, z, **starting_sampler_data)
                                                               for z in zs_eff]
    return mris

def buildgMRI(sampler, energy_matrix, zs, tol, nmax = 1000,
              track_indicator = False, bisections = 0,
              **starting_sampler_data):
    """
    Build gMRI with potential bisections of parameter range.
    
    Args:
        sampler, energy_matrix: first two arguments of gMRI;
        zs: potential support points (vector);
        tol: greedy tolerance;
        nmax(optional): maximum number of greedy iterations; defaults to
            1e3;
        track_indicator(optional): whether to keep track of the greedy
            error indicator's evolution; defaults to False;
        bisections(optional): how many times to bisect parameter range;
            defaults to 0, i.e., no bisections;
        starting_sampler_data(optional): keyword arguments for loading
            precomputed samples in samplingEngine;

    Returns:
        List of trained gMRIs.
    """
    if bisections: track_indicator = False
    mris = [gMRI(sampler, energy_matrix, zs, tol, nmax, track_indicator,
                 **starting_sampler_data)]
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
                                  track_indicator, **sampler_data)]
        mris = new_mris
    return mris
