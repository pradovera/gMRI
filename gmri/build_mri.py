import numpy as np
from .mri import barycentricRationalFunctionMulti, MRI, gMRI, StabilityError

__all__ = ['buildMRI', 'buildgMRI']

def buildMRI(sampler, energy_matrix, zs, eps_stab = None, subdivisions = 1,
             **starting_sampler_data):
    """
    Build MRI with potential subdivisions of parameter range.
    
    Args:
        sampler, energy_matrix: first two arguments of MRI;
        zs: support points (vector);
        eps_stab(optional): tolerance for SVD stability check; defaults to
            None, i.e., no stability check;
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
    mris = []
    for z in zs_eff: # loop over subdivisions
        mri = MRI(sampler, energy_matrix, z, eps_stab,
                  **starting_sampler_data)
        assert mri.sampler.nsamples == len(z), "Loading stored samples not implemented"
        try:
            mri.build(z)
        except StabilityError as e:
            print(f"Error raised on {z=}: {str(e)}")
        mris += [mri]
    return barycentricRationalFunctionMulti(mris)

def buildgMRI(sampler, energy_matrix, zs, tol, eps_stab = None, nmax = 1000,
              track_indicator = False, bisections = 0,
              **starting_sampler_data):
    """
    Build gMRI with potential bisections of parameter range.
    
    Args:
        sampler, energy_matrix: first two arguments of gMRI;
        zs: potential support points (vector);
        tol: greedy tolerance;
        eps_stab(optional): tolerance for SVD stability check; defaults to
            None, i.e., no stability check;
        nmax(optional): maximum number of greedy iterations; defaults to
            1e3;
        track_indicator(optional): whether to keep track of the greedy
            error indicator's evolution; defaults to False;
        bisections(optional): how many times to bisect parameter range;
            defaults to 0, i.e., no bisections; if "AUTO", bisect
            automatically whenever MRI building raises an error;
        starting_sampler_data(optional): keyword arguments for loading
            precomputed samples in samplingEngine;

    Returns:
        List of trained gMRIs.
    """
    is_bisection_auto = (isinstance(bisections, str)
                      and bisections.upper() == "AUTO")
    if bisections: track_indicator = False
    layer = 0
    to_split, mris = [], []
    mri = gMRI(sampler, energy_matrix, eps_stab, **starting_sampler_data)
    try:
        mri.build(zs, tol, nmax, track_indicator)
    except StabilityError as e:
        print(f"Error raised at {layer=}: {str(e)}")
        to_split += [len(mris)]
    mris += [mri]
    
    while ((not is_bisection_auto and layer < bisections)
        or (is_bisection_auto and len(to_split))):
        layer += 1
        new_to_split, new_mris = [], []
        # loop over models to bisect (maybe)
        for j, mri in enumerate(mris):
            if is_bisection_auto and j not in to_split:
                new_mris += [mri]
                continue
            zsamples = mri.supp[0]
            idxs_sort = np.argsort(zsamples)
            idxs_split = [idxs_sort[: len(idxs_sort) // 2 + 1],
                          idxs_sort[len(idxs_sort) // 2 :]] # split at half point
            print("\nsplitting at {}".format(zsamples[idxs_split[1][0]]))
            # train left and right sub-models
            for idxs in idxs_split:
                zsamples_ = zsamples[idxs]
                samples_ = mri.sampler.samples[:, idxs]
                samples_ortho_, Rfactor_ = (
                               mri.sampler.orthoEngine.generalizedQR(samples_))
                sampler_data = {"nsamples": len(zsamples_), "zs": zsamples_,
                                "samples": samples_, "Rfactor": Rfactor_,
                                "samples_ortho": samples_ortho_}
                zs_ = zs[np.logical_and(zs > zsamples_[0], zs < zsamples_[-1])]
                new_mri = gMRI(sampler, energy_matrix, eps_stab,
                               **sampler_data)
                try:
                    new_mri.build(zs_, tol, nmax)
                except StabilityError as e:
                    print(f"Error raised at {layer=}, {zsamples_[0]}<z<{zsamples_[-1]}: {str(e)}")
                    new_to_split += [len(new_mris)]
                new_mris += [new_mri]
        to_split, mris = new_to_split, new_mris
    return barycentricRationalFunctionMulti(mris)
