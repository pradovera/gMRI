import numpy as np
from .bisection import bisect
from .mri import barycentricRationalFunctionMulti, MRI, gMRI, StabilityError

__all__ = ['buildMRI', 'buildgMRI']


def buildMRI(sampler, energy_matrix, zs, eps_stab = None, subdivisions = 1,
             bisections = 0, starting_sampler_data = None,
             force_cartesian_bisection = False, is_1d = False):
    """
    Build MRI with potential subdivisions of parameter range.

    Args:
        sampler, energy_matrix: first two arguments of MRI;
        zs: support points (vector);
        eps_stab(optional): tolerance for SVD stability check; defaults to
            None, i.e., no stability check;
        subdivisions(optional): number of subdivisions of parameter range;
            defaults to 1, i.e., no subdivisions;
        bisections(optional): how many times to bisect parameter range;
            defaults to 0, i.e., no bisections;
        starting_sampler_data(optional): dict with keyword arguments for loading
            precomputed samples in samplingEngine;
        force_cartesian_bisection(optional): whether to force bisection along
            cartesian axes; defaults to False;
        is_1d(optional): whether sample points are on a line segment; defaults
            to False.

    Returns:
        List of trained MRIs.
    """
    if subdivisions > 1 and bisections == 0: bisections = 1
    if bisections > 0 and subdivisions <= 1: subdivisions = 2
    if starting_sampler_data is None: starting_sampler_data = {}

    zs_eff = [np.array(zs).flatten()]
    for _ in range(bisections):
        new_zs_eff = []
        for zs_ in zs_eff:
            idxs_bisected = bisect(zs_, nparts = subdivisions,
                                   force_cartesian = force_cartesian_bisection)[0]
            new_zs_eff += [zs_[idx] for idx in idxs_bisected]
        zs_eff = new_zs_eff

    mris = []
    for z in zs_eff: # loop over subdivisions
        mri = MRI(sampler, energy_matrix, z, eps_stab,
                  starting_sampler_data)
        assert mri.sampler.nsamples == len(z), "Loading stored samples not implemented"
        try:
            mri.build(z)
        except StabilityError as e:
            print(f"Error raised on {z=}: {str(e)}")
        mris += [mri]
    return barycentricRationalFunctionMulti(mris, is_1d = is_1d)


def buildgMRI(sampler, energy_matrix, zs, tol, eps_stab = None, nmax = 1000,
              track_indicator = False, bisections = 0, starting_sampler_data = None,
              force_cartesian_bisection = False, is_1d = False):
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
        starting_sampler_data(optional): dict with keyword arguments for loading
            precomputed samples in samplingEngine;
        force_cartesian_bisection(optional): whether to force bisection along
            cartesian axes; defaults to False;
        is_1d(optional): whether sample points are on a line segment.

    Returns:
        List of trained gMRIs.
    """
    if starting_sampler_data is None: starting_sampler_data = {}
    is_bisection_auto = (isinstance(bisections, str)
                      and bisections.upper() == "AUTO")
    layer = 0
    to_split, mris, zss = [], [], []
    mri = gMRI(sampler, energy_matrix, eps_stab, starting_sampler_data)
    try:
        mri.build(zs, tol, nmax, track_indicator)
    except StabilityError as e:
        print(f"Error raised at {layer=}: {str(e)}")
        to_split += [len(mris)]
    mris += [mri]
    zss += [zs]

    while ((not is_bisection_auto and layer < bisections)
         or (is_bisection_auto and len(to_split))):
        layer += 1
        new_to_split, new_mris, new_zss = [], [], []
        # loop over models to bisect (maybe)
        for j, (mri, zs_) in enumerate(zip(mris, zss)):
            if is_bisection_auto and j not in to_split:
                new_mris += [mri]
                new_zss += [zs_]
                continue
            zsamples = mri.support
            idxs_samples_split, idxs_zs_split = bisect(zsamples, zs_,
                                                       force_cartesian = force_cartesian_bisection)
            print("\nsplitting at {}".format(zsamples[idxs_samples_split[1][0]]))
            # train left and right sub-models
            for idxs_samples, idxs_zs in zip(idxs_samples_split, idxs_zs_split[0]):
                zsamples_ = zsamples[idxs_samples]
                samples_ = mri.sampler.samples[:, idxs_samples]
                samples_ortho_, Rfactor_ = (
                               mri.sampler.orthoEngine.generalizedQR(samples_))
                sampler_data = {"nsamples": len(zsamples_), "zs": zsamples_,
                                "samples": samples_, "Rfactor": Rfactor_,
                                "samples_ortho": samples_ortho_}
                new_zs = zs_[idxs_zs]
                new_mri = gMRI(sampler, energy_matrix, eps_stab, sampler_data)
                try:
                    new_mri.build(new_zs, tol, nmax, track_indicator)
                except StabilityError as e:
                    print(f"Error raised at {layer=}, {zsamples_[0]}<z<{zsamples_[-1]}: {str(e)}")
                    new_to_split += [len(new_mris)]
                if track_indicator:
                    new_mri.indicator = mri.indicator + new_mri.indicator
                new_mris += [new_mri]
                new_zss += [new_zs]
        to_split, mris, zss = new_to_split, new_mris, new_zss
    return barycentricRationalFunctionMulti(mris, is_1d = is_1d)
