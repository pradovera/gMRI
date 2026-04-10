import numpy as np
from scipy.spatial import ConvexHull, QhullError

__all__ = ['find_principal_axis', 'bisect', 'convexHullWithDistance']


def find_principal_axis(x):
    y = x - np.mean(x)
    return np.linalg.svd(np.stack([np.real(y), np.imag(y)], axis = 1),
                         full_matrices = False)[2][0].dot([1., 1j])


def bisect(x, *ys, nparts = 2, force_cartesian = False):
    assert nparts > 1
    x_mean = np.mean(x)
    axis = find_principal_axis(x)
    if force_cartesian:
        if np.abs(np.real(axis)) >= np.abs(np.imag(axis)):
            axis = 1.
        else:
            axis = 1j

    idx = np.argsort(np.real((x - x_mean) / axis))
    idx_split = list(map(lambda x: int(round(x)), np.linspace(0., len(x), nparts + 1)))
    x_split = [idx[i : j + 1] for i, j in zip(idx_split[: -1], idx_split[1 :])]

    ys_split = []
    splitting_points = [np.real((x[idx[0]] - x_mean) / axis) for idx in x_split[1 :]]
    for y in ys:
        align = np.real((y - x_mean) / axis)
        mask_split = ([align <= splitting_points[0]]
                    + [np.logical_and(align >= s_i, align <= s_j)
                       for s_i, s_j in zip(splitting_points[: -1], splitting_points[1 :])]
                    + [align >= splitting_points[-1]])
        ys_split.append([np.where(mask)[0] for mask in mask_split])
    return x_split, ys_split


class convexHullWithDistance(ConvexHull):
    def __init__(self, points, is_1d = False, **kwargs):
        points_1d = points[:] if points.ndim > 1 else points
        assert len(points_1d) > 1, "need at least two points"
        points_2d = np.stack([np.real(points_1d), np.imag(points_1d)], axis = 1)
        # Hull plane equations: A x + b = 0
        try:
            super().__init__(points_2d, **kwargs)
            A = self.equations[:, :-1]
            b = self.equations[:, -1]
        except QhullError:  #points are aligned -> QHull is flat
            axis = points_2d[-1] - points_2d[0]
            align = points_2d.dot(axis)
            if is_1d:
                A = np.array([- axis, axis])
                b = np.array([np.min(align), - np.max(align)])
            else:
                axis_orth = np.array([- axis[1], axis[0]])
                align_orth = points_2d[0].dot(axis_orth)
                A = np.array([- axis, axis, - axis_orth, axis_orth])
                b = np.array([np.min(align), - np.max(align), align_orth, - align_orth])
        # Normalize planes
        norms = np.linalg.norm(A, axis = 1)
        self.At = A.T / norms
        self.b = b / norms

    def distance(self, p):
        """
        Distance from point(s) p to the convex hull.

        Parameters
        ----------
        p : array_like, shape (M,)

        Returns
        -------
        distances : float or ndarray, shape (M,)
        """
        p_1d = p[:] if p.ndim > 1 else p
        p_2d = np.stack([np.real(p_1d), np.imag(p_1d)], axis = 1)
        signed = p_2d @ self.At + self.b
        return np.maximum(np.max(signed, axis = 1), 0.0)