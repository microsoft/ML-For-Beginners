"""
Various transforms used for by the 3D code
"""

import numpy as np
import numpy.linalg as linalg


def _line2d_seg_dist(p, s0, s1):
    """
    Return the distance(s) from point(s) *p* to segment(s) (*s0*, *s1*).

    Parameters
    ----------
    p : (ndim,) or (N, ndim) array-like
        The points from which the distances are computed.
    s0, s1 : (ndim,) or (N, ndim) array-like
        The xy(z...) coordinates of the segment endpoints.
    """
    s0 = np.asarray(s0)
    s01 = s1 - s0  # shape (ndim,) or (N, ndim)
    s0p = p - s0  # shape (ndim,) or (N, ndim)
    l2 = s01 @ s01  # squared segment length
    # Avoid div. by zero for degenerate segments (for them, s01 = (0, 0, ...)
    # so the value of l2 doesn't matter; this just replaces 0/0 by 0/1).
    l2 = np.where(l2, l2, 1)
    # Project onto segment, without going past segment ends.
    p1 = s0 + np.multiply.outer(np.clip(s0p @ s01 / l2, 0, 1), s01)
    return ((p - p1) ** 2).sum(axis=-1) ** (1/2)


def world_transformation(xmin, xmax,
                         ymin, ymax,
                         zmin, zmax, pb_aspect=None):
    """
    Produce a matrix that scales homogeneous coords in the specified ranges
    to [0, 1], or [0, pb_aspect[i]] if the plotbox aspect ratio is specified.
    """
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    if pb_aspect is not None:
        ax, ay, az = pb_aspect
        dx /= ax
        dy /= ay
        dz /= az

    return np.array([[1/dx, 0,    0,    -xmin/dx],
                     [0,    1/dy, 0,    -ymin/dy],
                     [0,    0,    1/dz, -zmin/dz],
                     [0,    0,    0,    1]])


def rotation_about_vector(v, angle):
    """
    Produce a rotation matrix for an angle in radians about a vector.
    """
    vx, vy, vz = v / np.linalg.norm(v)
    s = np.sin(angle)
    c = np.cos(angle)
    t = 2*np.sin(angle/2)**2  # more numerically stable than t = 1-c

    R = np.array([
        [t*vx*vx + c,    t*vx*vy - vz*s, t*vx*vz + vy*s],
        [t*vy*vx + vz*s, t*vy*vy + c,    t*vy*vz - vx*s],
        [t*vz*vx - vy*s, t*vz*vy + vx*s, t*vz*vz + c]])

    return R


def _view_axes(E, R, V, roll):
    """
    Get the unit viewing axes in data coordinates.

    Parameters
    ----------
    E : 3-element numpy array
        The coordinates of the eye/camera.
    R : 3-element numpy array
        The coordinates of the center of the view box.
    V : 3-element numpy array
        Unit vector in the direction of the vertical axis.
    roll : float
        The roll angle in radians.

    Returns
    -------
    u : 3-element numpy array
        Unit vector pointing towards the right of the screen.
    v : 3-element numpy array
        Unit vector pointing towards the top of the screen.
    w : 3-element numpy array
        Unit vector pointing out of the screen.
    """
    w = (E - R)
    w = w/np.linalg.norm(w)
    u = np.cross(V, w)
    u = u/np.linalg.norm(u)
    v = np.cross(w, u)  # Will be a unit vector

    # Save some computation for the default roll=0
    if roll != 0:
        # A positive rotation of the camera is a negative rotation of the world
        Rroll = rotation_about_vector(w, -roll)
        u = np.dot(Rroll, u)
        v = np.dot(Rroll, v)
    return u, v, w


def _view_transformation_uvw(u, v, w, E):
    """
    Return the view transformation matrix.

    Parameters
    ----------
    u : 3-element numpy array
        Unit vector pointing towards the right of the screen.
    v : 3-element numpy array
        Unit vector pointing towards the top of the screen.
    w : 3-element numpy array
        Unit vector pointing out of the screen.
    E : 3-element numpy array
        The coordinates of the eye/camera.
    """
    Mr = np.eye(4)
    Mt = np.eye(4)
    Mr[:3, :3] = [u, v, w]
    Mt[:3, -1] = -E
    M = np.dot(Mr, Mt)
    return M


def view_transformation(E, R, V, roll):
    """
    Return the view transformation matrix.

    Parameters
    ----------
    E : 3-element numpy array
        The coordinates of the eye/camera.
    R : 3-element numpy array
        The coordinates of the center of the view box.
    V : 3-element numpy array
        Unit vector in the direction of the vertical axis.
    roll : float
        The roll angle in radians.
    """
    u, v, w = _view_axes(E, R, V, roll)
    M = _view_transformation_uvw(u, v, w, E)
    return M


def persp_transformation(zfront, zback, focal_length):
    e = focal_length
    a = 1  # aspect ratio
    b = (zfront+zback)/(zfront-zback)
    c = -2*(zfront*zback)/(zfront-zback)
    proj_matrix = np.array([[e,   0,  0, 0],
                            [0, e/a,  0, 0],
                            [0,   0,  b, c],
                            [0,   0, -1, 0]])
    return proj_matrix


def ortho_transformation(zfront, zback):
    # note: w component in the resulting vector will be (zback-zfront), not 1
    a = -(zfront + zback)
    b = -(zfront - zback)
    proj_matrix = np.array([[2, 0,  0, 0],
                            [0, 2,  0, 0],
                            [0, 0, -2, 0],
                            [0, 0,  a, b]])
    return proj_matrix


def _proj_transform_vec(vec, M):
    vecw = np.dot(M, vec)
    w = vecw[3]
    # clip here..
    txs, tys, tzs = vecw[0]/w, vecw[1]/w, vecw[2]/w
    return txs, tys, tzs


def _proj_transform_vec_clip(vec, M):
    vecw = np.dot(M, vec)
    w = vecw[3]
    # clip here.
    txs, tys, tzs = vecw[0] / w, vecw[1] / w, vecw[2] / w
    tis = (0 <= vecw[0]) & (vecw[0] <= 1) & (0 <= vecw[1]) & (vecw[1] <= 1)
    if np.any(tis):
        tis = vecw[1] < 1
    return txs, tys, tzs, tis


def inv_transform(xs, ys, zs, M):
    """
    Transform the points by the inverse of the projection matrix *M*.
    """
    iM = linalg.inv(M)
    vec = _vec_pad_ones(xs, ys, zs)
    vecr = np.dot(iM, vec)
    try:
        vecr = vecr / vecr[3]
    except OverflowError:
        pass
    return vecr[0], vecr[1], vecr[2]


def _vec_pad_ones(xs, ys, zs):
    return np.array([xs, ys, zs, np.ones_like(xs)])


def proj_transform(xs, ys, zs, M):
    """
    Transform the points by the projection matrix *M*.
    """
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec(vec, M)


transform = proj_transform


def proj_transform_clip(xs, ys, zs, M):
    """
    Transform the points by the projection matrix
    and return the clipping result
    returns txs, tys, tzs, tis
    """
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec_clip(vec, M)


def proj_points(points, M):
    return np.column_stack(proj_trans_points(points, M))


def proj_trans_points(points, M):
    xs, ys, zs = zip(*points)
    return proj_transform(xs, ys, zs, M)


def rot_x(V, alpha):
    cosa, sina = np.cos(alpha), np.sin(alpha)
    M1 = np.array([[1, 0, 0, 0],
                   [0, cosa, -sina, 0],
                   [0, sina, cosa, 0],
                   [0, 0, 0, 1]])
    return np.dot(M1, V)
