# -*- coding: utf-8 -*-
"""
This file contains a Python version of the gradient projection rotation
algorithms (GPA) developed by Bernaards, C.A. and Jennrich, R.I.
The code is based on code developed Bernaards, C.A. and Jennrich, R.I.
and is ported and made available with permission of the authors.

References
----------
[1] Bernaards, C.A. and Jennrich, R.I. (2005) Gradient Projection Algorithms
and Software for Arbitrary Rotation Criteria in Factor Analysis. Educational
and Psychological Measurement, 65 (5), 676-696.

[2] Jennrich, R.I. (2001). A simple general procedure for orthogonal rotation.
Psychometrika, 66, 289-306.

[3] Jennrich, R.I. (2002). A simple general method for oblique rotation.
Psychometrika, 67, 7-19.

[4] http://www.stat.ucla.edu/research/gpa/matlab.net

[5] http://www.stat.ucla.edu/research/gpa/GPderfree.txt
"""

import numpy as np


def GPA(A, ff=None, vgQ=None, T=None, max_tries=501,
        rotation_method='orthogonal', tol=1e-5):
    r"""
    The gradient projection algorithm (GPA) minimizes a target function
    :math:`\phi(L)`, where :math:`L` is a matrix with rotated factors.

    For orthogonal rotation methods :math:`L=AT`, where :math:`T` is an
    orthogonal matrix. For oblique rotation matrices :math:`L=A(T^*)^{-1}`,
    where :math:`T` is a normal matrix, i.e., :math:`TT^*=T^*T`. Oblique
    rotations relax the orthogonality constraint in order to gain simplicity
    in the interpretation.

    Parameters
    ----------
    A : numpy matrix
        non rotated factors
    T : numpy matrix (default identity matrix)
        initial guess of rotation matrix
    ff : function (defualt None)
        criterion :math:`\phi` to optimize. Should have A, T, L as keyword
        arguments
        and mapping to a float. Only used (and required) if vgQ is not
        provided.
    vgQ : function (defualt None)
        criterion :math:`\phi` to optimize and its derivative. Should have
         A, T, L as keyword arguments and mapping to a tuple containing a
        float and vector. Can be omitted if ff is provided.
    max_tries : int (default 501)
        maximum number of iterations
    rotation_method : str
        should be one of {orthogonal, oblique}
    tol : float
        stop criterion, algorithm stops if Frobenius norm of gradient is
        smaller then tol
    """
    # pre processing
    if rotation_method not in ['orthogonal', 'oblique']:
        raise ValueError('rotation_method should be one of '
                         '{orthogonal, oblique}')
    if vgQ is None:
        if ff is None:
            raise ValueError('ff should be provided if vgQ is not')
        derivative_free = True
        Gff = lambda x: Gf(x, lambda y: ff(T=y, A=A, L=None))
    else:
        derivative_free = False
    if T is None:
        T = np.eye(A.shape[1])
    # pre processing for iteration
    al = 1
    table = []
    # pre processing for iteration: initialize f and G
    if derivative_free:
        f = ff(T=T, A=A, L=None)
        G = Gff(T)
    elif rotation_method == 'orthogonal':  # and not derivative_free
        L = A.dot(T)
        f, Gq = vgQ(L=L)
        G = (A.T).dot(Gq)
    else:  # i.e. rotation_method == 'oblique' and not derivative_free
        Ti = np.linalg.inv(T)
        L = A.dot(Ti.T)
        f, Gq = vgQ(L=L)
        G = -((L.T).dot(Gq).dot(Ti)).T
    # iteration
    for i_try in range(0, max_tries):
        # determine Gp
        if rotation_method == 'orthogonal':
            M = (T.T).dot(G)
            S = (M + M.T)/2
            Gp = G - T.dot(S)
        else:  # i.e. if rotation_method == 'oblique':
            Gp = G-T.dot(np.diag(np.sum(T*G, axis=0)))
        s = np.linalg.norm(Gp, 'fro')
        table.append([i_try, f, np.log10(s), al])
        # if we are close stop
        if s < tol:
            break
        # update T
        al = 2*al
        for i in range(11):
            # determine Tt
            X = T - al*Gp
            if rotation_method == 'orthogonal':
                U, D, V = np.linalg.svd(X, full_matrices=False)
                Tt = U.dot(V)
            else:  # i.e. if rotation_method == 'oblique':
                v = 1/np.sqrt(np.sum(X**2, axis=0))
                Tt = X.dot(np.diag(v))
            # calculate objective using Tt
            if derivative_free:
                ft = ff(T=Tt, A=A, L=None)
            elif rotation_method == 'orthogonal':  # and not derivative_free
                L = A.dot(Tt)
                ft, Gq = vgQ(L=L)
            else:  # i.e. rotation_method == 'oblique' and not derivative_free
                Ti = np.linalg.inv(Tt)
                L = A.dot(Ti.T)
                ft, Gq = vgQ(L=L)
            # if sufficient improvement in objective -> use this T
            if ft < f-.5*s**2*al:
                break
            al = al/2
        # post processing for next iteration
        T = Tt
        f = ft
        if derivative_free:
            G = Gff(T)
        elif rotation_method == 'orthogonal':  # and not derivative_free
            G = (A.T).dot(Gq)
        else:  # i.e. rotation_method == 'oblique' and not derivative_free
            G = -((L.T).dot(Gq).dot(Ti)).T
    # post processing
    Th = T
    Lh = rotateA(A, T, rotation_method=rotation_method)
    Phi = (T.T).dot(T)
    return Lh, Phi, Th, table


def Gf(T, ff):
    """
    Subroutine for the gradient of f using numerical derivatives.
    """
    k = T.shape[0]
    ep = 1e-4
    G = np.zeros((k, k))
    for r in range(k):
        for s in range(k):
            dT = np.zeros((k, k))
            dT[r, s] = ep
            G[r, s] = (ff(T+dT)-ff(T-dT))/(2*ep)
    return G


def rotateA(A, T, rotation_method='orthogonal'):
    r"""
    For orthogonal rotation methods :math:`L=AT`, where :math:`T` is an
    orthogonal matrix. For oblique rotation matrices :math:`L=A(T^*)^{-1}`,
    where :math:`T` is a normal matrix, i.e., :math:`TT^*=T^*T`. Oblique
    rotations relax the orthogonality constraint in order to gain simplicity
    in the interpretation.
    """
    if rotation_method == 'orthogonal':
        L = A.dot(T)
    elif rotation_method == 'oblique':
        L = A.dot(np.linalg.inv(T.T))
    else:  # i.e. if rotation_method == 'oblique':
        raise ValueError('rotation_method should be one of '
                         '{orthogonal, oblique}')
    return L


def oblimin_objective(L=None, A=None, T=None, gamma=0,
                      rotation_method='orthogonal',
                      return_gradient=True):
    r"""
    Objective function for the oblimin family for orthogonal or
    oblique rotation wich minimizes:

    .. math::
        \phi(L) = \frac{1}{4}(L\circ L,(I-\gamma C)(L\circ L)N),

    where :math:`L` is a :math:`p\times k` matrix, :math:`N` is
    :math:`k\times k`
    matrix with zeros on the diagonal and ones elsewhere, :math:`C` is a
    :math:`p\times p` matrix with elements equal to :math:`1/p`,
    :math:`(X,Y)=\operatorname{Tr}(X^*Y)` is the Frobenius norm and
    :math:`\circ`
    is the element-wise product or Hadamard product.

    The gradient is given by

    .. math::
        L\circ\left[(I-\gamma C) (L \circ L)N\right].

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be
    provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L`
    satisfies

    .. math::
        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    The oblimin family is parametrized by the parameter :math:`\gamma`. For
    orthogonal rotations:

    * :math:`\gamma=0` corresponds to quartimax,
    * :math:`\gamma=\frac{1}{2}` corresponds to biquartimax,
    * :math:`\gamma=1` corresponds to varimax,
    * :math:`\gamma=\frac{1}{p}` corresponds to equamax.
    For oblique rotations rotations:

    * :math:`\gamma=0` corresponds to quartimin,
    * :math:`\gamma=\frac{1}{2}` corresponds to biquartimin.

    Parameters
    ----------
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    gamma : float (default 0)
        a parameter
    rotation_method : str
        should be one of {orthogonal, oblique}
    return_gradient : bool (default True)
        toggles return of gradient
    """
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method=rotation_method)
    p, k = L.shape
    L2 = L**2
    N = np.ones((k, k))-np.eye(k)
    if np.isclose(gamma, 0):
        X = L2.dot(N)
    else:
        C = np.ones((p, p))/p
        X = (np.eye(p) - gamma*C).dot(L2).dot(N)
    phi = np.sum(L2*X)/4
    if return_gradient:
        Gphi = L*X
        return phi, Gphi
    else:
        return phi


def orthomax_objective(L=None, A=None, T=None, gamma=0, return_gradient=True):
    r"""
    Objective function for the orthomax family for orthogonal
    rotation wich minimizes the following objective:

    .. math::
        \phi(L) = -\frac{1}{4}(L\circ L,(I-\gamma C)(L\circ L)),

    where :math:`0\leq\gamma\leq1`, :math:`L` is a :math:`p\times k` matrix,
    :math:`C` is a  :math:`p\times p` matrix with elements equal to
    :math:`1/p`,
    :math:`(X,Y)=\operatorname{Tr}(X^*Y)` is the Frobenius norm and
    :math:`\circ` is the element-wise product or Hadamard product.

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be
    provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix.

    The orthomax family is parametrized by the parameter :math:`\gamma`:

    * :math:`\gamma=0` corresponds to quartimax,
    * :math:`\gamma=\frac{1}{2}` corresponds to biquartimax,
    * :math:`\gamma=1` corresponds to varimax,
    * :math:`\gamma=\frac{1}{p}` corresponds to equamax.

    Parameters
    ----------
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    gamma : float (default 0)
        a parameter
    return_gradient : bool (default True)
        toggles return of gradient
    """
    assert 0 <= gamma <= 1, "Gamma should be between 0 and 1"
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method='orthogonal')
    p, k = L.shape
    L2 = L**2
    if np.isclose(gamma, 0):
        X = L2
    else:
        C = np.ones((p, p))/p
        X = (np.eye(p)-gamma*C).dot(L2)
    phi = -np.sum(L2*X)/4
    if return_gradient:
        Gphi = -L*X
        return phi, Gphi
    else:
        return phi


def CF_objective(L=None, A=None, T=None, kappa=0,
                 rotation_method='orthogonal',
                 return_gradient=True):
    r"""
    Objective function for the Crawford-Ferguson family for orthogonal
    and oblique rotation wich minimizes the following objective:

    .. math::
        \phi(L) =\frac{1-\kappa}{4} (L\circ L,(L\circ L)N)
                  -\frac{1}{4}(L\circ L,M(L\circ L)),

    where :math:`0\leq\kappa\leq1`, :math:`L` is a :math:`p\times k` matrix,
    :math:`N` is :math:`k\times k` matrix with zeros on the diagonal and ones
    elsewhere,
    :math:`M` is :math:`p\times p` matrix with zeros on the diagonal and ones
    elsewhere
    :math:`(X,Y)=\operatorname{Tr}(X^*Y)` is the Frobenius norm and
    :math:`\circ` is the element-wise product or Hadamard product.

    The gradient is given by

    .. math::
       d\phi(L) = (1-\kappa) L\circ\left[(L\circ L)N\right]
                   -\kappa L\circ \left[M(L\circ L)\right].

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be
    provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L`
    satisfies

    .. math::
        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    For orthogonal rotations the oblimin (and orthomax) family of rotations is
    equivalent to the Crawford-Ferguson family. To be more precise:

    * :math:`\kappa=0` corresponds to quartimax,
    * :math:`\kappa=\frac{1}{p}` corresponds to variamx,
    * :math:`\kappa=\frac{k-1}{p+k-2}` corresponds to parsimax,
    * :math:`\kappa=1` corresponds to factor parsimony.

    Parameters
    ----------
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    gamma : float (default 0)
        a parameter
    rotation_method : str
        should be one of {orthogonal, oblique}
    return_gradient : bool (default True)
        toggles return of gradient
    """
    assert 0 <= kappa <= 1, "Kappa should be between 0 and 1"
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method=rotation_method)
    p, k = L.shape
    L2 = L**2
    X = None
    if not np.isclose(kappa, 1):
        N = np.ones((k, k)) - np.eye(k)
        X = (1 - kappa)*L2.dot(N)
    if not np.isclose(kappa, 0):
        M = np.ones((p, p)) - np.eye(p)
        if X is None:
            X = kappa*M.dot(L2)
        else:
            X += kappa*M.dot(L2)
    phi = np.sum(L2 * X) / 4
    if return_gradient:
        Gphi = L*X
        return phi, Gphi
    else:
        return phi


def vgQ_target(H, L=None, A=None, T=None, rotation_method='orthogonal'):
    r"""
    Subroutine for the value of vgQ using orthogonal or oblique rotation
    towards a target matrix, i.e., we minimize:

    .. math::
        \phi(L) =\frac{1}{2}\|L-H\|^2

    and the gradient is given by

    .. math::
        d\phi(L)=L-H.

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be
    provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L`
    satisfies

    .. math::
        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    Parameters
    ----------
    H : numpy matrix
        target matrix
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    rotation_method : str
        should be one of {orthogonal, oblique}
    """
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method=rotation_method)
    q = np.linalg.norm(L-H, 'fro')**2
    Gq = 2*(L-H)
    return q, Gq


def ff_target(H, L=None, A=None, T=None, rotation_method='orthogonal'):
    r"""
    Subroutine for the value of f using (orthogonal or oblique) rotation
    towards a target matrix, i.e., we minimize:

    .. math::
        \phi(L) =\frac{1}{2}\|L-H\|^2.

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be
    provided. For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix. For oblique rotations
    :math:`L` satisfies

    .. math::
        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    Parameters
    ----------
    H : numpy matrix
        target matrix
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    rotation_method : str
        should be one of {orthogonal, oblique}
    """
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method=rotation_method)
    return np.linalg.norm(L-H, 'fro')**2


def vgQ_partial_target(H, W=None, L=None, A=None, T=None):
    r"""
    Subroutine for the value of vgQ using orthogonal rotation towards a partial
    target matrix, i.e., we minimize:

    .. math::
        \phi(L) =\frac{1}{2}\|W\circ(L-H)\|^2,

    where :math:`\circ` is the element-wise product or Hadamard product and
    :math:`W` is a matrix whose entries can only be one or zero. The gradient
    is given by

    .. math::
        d\phi(L)=W\circ(L-H).

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be
    provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix.

    Parameters
    ----------
    H : numpy matrix
        target matrix
    W : numpy matrix (default matrix with equal weight one for all entries)
        matrix with weights, entries can either be one or zero
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    """
    if W is None:
        return vgQ_target(H, L=L, A=A, T=T)
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method='orthogonal')
    q = np.linalg.norm(W*(L-H), 'fro')**2
    Gq = 2*W*(L-H)
    return q, Gq


def ff_partial_target(H, W=None, L=None, A=None, T=None):
    r"""
    Subroutine for the value of vgQ using orthogonal rotation towards a partial
    target matrix, i.e., we minimize:

    .. math::
        \phi(L) =\frac{1}{2}\|W\circ(L-H)\|^2,

    where :math:`\circ` is the element-wise product or Hadamard product and
    :math:`W` is a matrix whose entries can only be one or zero. Either
    :math:`L` should be provided or :math:`A` and :math:`T` should be provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix.

    Parameters
    ----------
    H : numpy matrix
        target matrix
    W : numpy matrix (default matrix with equal weight one for all entries)
        matrix with weights, entries can either be one or zero
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    """
    if W is None:
        return ff_target(H, L=L, A=A, T=T)
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method='orthogonal')
    q = np.linalg.norm(W*(L-H), 'fro')**2
    return q
