# -*- coding: utf-8 -*-


from ._analytic_rotation import target_rotation
from ._gpa_rotation import oblimin_objective, orthomax_objective, CF_objective
from ._gpa_rotation import ff_partial_target, ff_target
from ._gpa_rotation import vgQ_partial_target, vgQ_target
from ._gpa_rotation import rotateA, GPA

__all__ = []


def rotate_factors(A, method, *method_args, **algorithm_kwargs):
    r"""
    Subroutine for orthogonal and oblique rotation of the matrix :math:`A`.
    For orthogonal rotations :math:`A` is rotated to :math:`L` according to

    .. math::

        L =  AT,

    where :math:`T` is an orthogonal matrix. And, for oblique rotations
    :math:`A` is rotated to :math:`L` according to

    .. math::

        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    Parameters
    ----------
    A : numpy matrix (default None)
        non rotated factors
    method : str
        should be one of the methods listed below
    method_args : list
        additional arguments that should be provided with each method
    algorithm_kwargs : dictionary
        algorithm : str (default gpa)
            should be one of:

            * 'gpa': a numerical method
            * 'gpa_der_free': a derivative free numerical method
            * 'analytic' : an analytic method

        Depending on the algorithm, there are algorithm specific keyword
        arguments. For the gpa and gpa_der_free, the following
        keyword arguments are available:

        max_tries : int (default 501)
            maximum number of iterations

        tol : float
            stop criterion, algorithm stops if Frobenius norm of gradient is
            smaller then tol

        For analytic, the supported arguments depend on the method, see above.

        See the lower level functions for more details.

    Returns
    -------
    The tuple :math:`(L,T)`

    Notes
    -----
    What follows is a list of available methods. Depending on the method
    additional argument are required and different algorithms
    are available. The algorithm_kwargs are additional keyword arguments
    passed to the selected algorithm (see the parameters section).
    Unless stated otherwise, only the gpa and
    gpa_der_free algorithm are available.

    Below,

        * :math:`L` is a :math:`p\times k` matrix;
        * :math:`N` is :math:`k\times k` matrix with zeros on the diagonal and ones
          elsewhere;
        * :math:`M` is :math:`p\times p` matrix with zeros on the diagonal and ones
          elsewhere;
        * :math:`C` is a :math:`p\times p` matrix with elements equal to
          :math:`1/p`;
        * :math:`(X,Y)=\operatorname{Tr}(X^*Y)` is the Frobenius norm;
        * :math:`\circ` is the element-wise product or Hadamard product.

    oblimin : orthogonal or oblique rotation that minimizes
        .. math::
            \phi(L) = \frac{1}{4}(L\circ L,(I-\gamma C)(L\circ L)N).

        For orthogonal rotations:

        * :math:`\gamma=0` corresponds to quartimax,
        * :math:`\gamma=\frac{1}{2}` corresponds to biquartimax,
        * :math:`\gamma=1` corresponds to varimax,
        * :math:`\gamma=\frac{1}{p}` corresponds to equamax.

        For oblique rotations rotations:

        * :math:`\gamma=0` corresponds to quartimin,
        * :math:`\gamma=\frac{1}{2}` corresponds to biquartimin.

        method_args:

        gamma : float
            oblimin family parameter
        rotation_method : str
            should be one of {orthogonal, oblique}

    orthomax : orthogonal rotation that minimizes

        .. math::
            \phi(L) = -\frac{1}{4}(L\circ L,(I-\gamma C)(L\circ L)),

        where :math:`0\leq\gamma\leq1`. The orthomax family is equivalent to
        the oblimin family (when restricted to orthogonal rotations).
        Furthermore,

        * :math:`\gamma=0` corresponds to quartimax,
        * :math:`\gamma=\frac{1}{2}` corresponds to biquartimax,
        * :math:`\gamma=1` corresponds to varimax,
        * :math:`\gamma=\frac{1}{p}` corresponds to equamax.

        method_args:

        gamma : float (between 0 and 1)
            orthomax family parameter

    CF : Crawford-Ferguson family for orthogonal and oblique rotation which
    minimizes:

        .. math::

            \phi(L) =\frac{1-\kappa}{4} (L\circ L,(L\circ L)N)
                     -\frac{1}{4}(L\circ L,M(L\circ L)),

        where :math:`0\leq\kappa\leq1`. For orthogonal rotations the oblimin
        (and orthomax) family of rotations is equivalent to the
        Crawford-Ferguson family.
        To be more precise:

        * :math:`\kappa=0` corresponds to quartimax,
        * :math:`\kappa=\frac{1}{p}` corresponds to varimax,
        * :math:`\kappa=\frac{k-1}{p+k-2}` corresponds to parsimax,
        * :math:`\kappa=1` corresponds to factor parsimony.

        method_args:

        kappa : float (between 0 and 1)
            Crawford-Ferguson family parameter
        rotation_method : str
            should be one of {orthogonal, oblique}

    quartimax : orthogonal rotation method
        minimizes the orthomax objective with :math:`\gamma=0`

    biquartimax : orthogonal rotation method
        minimizes the orthomax objective with :math:`\gamma=\frac{1}{2}`

    varimax : orthogonal rotation method
        minimizes the orthomax objective with :math:`\gamma=1`

    equamax : orthogonal rotation method
        minimizes the orthomax objective with :math:`\gamma=\frac{1}{p}`

    parsimax : orthogonal rotation method
        minimizes the Crawford-Ferguson family objective with
        :math:`\kappa=\frac{k-1}{p+k-2}`

    parsimony : orthogonal rotation method
        minimizes the Crawford-Ferguson family objective with :math:`\kappa=1`

    quartimin : oblique rotation method that minimizes
        minimizes the oblimin objective with :math:`\gamma=0`

    quartimin : oblique rotation method that minimizes
        minimizes the oblimin objective with :math:`\gamma=\frac{1}{2}`

    target : orthogonal or oblique rotation that rotates towards a target

    matrix : math:`H` by minimizing the objective

        .. math::

            \phi(L) =\frac{1}{2}\|L-H\|^2.

        method_args:

        H : numpy matrix
            target matrix
        rotation_method : str
            should be one of {orthogonal, oblique}

        For orthogonal rotations the algorithm can be set to analytic in which
        case the following keyword arguments are available:

        full_rank : bool (default False)
            if set to true full rank is assumed

    partial_target : orthogonal (default) or oblique rotation that partially
    rotates towards a target matrix :math:`H` by minimizing the objective:

        .. math::

            \phi(L) =\frac{1}{2}\|W\circ(L-H)\|^2.

        method_args:

        H : numpy matrix
            target matrix
        W : numpy matrix (default matrix with equal weight one for all entries)
            matrix with weights, entries can either be one or zero

    Examples
    --------
    >>> A = np.random.randn(8,2)
    >>> L, T = rotate_factors(A,'varimax')
    >>> np.allclose(L,A.dot(T))
    >>> L, T = rotate_factors(A,'orthomax',0.5)
    >>> np.allclose(L,A.dot(T))
    >>> L, T = rotate_factors(A,'quartimin',0.5)
    >>> np.allclose(L,A.dot(np.linalg.inv(T.T)))
    """
    if 'algorithm' in algorithm_kwargs:
        algorithm = algorithm_kwargs['algorithm']
        algorithm_kwargs.pop('algorithm')
    else:
        algorithm = 'gpa'
    assert not ('rotation_method' in algorithm_kwargs), (
        'rotation_method cannot be provided as keyword argument')
    L = None
    T = None
    ff = None
    vgQ = None
    p, k = A.shape
    # set ff or vgQ to appropriate objective function, compute solution using
    # recursion or analytically compute solution
    if method == 'orthomax':
        assert len(method_args) == 1, ('Only %s family parameter should be '
                                       'provided' % method)
        rotation_method = 'orthogonal'
        gamma = method_args[0]
        if algorithm == 'gpa':
            vgQ = lambda L=None, A=None, T=None: orthomax_objective(
                L=L, A=A, T=T, gamma=gamma, return_gradient=True)
        elif algorithm == 'gpa_der_free':
            ff = lambda L=None, A=None, T=None: orthomax_objective(
                L=L, A=A, T=T, gamma=gamma, return_gradient=False)
        else:
            raise ValueError('Algorithm %s is not possible for %s '
                             'rotation' % (algorithm, method))
    elif method == 'oblimin':
        assert len(method_args) == 2, ('Both %s family parameter and '
                                       'rotation_method should be '
                                       'provided' % method)
        rotation_method = method_args[1]
        assert rotation_method in ['orthogonal', 'oblique'], (
            'rotation_method should be one of {orthogonal, oblique}')
        gamma = method_args[0]
        if algorithm == 'gpa':
            vgQ = lambda L=None, A=None, T=None: oblimin_objective(
                L=L, A=A, T=T, gamma=gamma, return_gradient=True)
        elif algorithm == 'gpa_der_free':
            ff = lambda L=None, A=None, T=None: oblimin_objective(
                L=L, A=A, T=T, gamma=gamma, rotation_method=rotation_method,
                return_gradient=False)
        else:
            raise ValueError('Algorithm %s is not possible for %s '
                             'rotation' % (algorithm, method))
    elif method == 'CF':
        assert len(method_args) == 2, ('Both %s family parameter and '
                                       'rotation_method should be provided'
                                       % method)
        rotation_method = method_args[1]
        assert rotation_method in ['orthogonal', 'oblique'], (
            'rotation_method should be one of {orthogonal, oblique}')
        kappa = method_args[0]
        if algorithm == 'gpa':
            vgQ = lambda L=None, A=None, T=None: CF_objective(
                L=L, A=A, T=T, kappa=kappa, rotation_method=rotation_method,
                return_gradient=True)
        elif algorithm == 'gpa_der_free':
            ff = lambda L=None, A=None, T=None: CF_objective(
                L=L, A=A, T=T, kappa=kappa, rotation_method=rotation_method,
                return_gradient=False)
        else:
            raise ValueError('Algorithm %s is not possible for %s '
                             'rotation' % (algorithm, method))
    elif method == 'quartimax':
        return rotate_factors(A, 'orthomax', 0, **algorithm_kwargs)
    elif method == 'biquartimax':
        return rotate_factors(A, 'orthomax', 0.5, **algorithm_kwargs)
    elif method == 'varimax':
        return rotate_factors(A, 'orthomax', 1, **algorithm_kwargs)
    elif method == 'equamax':
        return rotate_factors(A, 'orthomax', 1/p, **algorithm_kwargs)
    elif method == 'parsimax':
        return rotate_factors(A, 'CF', (k-1)/(p+k-2),
                              'orthogonal', **algorithm_kwargs)
    elif method == 'parsimony':
        return rotate_factors(A, 'CF', 1, 'orthogonal', **algorithm_kwargs)
    elif method == 'quartimin':
        return rotate_factors(A, 'oblimin', 0, 'oblique', **algorithm_kwargs)
    elif method == 'biquartimin':
        return rotate_factors(A, 'oblimin', 0.5, 'oblique', **algorithm_kwargs)
    elif method == 'target':
        assert len(method_args) == 2, (
            'only the rotation target and orthogonal/oblique should be provide'
            ' for %s rotation' % method)
        H = method_args[0]
        rotation_method = method_args[1]
        assert rotation_method in ['orthogonal', 'oblique'], (
            'rotation_method should be one of {orthogonal, oblique}')
        if algorithm == 'gpa':
            vgQ = lambda L=None, A=None, T=None: vgQ_target(
                H, L=L, A=A, T=T, rotation_method=rotation_method)
        elif algorithm == 'gpa_der_free':
            ff = lambda L=None, A=None, T=None: ff_target(
                H, L=L, A=A, T=T, rotation_method=rotation_method)
        elif algorithm == 'analytic':
            assert rotation_method == 'orthogonal', (
                'For analytic %s rotation only orthogonal rotation is '
                'supported')
            T = target_rotation(A, H, **algorithm_kwargs)
        else:
            raise ValueError('Algorithm %s is not possible for %s rotation'
                             % (algorithm, method))
    elif method == 'partial_target':
        assert len(method_args) == 2, ('2 additional arguments are expected '
                                       'for %s rotation' % method)
        H = method_args[0]
        W = method_args[1]
        rotation_method = 'orthogonal'
        if algorithm == 'gpa':
            vgQ = lambda L=None, A=None, T=None: vgQ_partial_target(
                H, W=W, L=L, A=A, T=T)
        elif algorithm == 'gpa_der_free':
            ff = lambda L=None, A=None, T=None: ff_partial_target(
                H, W=W, L=L, A=A, T=T)
        else:
            raise ValueError('Algorithm %s is not possible for %s '
                             'rotation' % (algorithm, method))
    else:
        raise ValueError('Invalid method')
    # compute L and T if not already done
    if T is None:
        L, phi, T, table = GPA(A, vgQ=vgQ, ff=ff,
                               rotation_method=rotation_method,
                               **algorithm_kwargs)
    if L is None:
        assert T is not None, 'Cannot compute L without T'
        L = rotateA(A, T, rotation_method=rotation_method)
    return L, T
