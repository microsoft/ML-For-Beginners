# Author: Travis Oliphant 2001
# Author: Nathan Woods 2013 (nquad &c)
import sys
import warnings
from functools import partial

from . import _quadpack
import numpy as np

__all__ = ["quad", "dblquad", "tplquad", "nquad", "IntegrationWarning"]


error = _quadpack.error

class IntegrationWarning(UserWarning):
    """
    Warning on issues during integration.
    """
    pass


def quad(func, a, b, args=(), full_output=0, epsabs=1.49e-8, epsrel=1.49e-8,
         limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50,
         limlst=50, complex_func=False):
    """
    Compute a definite integral.

    Integrate func from `a` to `b` (possibly infinite interval) using a
    technique from the Fortran library QUADPACK.

    Parameters
    ----------
    func : {function, scipy.LowLevelCallable}
        A Python function or method to integrate. If `func` takes many
        arguments, it is integrated along the axis corresponding to the
        first argument.

        If the user desires improved integration performance, then `f` may
        be a `scipy.LowLevelCallable` with one of the signatures::

            double func(double x)
            double func(double x, void *user_data)
            double func(int n, double *xx)
            double func(int n, double *xx, void *user_data)

        The ``user_data`` is the data contained in the `scipy.LowLevelCallable`.
        In the call forms with ``xx``,  ``n`` is the length of the ``xx``
        array which contains ``xx[0] == x`` and the rest of the items are
        numbers contained in the ``args`` argument of quad.

        In addition, certain ctypes call signatures are supported for
        backward compatibility, but those should not be used in new code.
    a : float
        Lower limit of integration (use -numpy.inf for -infinity).
    b : float
        Upper limit of integration (use numpy.inf for +infinity).
    args : tuple, optional
        Extra arguments to pass to `func`.
    full_output : int, optional
        Non-zero to return a dictionary of integration information.
        If non-zero, warning messages are also suppressed and the
        message is appended to the output tuple.
    complex_func : bool, optional
        Indicate if the function's (`func`) return type is real
        (``complex_func=False``: default) or complex (``complex_func=True``).
        In both cases, the function's argument is real.
        If full_output is also non-zero, the `infodict`, `message`, and
        `explain` for the real and complex components are returned in
        a dictionary with keys "real output" and "imag output".

    Returns
    -------
    y : float
        The integral of func from `a` to `b`.
    abserr : float
        An estimate of the absolute error in the result.
    infodict : dict
        A dictionary containing additional information.
    message
        A convergence message.
    explain
        Appended only with 'cos' or 'sin' weighting and infinite
        integration limits, it contains an explanation of the codes in
        infodict['ierlst']

    Other Parameters
    ----------------
    epsabs : float or int, optional
        Absolute error tolerance. Default is 1.49e-8. `quad` tries to obtain
        an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
        where ``i`` = integral of `func` from `a` to `b`, and ``result`` is the
        numerical approximation. See `epsrel` below.
    epsrel : float or int, optional
        Relative error tolerance. Default is 1.49e-8.
        If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
        and ``50 * (machine epsilon)``. See `epsabs` above.
    limit : float or int, optional
        An upper bound on the number of subintervals used in the adaptive
        algorithm.
    points : (sequence of floats,ints), optional
        A sequence of break points in the bounded integration interval
        where local difficulties of the integrand may occur (e.g.,
        singularities, discontinuities). The sequence does not have
        to be sorted. Note that this option cannot be used in conjunction
        with ``weight``.
    weight : float or int, optional
        String indicating weighting function. Full explanation for this
        and the remaining arguments can be found below.
    wvar : optional
        Variables for use with weighting functions.
    wopts : optional
        Optional input for reusing Chebyshev moments.
    maxp1 : float or int, optional
        An upper bound on the number of Chebyshev moments.
    limlst : int, optional
        Upper bound on the number of cycles (>=3) for use with a sinusoidal
        weighting and an infinite end-point.

    See Also
    --------
    dblquad : double integral
    tplquad : triple integral
    nquad : n-dimensional integrals (uses `quad` recursively)
    fixed_quad : fixed-order Gaussian quadrature
    quadrature : adaptive Gaussian quadrature
    odeint : ODE integrator
    ode : ODE integrator
    simpson : integrator for sampled data
    romb : integrator for sampled data
    scipy.special : for coefficients and roots of orthogonal polynomials

    Notes
    -----
    For valid results, the integral must converge; behavior for divergent
    integrals is not guaranteed.

    **Extra information for quad() inputs and outputs**

    If full_output is non-zero, then the third output argument
    (infodict) is a dictionary with entries as tabulated below. For
    infinite limits, the range is transformed to (0,1) and the
    optional outputs are given with respect to this transformed range.
    Let M be the input argument limit and let K be infodict['last'].
    The entries are:

    'neval'
        The number of function evaluations.
    'last'
        The number, K, of subintervals produced in the subdivision process.
    'alist'
        A rank-1 array of length M, the first K elements of which are the
        left end points of the subintervals in the partition of the
        integration range.
    'blist'
        A rank-1 array of length M, the first K elements of which are the
        right end points of the subintervals.
    'rlist'
        A rank-1 array of length M, the first K elements of which are the
        integral approximations on the subintervals.
    'elist'
        A rank-1 array of length M, the first K elements of which are the
        moduli of the absolute error estimates on the subintervals.
    'iord'
        A rank-1 integer array of length M, the first L elements of
        which are pointers to the error estimates over the subintervals
        with ``L=K`` if ``K<=M/2+2`` or ``L=M+1-K`` otherwise. Let I be the
        sequence ``infodict['iord']`` and let E be the sequence
        ``infodict['elist']``.  Then ``E[I[1]], ..., E[I[L]]`` forms a
        decreasing sequence.

    If the input argument points is provided (i.e., it is not None),
    the following additional outputs are placed in the output
    dictionary. Assume the points sequence is of length P.

    'pts'
        A rank-1 array of length P+2 containing the integration limits
        and the break points of the intervals in ascending order.
        This is an array giving the subintervals over which integration
        will occur.
    'level'
        A rank-1 integer array of length M (=limit), containing the
        subdivision levels of the subintervals, i.e., if (aa,bb) is a
        subinterval of ``(pts[1], pts[2])`` where ``pts[0]`` and ``pts[2]``
        are adjacent elements of ``infodict['pts']``, then (aa,bb) has level l
        if ``|bb-aa| = |pts[2]-pts[1]| * 2**(-l)``.
    'ndin'
        A rank-1 integer array of length P+2. After the first integration
        over the intervals (pts[1], pts[2]), the error estimates over some
        of the intervals may have been increased artificially in order to
        put their subdivision forward. This array has ones in slots
        corresponding to the subintervals for which this happens.

    **Weighting the integrand**

    The input variables, *weight* and *wvar*, are used to weight the
    integrand by a select list of functions. Different integration
    methods are used to compute the integral with these weighting
    functions, and these do not support specifying break points. The
    possible values of weight and the corresponding weighting functions are.

    ==========  ===================================   =====================
    ``weight``  Weight function used                  ``wvar``
    ==========  ===================================   =====================
    'cos'       cos(w*x)                              wvar = w
    'sin'       sin(w*x)                              wvar = w
    'alg'       g(x) = ((x-a)**alpha)*((b-x)**beta)   wvar = (alpha, beta)
    'alg-loga'  g(x)*log(x-a)                         wvar = (alpha, beta)
    'alg-logb'  g(x)*log(b-x)                         wvar = (alpha, beta)
    'alg-log'   g(x)*log(x-a)*log(b-x)                wvar = (alpha, beta)
    'cauchy'    1/(x-c)                               wvar = c
    ==========  ===================================   =====================

    wvar holds the parameter w, (alpha, beta), or c depending on the weight
    selected. In these expressions, a and b are the integration limits.

    For the 'cos' and 'sin' weighting, additional inputs and outputs are
    available.

    For finite integration limits, the integration is performed using a
    Clenshaw-Curtis method which uses Chebyshev moments. For repeated
    calculations, these moments are saved in the output dictionary:

    'momcom'
        The maximum level of Chebyshev moments that have been computed,
        i.e., if ``M_c`` is ``infodict['momcom']`` then the moments have been
        computed for intervals of length ``|b-a| * 2**(-l)``,
        ``l=0,1,...,M_c``.
    'nnlog'
        A rank-1 integer array of length M(=limit), containing the
        subdivision levels of the subintervals, i.e., an element of this
        array is equal to l if the corresponding subinterval is
        ``|b-a|* 2**(-l)``.
    'chebmo'
        A rank-2 array of shape (25, maxp1) containing the computed
        Chebyshev moments. These can be passed on to an integration
        over the same interval by passing this array as the second
        element of the sequence wopts and passing infodict['momcom'] as
        the first element.

    If one of the integration limits is infinite, then a Fourier integral is
    computed (assuming w neq 0). If full_output is 1 and a numerical error
    is encountered, besides the error message attached to the output tuple,
    a dictionary is also appended to the output tuple which translates the
    error codes in the array ``info['ierlst']`` to English messages. The
    output information dictionary contains the following entries instead of
    'last', 'alist', 'blist', 'rlist', and 'elist':

    'lst'
        The number of subintervals needed for the integration (call it ``K_f``).
    'rslst'
        A rank-1 array of length M_f=limlst, whose first ``K_f`` elements
        contain the integral contribution over the interval
        ``(a+(k-1)c, a+kc)`` where ``c = (2*floor(|w|) + 1) * pi / |w|``
        and ``k=1,2,...,K_f``.
    'erlst'
        A rank-1 array of length ``M_f`` containing the error estimate
        corresponding to the interval in the same position in
        ``infodict['rslist']``.
    'ierlst'
        A rank-1 integer array of length ``M_f`` containing an error flag
        corresponding to the interval in the same position in
        ``infodict['rslist']``.  See the explanation dictionary (last entry
        in the output tuple) for the meaning of the codes.


    **Details of QUADPACK level routines**

    `quad` calls routines from the FORTRAN library QUADPACK. This section
    provides details on the conditions for each routine to be called and a
    short description of each routine. The routine called depends on
    `weight`, `points` and the integration limits `a` and `b`.

    ================  ==============  ==========  =====================
    QUADPACK routine  `weight`        `points`    infinite bounds
    ================  ==============  ==========  =====================
    qagse             None            No          No
    qagie             None            No          Yes
    qagpe             None            Yes         No
    qawoe             'sin', 'cos'    No          No
    qawfe             'sin', 'cos'    No          either `a` or `b`
    qawse             'alg*'          No          No
    qawce             'cauchy'        No          No
    ================  ==============  ==========  =====================

    The following provides a short description from [1]_ for each
    routine.

    qagse
        is an integrator based on globally adaptive interval
        subdivision in connection with extrapolation, which will
        eliminate the effects of integrand singularities of
        several types.
    qagie
        handles integration over infinite intervals. The infinite range is
        mapped onto a finite interval and subsequently the same strategy as
        in ``QAGS`` is applied.
    qagpe
        serves the same purposes as QAGS, but also allows the
        user to provide explicit information about the location
        and type of trouble-spots i.e. the abscissae of internal
        singularities, discontinuities and other difficulties of
        the integrand function.
    qawoe
        is an integrator for the evaluation of
        :math:`\\int^b_a \\cos(\\omega x)f(x)dx` or
        :math:`\\int^b_a \\sin(\\omega x)f(x)dx`
        over a finite interval [a,b], where :math:`\\omega` and :math:`f`
        are specified by the user. The rule evaluation component is based
        on the modified Clenshaw-Curtis technique

        An adaptive subdivision scheme is used in connection
        with an extrapolation procedure, which is a modification
        of that in ``QAGS`` and allows the algorithm to deal with
        singularities in :math:`f(x)`.
    qawfe
        calculates the Fourier transform
        :math:`\\int^\\infty_a \\cos(\\omega x)f(x)dx` or
        :math:`\\int^\\infty_a \\sin(\\omega x)f(x)dx`
        for user-provided :math:`\\omega` and :math:`f`. The procedure of
        ``QAWO`` is applied on successive finite intervals, and convergence
        acceleration by means of the :math:`\\varepsilon`-algorithm is applied
        to the series of integral approximations.
    qawse
        approximate :math:`\\int^b_a w(x)f(x)dx`, with :math:`a < b` where
        :math:`w(x) = (x-a)^{\\alpha}(b-x)^{\\beta}v(x)` with
        :math:`\\alpha,\\beta > -1`, where :math:`v(x)` may be one of the
        following functions: :math:`1`, :math:`\\log(x-a)`, :math:`\\log(b-x)`,
        :math:`\\log(x-a)\\log(b-x)`.

        The user specifies :math:`\\alpha`, :math:`\\beta` and the type of the
        function :math:`v`. A globally adaptive subdivision strategy is
        applied, with modified Clenshaw-Curtis integration on those
        subintervals which contain `a` or `b`.
    qawce
        compute :math:`\\int^b_a f(x) / (x-c)dx` where the integral must be
        interpreted as a Cauchy principal value integral, for user specified
        :math:`c` and :math:`f`. The strategy is globally adaptive. Modified
        Clenshaw-Curtis integration is used on those intervals containing the
        point :math:`x = c`.

    **Integration of Complex Function of a Real Variable**

    A complex valued function, :math:`f`, of a real variable can be written as
    :math:`f = g + ih`.  Similarly, the integral of :math:`f` can be
    written as

    .. math::
        \\int_a^b f(x) dx = \\int_a^b g(x) dx + i\\int_a^b h(x) dx

    assuming that the integrals of :math:`g` and :math:`h` exist
    over the interval :math:`[a,b]` [2]_. Therefore, ``quad`` integrates
    complex-valued functions by integrating the real and imaginary components
    separately.


    References
    ----------

    .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
           Überhuber, Christoph W.; Kahaner, David (1983).
           QUADPACK: A subroutine package for automatic integration.
           Springer-Verlag.
           ISBN 978-3-540-12553-2.

    .. [2] McCullough, Thomas; Phillips, Keith (1973).
           Foundations of Analysis in the Complex Plane.
           Holt Rinehart Winston.
           ISBN 0-03-086370-8

    Examples
    --------
    Calculate :math:`\\int^4_0 x^2 dx` and compare with an analytic result

    >>> from scipy import integrate
    >>> import numpy as np
    >>> x2 = lambda x: x**2
    >>> integrate.quad(x2, 0, 4)
    (21.333333333333332, 2.3684757858670003e-13)
    >>> print(4**3 / 3.)  # analytical result
    21.3333333333

    Calculate :math:`\\int^\\infty_0 e^{-x} dx`

    >>> invexp = lambda x: np.exp(-x)
    >>> integrate.quad(invexp, 0, np.inf)
    (1.0, 5.842605999138044e-11)

    Calculate :math:`\\int^1_0 a x \\,dx` for :math:`a = 1, 3`

    >>> f = lambda x, a: a*x
    >>> y, err = integrate.quad(f, 0, 1, args=(1,))
    >>> y
    0.5
    >>> y, err = integrate.quad(f, 0, 1, args=(3,))
    >>> y
    1.5

    Calculate :math:`\\int^1_0 x^2 + y^2 dx` with ctypes, holding
    y parameter as 1::

        testlib.c =>
            double func(int n, double args[n]){
                return args[0]*args[0] + args[1]*args[1];}
        compile to library testlib.*

    ::

       from scipy import integrate
       import ctypes
       lib = ctypes.CDLL('/home/.../testlib.*') #use absolute path
       lib.func.restype = ctypes.c_double
       lib.func.argtypes = (ctypes.c_int,ctypes.c_double)
       integrate.quad(lib.func,0,1,(1))
       #(1.3333333333333333, 1.4802973661668752e-14)
       print((1.0**3/3.0 + 1.0) - (0.0**3/3.0 + 0.0)) #Analytic result
       # 1.3333333333333333

    Be aware that pulse shapes and other sharp features as compared to the
    size of the integration interval may not be integrated correctly using
    this method. A simplified example of this limitation is integrating a
    y-axis reflected step function with many zero values within the integrals
    bounds.

    >>> y = lambda x: 1 if x<=0 else 0
    >>> integrate.quad(y, -1, 1)
    (1.0, 1.1102230246251565e-14)
    >>> integrate.quad(y, -1, 100)
    (1.0000000002199108, 1.0189464580163188e-08)
    >>> integrate.quad(y, -1, 10000)
    (0.0, 0.0)

    """
    if not isinstance(args, tuple):
        args = (args,)

    # check the limits of integration: \int_a^b, expect a < b
    flip, a, b = b < a, min(a, b), max(a, b)

    if complex_func:
        def imfunc(x, *args):
            return func(x, *args).imag

        def refunc(x, *args):
            return func(x, *args).real

        re_retval = quad(refunc, a, b, args, full_output, epsabs,
                         epsrel, limit, points, weight, wvar, wopts,
                         maxp1, limlst, complex_func=False)
        im_retval = quad(imfunc, a, b, args, full_output, epsabs,
                         epsrel, limit, points, weight, wvar, wopts,
                         maxp1, limlst, complex_func=False)
        integral = re_retval[0] + 1j*im_retval[0]
        error_estimate = re_retval[1] + 1j*im_retval[1]
        retval = integral, error_estimate
        if full_output:
            msgexp = {}
            msgexp["real"] = re_retval[2:]
            msgexp["imag"] = im_retval[2:]
            retval = retval + (msgexp,)

        return retval

    if weight is None:
        retval = _quad(func, a, b, args, full_output, epsabs, epsrel, limit,
                       points)
    else:
        if points is not None:
            msg = ("Break points cannot be specified when using weighted integrand.\n"
                   "Continuing, ignoring specified points.")
            warnings.warn(msg, IntegrationWarning, stacklevel=2)
        retval = _quad_weight(func, a, b, args, full_output, epsabs, epsrel,
                              limlst, limit, maxp1, weight, wvar, wopts)

    if flip:
        retval = (-retval[0],) + retval[1:]

    ier = retval[-1]
    if ier == 0:
        return retval[:-1]

    msgs = {80: "A Python error occurred possibly while calling the function.",
             1: f"The maximum number of subdivisions ({limit}) has been achieved.\n  "
                f"If increasing the limit yields no improvement it is advised to "
                f"analyze \n  the integrand in order to determine the difficulties.  "
                f"If the position of a \n  local difficulty can be determined "
                f"(singularity, discontinuity) one will \n  probably gain from "
                f"splitting up the interval and calling the integrator \n  on the "
                f"subranges.  Perhaps a special-purpose integrator should be used.",
             2: "The occurrence of roundoff error is detected, which prevents \n  "
                "the requested tolerance from being achieved.  "
                "The error may be \n  underestimated.",
             3: "Extremely bad integrand behavior occurs at some points of the\n  "
                "integration interval.",
             4: "The algorithm does not converge.  Roundoff error is detected\n  "
                "in the extrapolation table.  It is assumed that the requested "
                "tolerance\n  cannot be achieved, and that the returned result "
                "(if full_output = 1) is \n  the best which can be obtained.",
             5: "The integral is probably divergent, or slowly convergent.",
             6: "The input is invalid.",
             7: "Abnormal termination of the routine.  The estimates for result\n  "
                "and error are less reliable.  It is assumed that the requested "
                "accuracy\n  has not been achieved.",
            'unknown': "Unknown error."}

    if weight in ['cos','sin'] and (b == np.inf or a == -np.inf):
        msgs[1] = (
            "The maximum number of cycles allowed has been achieved., e.e.\n  of "
            "subintervals (a+(k-1)c, a+kc) where c = (2*int(abs(omega)+1))\n  "
            "*pi/abs(omega), for k = 1, 2, ..., lst.  "
            "One can allow more cycles by increasing the value of limlst.  "
            "Look at info['ierlst'] with full_output=1."
        )
        msgs[4] = (
            "The extrapolation table constructed for convergence acceleration\n  of "
            "the series formed by the integral contributions over the cycles, \n  does "
            "not converge to within the requested accuracy.  "
            "Look at \n  info['ierlst'] with full_output=1."
        )
        msgs[7] = (
            "Bad integrand behavior occurs within one or more of the cycles.\n  "
            "Location and type of the difficulty involved can be determined from \n  "
            "the vector info['ierlist'] obtained with full_output=1."
        )
        explain = {1: "The maximum number of subdivisions (= limit) has been \n  "
                      "achieved on this cycle.",
                   2: "The occurrence of roundoff error is detected and prevents\n  "
                      "the tolerance imposed on this cycle from being achieved.",
                   3: "Extremely bad integrand behavior occurs at some points of\n  "
                      "this cycle.",
                   4: "The integral over this cycle does not converge (to within the "
                      "required accuracy) due to roundoff in the extrapolation "
                      "procedure invoked on this cycle.  It is assumed that the result "
                      "on this interval is the best which can be obtained.",
                   5: "The integral over this cycle is probably divergent or "
                      "slowly convergent."}

    try:
        msg = msgs[ier]
    except KeyError:
        msg = msgs['unknown']

    if ier in [1,2,3,4,5,7]:
        if full_output:
            if weight in ['cos', 'sin'] and (b == np.inf or a == -np.inf):
                return retval[:-1] + (msg, explain)
            else:
                return retval[:-1] + (msg,)
        else:
            warnings.warn(msg, IntegrationWarning, stacklevel=2)
            return retval[:-1]

    elif ier == 6:  # Forensic decision tree when QUADPACK throws ier=6
        if epsabs <= 0:  # Small error tolerance - applies to all methods
            if epsrel < max(50 * sys.float_info.epsilon, 5e-29):
                msg = ("If 'epsabs'<=0, 'epsrel' must be greater than both"
                       " 5e-29 and 50*(machine epsilon).")
            elif weight in ['sin', 'cos'] and (abs(a) + abs(b) == np.inf):
                msg = ("Sine or cosine weighted integrals with infinite domain"
                       " must have 'epsabs'>0.")

        elif weight is None:
            if points is None:  # QAGSE/QAGIE
                msg = ("Invalid 'limit' argument. There must be"
                       " at least one subinterval")
            else:  # QAGPE
                if not (min(a, b) <= min(points) <= max(points) <= max(a, b)):
                    msg = ("All break points in 'points' must lie within the"
                           " integration limits.")
                elif len(points) >= limit:
                    msg = (f"Number of break points ({len(points):d}) "
                           f"must be less than subinterval limit ({limit:d})")

        else:
            if maxp1 < 1:
                msg = "Chebyshev moment limit maxp1 must be >=1."

            elif weight in ('cos', 'sin') and abs(a+b) == np.inf:  # QAWFE
                msg = "Cycle limit limlst must be >=3."

            elif weight.startswith('alg'):  # QAWSE
                if min(wvar) < -1:
                    msg = "wvar parameters (alpha, beta) must both be >= -1."
                if b < a:
                    msg = "Integration limits a, b must satistfy a<b."

            elif weight == 'cauchy' and wvar in (a, b):
                msg = ("Parameter 'wvar' must not equal"
                       " integration limits 'a' or 'b'.")

    raise ValueError(msg)


def _quad(func,a,b,args,full_output,epsabs,epsrel,limit,points):
    infbounds = 0
    if (b != np.inf and a != -np.inf):
        pass   # standard integration
    elif (b == np.inf and a != -np.inf):
        infbounds = 1
        bound = a
    elif (b == np.inf and a == -np.inf):
        infbounds = 2
        bound = 0     # ignored
    elif (b != np.inf and a == -np.inf):
        infbounds = -1
        bound = b
    else:
        raise RuntimeError("Infinity comparisons don't work for you.")

    if points is None:
        if infbounds == 0:
            return _quadpack._qagse(func,a,b,args,full_output,epsabs,epsrel,limit)
        else:
            return _quadpack._qagie(func, bound, infbounds, args, full_output, 
                                    epsabs, epsrel, limit)
    else:
        if infbounds != 0:
            raise ValueError("Infinity inputs cannot be used with break points.")
        else:
            #Duplicates force function evaluation at singular points
            the_points = np.unique(points)
            the_points = the_points[a < the_points]
            the_points = the_points[the_points < b]
            the_points = np.concatenate((the_points, (0., 0.)))
            return _quadpack._qagpe(func, a, b, the_points, args, full_output,
                                    epsabs, epsrel, limit)


def _quad_weight(func, a, b, args, full_output, epsabs, epsrel,
                 limlst, limit, maxp1,weight, wvar, wopts):
    if weight not in ['cos','sin','alg','alg-loga','alg-logb','alg-log','cauchy']:
        raise ValueError("%s not a recognized weighting function." % weight)

    strdict = {'cos':1,'sin':2,'alg':1,'alg-loga':2,'alg-logb':3,'alg-log':4}

    if weight in ['cos','sin']:
        integr = strdict[weight]
        if (b != np.inf and a != -np.inf):  # finite limits
            if wopts is None:         # no precomputed Chebyshev moments
                return _quadpack._qawoe(func, a, b, wvar, integr, args, full_output,
                                        epsabs, epsrel, limit, maxp1,1)
            else:                     # precomputed Chebyshev moments
                momcom = wopts[0]
                chebcom = wopts[1]
                return _quadpack._qawoe(func, a, b, wvar, integr, args,
                                        full_output,epsabs, epsrel, limit, maxp1, 2,
                                        momcom, chebcom)

        elif (b == np.inf and a != -np.inf):
            return _quadpack._qawfe(func, a, wvar, integr, args, full_output,
                                    epsabs, limlst, limit, maxp1)
        elif (b != np.inf and a == -np.inf):  # remap function and interval
            if weight == 'cos':
                def thefunc(x,*myargs):
                    y = -x
                    func = myargs[0]
                    myargs = (y,) + myargs[1:]
                    return func(*myargs)
            else:
                def thefunc(x,*myargs):
                    y = -x
                    func = myargs[0]
                    myargs = (y,) + myargs[1:]
                    return -func(*myargs)
            args = (func,) + args
            return _quadpack._qawfe(thefunc, -b, wvar, integr, args,
                                    full_output, epsabs, limlst, limit, maxp1)
        else:
            raise ValueError("Cannot integrate with this weight from -Inf to +Inf.")
    else:
        if a in [-np.inf, np.inf] or b in [-np.inf, np.inf]:
            message = "Cannot integrate with this weight over an infinite interval."
            raise ValueError(message)

        if weight.startswith('alg'):
            integr = strdict[weight]
            return _quadpack._qawse(func, a, b, wvar, integr, args,
                                    full_output, epsabs, epsrel, limit)
        else:  # weight == 'cauchy'
            return _quadpack._qawce(func, a, b, wvar, args, full_output,
                                    epsabs, epsrel, limit)


def dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8):
    """
    Compute a double integral.

    Return the double (definite) integral of ``func(y, x)`` from ``x = a..b``
    and ``y = gfun(x)..hfun(x)``.

    Parameters
    ----------
    func : callable
        A Python function or method of at least two variables: y must be the
        first argument and x the second argument.
    a, b : float
        The limits of integration in x: `a` < `b`
    gfun : callable or float
        The lower boundary curve in y which is a function taking a single
        floating point argument (x) and returning a floating point result
        or a float indicating a constant boundary curve.
    hfun : callable or float
        The upper boundary curve in y (same requirements as `gfun`).
    args : sequence, optional
        Extra arguments to pass to `func`.
    epsabs : float, optional
        Absolute tolerance passed directly to the inner 1-D quadrature
        integration. Default is 1.49e-8. ``dblquad`` tries to obtain
        an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
        where ``i`` = inner integral of ``func(y, x)`` from ``gfun(x)``
        to ``hfun(x)``, and ``result`` is the numerical approximation.
        See `epsrel` below.
    epsrel : float, optional
        Relative tolerance of the inner 1-D integrals. Default is 1.49e-8.
        If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
        and ``50 * (machine epsilon)``. See `epsabs` above.

    Returns
    -------
    y : float
        The resultant integral.
    abserr : float
        An estimate of the error.

    See Also
    --------
    quad : single integral
    tplquad : triple integral
    nquad : N-dimensional integrals
    fixed_quad : fixed-order Gaussian quadrature
    quadrature : adaptive Gaussian quadrature
    odeint : ODE integrator
    ode : ODE integrator
    simpson : integrator for sampled data
    romb : integrator for sampled data
    scipy.special : for coefficients and roots of orthogonal polynomials


    Notes
    -----
    For valid results, the integral must converge; behavior for divergent
    integrals is not guaranteed.

    **Details of QUADPACK level routines**

    `quad` calls routines from the FORTRAN library QUADPACK. This section
    provides details on the conditions for each routine to be called and a
    short description of each routine. For each level of integration, ``qagse``
    is used for finite limits or ``qagie`` is used if either limit (or both!)
    are infinite. The following provides a short description from [1]_ for each
    routine.

    qagse
        is an integrator based on globally adaptive interval
        subdivision in connection with extrapolation, which will
        eliminate the effects of integrand singularities of
        several types.
    qagie
        handles integration over infinite intervals. The infinite range is
        mapped onto a finite interval and subsequently the same strategy as
        in ``QAGS`` is applied.

    References
    ----------

    .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
           Überhuber, Christoph W.; Kahaner, David (1983).
           QUADPACK: A subroutine package for automatic integration.
           Springer-Verlag.
           ISBN 978-3-540-12553-2.

    Examples
    --------
    Compute the double integral of ``x * y**2`` over the box
    ``x`` ranging from 0 to 2 and ``y`` ranging from 0 to 1.
    That is, :math:`\\int^{x=2}_{x=0} \\int^{y=1}_{y=0} x y^2 \\,dy \\,dx`.

    >>> import numpy as np
    >>> from scipy import integrate
    >>> f = lambda y, x: x*y**2
    >>> integrate.dblquad(f, 0, 2, 0, 1)
        (0.6666666666666667, 7.401486830834377e-15)

    Calculate :math:`\\int^{x=\\pi/4}_{x=0} \\int^{y=\\cos(x)}_{y=\\sin(x)} 1
    \\,dy \\,dx`.

    >>> f = lambda y, x: 1
    >>> integrate.dblquad(f, 0, np.pi/4, np.sin, np.cos)
        (0.41421356237309503, 1.1083280054755938e-14)

    Calculate :math:`\\int^{x=1}_{x=0} \\int^{y=2-x}_{y=x} a x y \\,dy \\,dx`
    for :math:`a=1, 3`.

    >>> f = lambda y, x, a: a*x*y
    >>> integrate.dblquad(f, 0, 1, lambda x: x, lambda x: 2-x, args=(1,))
        (0.33333333333333337, 5.551115123125783e-15)
    >>> integrate.dblquad(f, 0, 1, lambda x: x, lambda x: 2-x, args=(3,))
        (0.9999999999999999, 1.6653345369377348e-14)

    Compute the two-dimensional Gaussian Integral, which is the integral of the
    Gaussian function :math:`f(x,y) = e^{-(x^{2} + y^{2})}`, over
    :math:`(-\\infty,+\\infty)`. That is, compute the integral
    :math:`\\iint^{+\\infty}_{-\\infty} e^{-(x^{2} + y^{2})} \\,dy\\,dx`.

    >>> f = lambda x, y: np.exp(-(x ** 2 + y ** 2))
    >>> integrate.dblquad(f, -np.inf, np.inf, -np.inf, np.inf)
        (3.141592653589777, 2.5173086737433208e-08)

    """

    def temp_ranges(*args):
        return [gfun(args[0]) if callable(gfun) else gfun,
                hfun(args[0]) if callable(hfun) else hfun]

    return nquad(func, [temp_ranges, [a, b]], args=args,
            opts={"epsabs": epsabs, "epsrel": epsrel})


def tplquad(func, a, b, gfun, hfun, qfun, rfun, args=(), epsabs=1.49e-8,
            epsrel=1.49e-8):
    """
    Compute a triple (definite) integral.

    Return the triple integral of ``func(z, y, x)`` from ``x = a..b``,
    ``y = gfun(x)..hfun(x)``, and ``z = qfun(x,y)..rfun(x,y)``.

    Parameters
    ----------
    func : function
        A Python function or method of at least three variables in the
        order (z, y, x).
    a, b : float
        The limits of integration in x: `a` < `b`
    gfun : function or float
        The lower boundary curve in y which is a function taking a single
        floating point argument (x) and returning a floating point result
        or a float indicating a constant boundary curve.
    hfun : function or float
        The upper boundary curve in y (same requirements as `gfun`).
    qfun : function or float
        The lower boundary surface in z.  It must be a function that takes
        two floats in the order (x, y) and returns a float or a float
        indicating a constant boundary surface.
    rfun : function or float
        The upper boundary surface in z. (Same requirements as `qfun`.)
    args : tuple, optional
        Extra arguments to pass to `func`.
    epsabs : float, optional
        Absolute tolerance passed directly to the innermost 1-D quadrature
        integration. Default is 1.49e-8.
    epsrel : float, optional
        Relative tolerance of the innermost 1-D integrals. Default is 1.49e-8.

    Returns
    -------
    y : float
        The resultant integral.
    abserr : float
        An estimate of the error.

    See Also
    --------
    quad : Adaptive quadrature using QUADPACK
    quadrature : Adaptive Gaussian quadrature
    fixed_quad : Fixed-order Gaussian quadrature
    dblquad : Double integrals
    nquad : N-dimensional integrals
    romb : Integrators for sampled data
    simpson : Integrators for sampled data
    ode : ODE integrators
    odeint : ODE integrators
    scipy.special : For coefficients and roots of orthogonal polynomials

    Notes
    -----
    For valid results, the integral must converge; behavior for divergent
    integrals is not guaranteed.

    **Details of QUADPACK level routines**

    `quad` calls routines from the FORTRAN library QUADPACK. This section
    provides details on the conditions for each routine to be called and a
    short description of each routine. For each level of integration, ``qagse``
    is used for finite limits or ``qagie`` is used, if either limit (or both!)
    are infinite. The following provides a short description from [1]_ for each
    routine.

    qagse
        is an integrator based on globally adaptive interval
        subdivision in connection with extrapolation, which will
        eliminate the effects of integrand singularities of
        several types.
    qagie
        handles integration over infinite intervals. The infinite range is
        mapped onto a finite interval and subsequently the same strategy as
        in ``QAGS`` is applied.

    References
    ----------

    .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
           Überhuber, Christoph W.; Kahaner, David (1983).
           QUADPACK: A subroutine package for automatic integration.
           Springer-Verlag.
           ISBN 978-3-540-12553-2.

    Examples
    --------
    Compute the triple integral of ``x * y * z``, over ``x`` ranging
    from 1 to 2, ``y`` ranging from 2 to 3, ``z`` ranging from 0 to 1.
    That is, :math:`\\int^{x=2}_{x=1} \\int^{y=3}_{y=2} \\int^{z=1}_{z=0} x y z
    \\,dz \\,dy \\,dx`.

    >>> import numpy as np
    >>> from scipy import integrate
    >>> f = lambda z, y, x: x*y*z
    >>> integrate.tplquad(f, 1, 2, 2, 3, 0, 1)
    (1.8749999999999998, 3.3246447942574074e-14)

    Calculate :math:`\\int^{x=1}_{x=0} \\int^{y=1-2x}_{y=0}
    \\int^{z=1-x-2y}_{z=0} x y z \\,dz \\,dy \\,dx`.
    Note: `qfun`/`rfun` takes arguments in the order (x, y), even though ``f``
    takes arguments in the order (z, y, x).

    >>> f = lambda z, y, x: x*y*z
    >>> integrate.tplquad(f, 0, 1, 0, lambda x: 1-2*x, 0, lambda x, y: 1-x-2*y)
    (0.05416666666666668, 2.1774196738157757e-14)

    Calculate :math:`\\int^{x=1}_{x=0} \\int^{y=1}_{y=0} \\int^{z=1}_{z=0}
    a x y z \\,dz \\,dy \\,dx` for :math:`a=1, 3`.

    >>> f = lambda z, y, x, a: a*x*y*z
    >>> integrate.tplquad(f, 0, 1, 0, 1, 0, 1, args=(1,))
        (0.125, 5.527033708952211e-15)
    >>> integrate.tplquad(f, 0, 1, 0, 1, 0, 1, args=(3,))
        (0.375, 1.6581101126856635e-14)

    Compute the three-dimensional Gaussian Integral, which is the integral of
    the Gaussian function :math:`f(x,y,z) = e^{-(x^{2} + y^{2} + z^{2})}`, over
    :math:`(-\\infty,+\\infty)`. That is, compute the integral
    :math:`\\iiint^{+\\infty}_{-\\infty} e^{-(x^{2} + y^{2} + z^{2})} \\,dz
    \\,dy\\,dx`.

    >>> f = lambda x, y, z: np.exp(-(x ** 2 + y ** 2 + z ** 2))
    >>> integrate.tplquad(f, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)
        (5.568327996830833, 4.4619078828029765e-08)

    """
    # f(z, y, x)
    # qfun/rfun(x, y)
    # gfun/hfun(x)
    # nquad will hand (y, x, t0, ...) to ranges0
    # nquad will hand (x, t0, ...) to ranges1
    # Only qfun / rfun is different API...

    def ranges0(*args):
        return [qfun(args[1], args[0]) if callable(qfun) else qfun,
                rfun(args[1], args[0]) if callable(rfun) else rfun]

    def ranges1(*args):
        return [gfun(args[0]) if callable(gfun) else gfun,
                hfun(args[0]) if callable(hfun) else hfun]

    ranges = [ranges0, ranges1, [a, b]]
    return nquad(func, ranges, args=args,
            opts={"epsabs": epsabs, "epsrel": epsrel})


def nquad(func, ranges, args=None, opts=None, full_output=False):
    r"""
    Integration over multiple variables.

    Wraps `quad` to enable integration over multiple variables.
    Various options allow improved integration of discontinuous functions, as
    well as the use of weighted integration, and generally finer control of the
    integration process.

    Parameters
    ----------
    func : {callable, scipy.LowLevelCallable}
        The function to be integrated. Has arguments of ``x0, ... xn``,
        ``t0, ... tm``, where integration is carried out over ``x0, ... xn``,
        which must be floats.  Where ``t0, ... tm`` are extra arguments
        passed in args.
        Function signature should be ``func(x0, x1, ..., xn, t0, t1, ..., tm)``.
        Integration is carried out in order.  That is, integration over ``x0``
        is the innermost integral, and ``xn`` is the outermost.

        If the user desires improved integration performance, then `f` may
        be a `scipy.LowLevelCallable` with one of the signatures::

            double func(int n, double *xx)
            double func(int n, double *xx, void *user_data)

        where ``n`` is the number of variables and args.  The ``xx`` array
        contains the coordinates and extra arguments. ``user_data`` is the data
        contained in the `scipy.LowLevelCallable`.
    ranges : iterable object
        Each element of ranges may be either a sequence  of 2 numbers, or else
        a callable that returns such a sequence. ``ranges[0]`` corresponds to
        integration over x0, and so on. If an element of ranges is a callable,
        then it will be called with all of the integration arguments available,
        as well as any parametric arguments. e.g., if
        ``func = f(x0, x1, x2, t0, t1)``, then ``ranges[0]`` may be defined as
        either ``(a, b)`` or else as ``(a, b) = range0(x1, x2, t0, t1)``.
    args : iterable object, optional
        Additional arguments ``t0, ... tn``, required by ``func``, ``ranges``,
        and ``opts``.
    opts : iterable object or dict, optional
        Options to be passed to `quad`. May be empty, a dict, or
        a sequence of dicts or functions that return a dict. If empty, the
        default options from scipy.integrate.quad are used. If a dict, the same
        options are used for all levels of integraion. If a sequence, then each
        element of the sequence corresponds to a particular integration. e.g.,
        ``opts[0]`` corresponds to integration over ``x0``, and so on. If a
        callable, the signature must be the same as for ``ranges``. The
        available options together with their default values are:

          - epsabs = 1.49e-08
          - epsrel = 1.49e-08
          - limit  = 50
          - points = None
          - weight = None
          - wvar   = None
          - wopts  = None

        For more information on these options, see `quad`.

    full_output : bool, optional
        Partial implementation of ``full_output`` from scipy.integrate.quad.
        The number of integrand function evaluations ``neval`` can be obtained
        by setting ``full_output=True`` when calling nquad.

    Returns
    -------
    result : float
        The result of the integration.
    abserr : float
        The maximum of the estimates of the absolute error in the various
        integration results.
    out_dict : dict, optional
        A dict containing additional information on the integration.

    See Also
    --------
    quad : 1-D numerical integration
    dblquad, tplquad : double and triple integrals
    fixed_quad : fixed-order Gaussian quadrature
    quadrature : adaptive Gaussian quadrature

    Notes
    -----
    For valid results, the integral must converge; behavior for divergent
    integrals is not guaranteed.

    **Details of QUADPACK level routines**

    `nquad` calls routines from the FORTRAN library QUADPACK. This section
    provides details on the conditions for each routine to be called and a
    short description of each routine. The routine called depends on
    `weight`, `points` and the integration limits `a` and `b`.

    ================  ==============  ==========  =====================
    QUADPACK routine  `weight`        `points`    infinite bounds
    ================  ==============  ==========  =====================
    qagse             None            No          No
    qagie             None            No          Yes
    qagpe             None            Yes         No
    qawoe             'sin', 'cos'    No          No
    qawfe             'sin', 'cos'    No          either `a` or `b`
    qawse             'alg*'          No          No
    qawce             'cauchy'        No          No
    ================  ==============  ==========  =====================

    The following provides a short description from [1]_ for each
    routine.

    qagse
        is an integrator based on globally adaptive interval
        subdivision in connection with extrapolation, which will
        eliminate the effects of integrand singularities of
        several types.
    qagie
        handles integration over infinite intervals. The infinite range is
        mapped onto a finite interval and subsequently the same strategy as
        in ``QAGS`` is applied.
    qagpe
        serves the same purposes as QAGS, but also allows the
        user to provide explicit information about the location
        and type of trouble-spots i.e. the abscissae of internal
        singularities, discontinuities and other difficulties of
        the integrand function.
    qawoe
        is an integrator for the evaluation of
        :math:`\int^b_a \cos(\omega x)f(x)dx` or
        :math:`\int^b_a \sin(\omega x)f(x)dx`
        over a finite interval [a,b], where :math:`\omega` and :math:`f`
        are specified by the user. The rule evaluation component is based
        on the modified Clenshaw-Curtis technique

        An adaptive subdivision scheme is used in connection
        with an extrapolation procedure, which is a modification
        of that in ``QAGS`` and allows the algorithm to deal with
        singularities in :math:`f(x)`.
    qawfe
        calculates the Fourier transform
        :math:`\int^\infty_a \cos(\omega x)f(x)dx` or
        :math:`\int^\infty_a \sin(\omega x)f(x)dx`
        for user-provided :math:`\omega` and :math:`f`. The procedure of
        ``QAWO`` is applied on successive finite intervals, and convergence
        acceleration by means of the :math:`\varepsilon`-algorithm is applied
        to the series of integral approximations.
    qawse
        approximate :math:`\int^b_a w(x)f(x)dx`, with :math:`a < b` where
        :math:`w(x) = (x-a)^{\alpha}(b-x)^{\beta}v(x)` with
        :math:`\alpha,\beta > -1`, where :math:`v(x)` may be one of the
        following functions: :math:`1`, :math:`\log(x-a)`, :math:`\log(b-x)`,
        :math:`\log(x-a)\log(b-x)`.

        The user specifies :math:`\alpha`, :math:`\beta` and the type of the
        function :math:`v`. A globally adaptive subdivision strategy is
        applied, with modified Clenshaw-Curtis integration on those
        subintervals which contain `a` or `b`.
    qawce
        compute :math:`\int^b_a f(x) / (x-c)dx` where the integral must be
        interpreted as a Cauchy principal value integral, for user specified
        :math:`c` and :math:`f`. The strategy is globally adaptive. Modified
        Clenshaw-Curtis integration is used on those intervals containing the
        point :math:`x = c`.

    References
    ----------

    .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
           Überhuber, Christoph W.; Kahaner, David (1983).
           QUADPACK: A subroutine package for automatic integration.
           Springer-Verlag.
           ISBN 978-3-540-12553-2.

    Examples
    --------
    Calculate

    .. math::

        \int^{1}_{-0.15} \int^{0.8}_{0.13} \int^{1}_{-1} \int^{1}_{0}
        f(x_0, x_1, x_2, x_3) \,dx_0 \,dx_1 \,dx_2 \,dx_3 ,

    where

    .. math::

        f(x_0, x_1, x_2, x_3) = \begin{cases}
          x_0^2+x_1 x_2-x_3^3+ \sin{x_0}+1 & (x_0-0.2 x_3-0.5-0.25 x_1 > 0) \\
          x_0^2+x_1 x_2-x_3^3+ \sin{x_0}+0 & (x_0-0.2 x_3-0.5-0.25 x_1 \leq 0)
        \end{cases} .

    >>> import numpy as np
    >>> from scipy import integrate
    >>> func = lambda x0,x1,x2,x3 : x0**2 + x1*x2 - x3**3 + np.sin(x0) + (
    ...                                 1 if (x0-.2*x3-.5-.25*x1>0) else 0)
    >>> def opts0(*args, **kwargs):
    ...     return {'points':[0.2*args[2] + 0.5 + 0.25*args[0]]}
    >>> integrate.nquad(func, [[0,1], [-1,1], [.13,.8], [-.15,1]],
    ...                 opts=[opts0,{},{},{}], full_output=True)
    (1.5267454070738633, 2.9437360001402324e-14, {'neval': 388962})

    Calculate

    .. math::

        \int^{t_0+t_1+1}_{t_0+t_1-1}
        \int^{x_2+t_0^2 t_1^3+1}_{x_2+t_0^2 t_1^3-1}
        \int^{t_0 x_1+t_1 x_2+1}_{t_0 x_1+t_1 x_2-1}
        f(x_0,x_1, x_2,t_0,t_1)
        \,dx_0 \,dx_1 \,dx_2,

    where

    .. math::

        f(x_0, x_1, x_2, t_0, t_1) = \begin{cases}
          x_0 x_2^2 + \sin{x_1}+2 & (x_0+t_1 x_1-t_0 > 0) \\
          x_0 x_2^2 +\sin{x_1}+1 & (x_0+t_1 x_1-t_0 \leq 0)
        \end{cases}

    and :math:`(t_0, t_1) = (0, 1)` .

    >>> def func2(x0, x1, x2, t0, t1):
    ...     return x0*x2**2 + np.sin(x1) + 1 + (1 if x0+t1*x1-t0>0 else 0)
    >>> def lim0(x1, x2, t0, t1):
    ...     return [t0*x1 + t1*x2 - 1, t0*x1 + t1*x2 + 1]
    >>> def lim1(x2, t0, t1):
    ...     return [x2 + t0**2*t1**3 - 1, x2 + t0**2*t1**3 + 1]
    >>> def lim2(t0, t1):
    ...     return [t0 + t1 - 1, t0 + t1 + 1]
    >>> def opts0(x1, x2, t0, t1):
    ...     return {'points' : [t0 - t1*x1]}
    >>> def opts1(x2, t0, t1):
    ...     return {}
    >>> def opts2(t0, t1):
    ...     return {}
    >>> integrate.nquad(func2, [lim0, lim1, lim2], args=(0,1),
    ...                 opts=[opts0, opts1, opts2])
    (36.099919226771625, 1.8546948553373528e-07)

    """
    depth = len(ranges)
    ranges = [rng if callable(rng) else _RangeFunc(rng) for rng in ranges]
    if args is None:
        args = ()
    if opts is None:
        opts = [dict([])] * depth

    if isinstance(opts, dict):
        opts = [_OptFunc(opts)] * depth
    else:
        opts = [opt if callable(opt) else _OptFunc(opt) for opt in opts]
    return _NQuad(func, ranges, opts, full_output).integrate(*args)


class _RangeFunc:
    def __init__(self, range_):
        self.range_ = range_

    def __call__(self, *args):
        """Return stored value.

        *args needed because range_ can be float or func, and is called with
        variable number of parameters.
        """
        return self.range_


class _OptFunc:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, *args):
        """Return stored dict."""
        return self.opt


class _NQuad:
    def __init__(self, func, ranges, opts, full_output):
        self.abserr = 0
        self.func = func
        self.ranges = ranges
        self.opts = opts
        self.maxdepth = len(ranges)
        self.full_output = full_output
        if self.full_output:
            self.out_dict = {'neval': 0}

    def integrate(self, *args, **kwargs):
        depth = kwargs.pop('depth', 0)
        if kwargs:
            raise ValueError('unexpected kwargs')

        # Get the integration range and options for this depth.
        ind = -(depth + 1)
        fn_range = self.ranges[ind]
        low, high = fn_range(*args)
        fn_opt = self.opts[ind]
        opt = dict(fn_opt(*args))

        if 'points' in opt:
            opt['points'] = [x for x in opt['points'] if low <= x <= high]
        if depth + 1 == self.maxdepth:
            f = self.func
        else:
            f = partial(self.integrate, depth=depth+1)
        quad_r = quad(f, low, high, args=args, full_output=self.full_output,
                      **opt)
        value = quad_r[0]
        abserr = quad_r[1]
        if self.full_output:
            infodict = quad_r[2]
            # The 'neval' parameter in full_output returns the total
            # number of times the integrand function was evaluated.
            # Therefore, only the innermost integration loop counts.
            if depth + 1 == self.maxdepth:
                self.out_dict['neval'] += infodict['neval']
        self.abserr = max(self.abserr, abserr)
        if depth > 0:
            return value
        else:
            # Final result of N-D integration with error
            if self.full_output:
                return value, self.abserr, self.out_dict
            else:
                return value, self.abserr
