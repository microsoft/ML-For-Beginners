import numpy as np

LOWER_BOUND = np.sqrt(np.finfo(float).eps)


class HoltWintersArgs:
    def __init__(self, xi, p, bounds, y, m, n, transform=False):
        self._xi = xi
        self._p = p
        self._bounds = bounds
        self._y = y
        self._lvl = np.empty(n)
        self._b = np.empty(n)
        self._s = np.empty(n + m - 1)
        self._m = m
        self._n = n
        self._transform = transform

    @property
    def xi(self):
        return self._xi

    @xi.setter
    def xi(self, value):
        self._xi = value

    @property
    def p(self):
        return self._p

    @property
    def bounds(self):
        return self._bounds

    @property
    def y(self):
        return self._y

    @property
    def lvl(self):
        return self._lvl

    @property
    def b(self):
        return self._b

    @property
    def s(self):
        return self._s

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value


def to_restricted(p, sel, bounds):
    """
    Transform parameters from the unrestricted [0,1] space
    to satisfy both the bounds and the 2 constraints
    beta <= alpha and gamma <= (1-alpha)

    Parameters
    ----------
    p : ndarray
        The parameters to transform
    sel : ndarray
        Array indicating whether a parameter is being estimated. If not
        estimated, not transformed.
    bounds : ndarray
        2-d array of bounds where bound for element i is in row i
        and stored as [lb, ub]

    Returns
    -------

    """
    a, b, g = p[:3]

    if sel[0]:
        lb = max(LOWER_BOUND, bounds[0, 0])
        ub = min(1 - LOWER_BOUND, bounds[0, 1])
        a = lb + a * (ub - lb)
    if sel[1]:
        lb = bounds[1, 0]
        ub = min(a, bounds[1, 1])
        b = lb + b * (ub - lb)
    if sel[2]:
        lb = bounds[2, 0]
        ub = min(1.0 - a, bounds[2, 1])
        g = lb + g * (ub - lb)

    return a, b, g


def to_unrestricted(p, sel, bounds):
    """
    Transform parameters to the unrestricted [0,1] space

    Parameters
    ----------
    p : ndarray
        Parameters that strictly satisfy the constraints

    Returns
    -------
    ndarray
        Parameters all in (0,1)
    """
    # eps < a < 1 - eps
    # eps < b <= a
    # eps < g <= 1 - a

    a, b, g = p[:3]

    if sel[0]:
        lb = max(LOWER_BOUND, bounds[0, 0])
        ub = min(1 - LOWER_BOUND, bounds[0, 1])
        a = (a - lb) / (ub - lb)
    if sel[1]:
        lb = bounds[1, 0]
        ub = min(p[0], bounds[1, 1])
        b = (b - lb) / (ub - lb)
    if sel[2]:
        lb = bounds[2, 0]
        ub = min(1.0 - p[0], bounds[2, 1])
        g = (g - lb) / (ub - lb)

    return a, b, g


def holt_init(x, hw_args: HoltWintersArgs):
    """
    Initialization for the Holt Models
    """
    # Map back to the full set of parameters
    hw_args.p[hw_args.xi.astype(bool)] = x

    # Ensure alpha and beta satisfy the requirements
    if hw_args.transform:
        alpha, beta, _ = to_restricted(hw_args.p, hw_args.xi, hw_args.bounds)
    else:
        alpha, beta = hw_args.p[:2]
    # Level, trend and dampening
    l0, b0, phi = hw_args.p[3:6]
    # Save repeated calculations
    alphac = 1 - alpha
    betac = 1 - beta
    # Setup alpha * y
    y_alpha = alpha * hw_args.y
    # In-place operations
    hw_args.lvl[0] = l0
    hw_args.b[0] = b0

    return alpha, beta, phi, alphac, betac, y_alpha


def holt__(x, hw_args: HoltWintersArgs):
    """
    Simple Exponential Smoothing
    Minimization Function
    (,)
    """
    _, _, _, alphac, _, y_alpha = holt_init(x, hw_args)
    n = hw_args.n
    lvl = hw_args.lvl
    for i in range(1, n):
        lvl[i] = (y_alpha[i - 1]) + (alphac * (lvl[i - 1]))
    return hw_args.y - lvl


def holt_mul_dam(x, hw_args: HoltWintersArgs):
    """
    Multiplicative and Multiplicative Damped
    Minimization Function
    (M,) & (Md,)
    """
    _, beta, phi, alphac, betac, y_alpha = holt_init(x, hw_args)
    lvl = hw_args.lvl
    b = hw_args.b
    for i in range(1, hw_args.n):
        lvl[i] = (y_alpha[i - 1]) + (alphac * (lvl[i - 1] * b[i - 1] ** phi))
        b[i] = (beta * (lvl[i] / lvl[i - 1])) + (betac * b[i - 1] ** phi)
    return hw_args.y - lvl * b**phi


def holt_add_dam(x, hw_args: HoltWintersArgs):
    """
    Additive and Additive Damped
    Minimization Function
    (A,) & (Ad,)
    """
    _, beta, phi, alphac, betac, y_alpha = holt_init(x, hw_args)
    lvl = hw_args.lvl
    b = hw_args.b
    for i in range(1, hw_args.n):
        lvl[i] = (y_alpha[i - 1]) + (alphac * (lvl[i - 1] + phi * b[i - 1]))
        b[i] = (beta * (lvl[i] - lvl[i - 1])) + (betac * phi * b[i - 1])
    return hw_args.y - (lvl + phi * b)


def holt_win_init(x, hw_args: HoltWintersArgs):
    """Initialization for the Holt Winters Seasonal Models"""
    hw_args.p[hw_args.xi.astype(bool)] = x
    if hw_args.transform:
        alpha, beta, gamma = to_restricted(
            hw_args.p, hw_args.xi, hw_args.bounds
        )
    else:
        alpha, beta, gamma = hw_args.p[:3]

    l0, b0, phi = hw_args.p[3:6]
    s0 = hw_args.p[6:]
    alphac = 1 - alpha
    betac = 1 - beta
    gammac = 1 - gamma
    y_alpha = alpha * hw_args.y
    y_gamma = gamma * hw_args.y
    hw_args.lvl[:] = 0
    hw_args.b[:] = 0
    hw_args.s[:] = 0
    hw_args.lvl[0] = l0
    hw_args.b[0] = b0
    hw_args.s[: hw_args.m] = s0
    return alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma


def holt_win__mul(x, hw_args: HoltWintersArgs):
    """
    Multiplicative Seasonal
    Minimization Function
    (,M)
    """
    (_, _, _, _, alphac, _, gammac, y_alpha, y_gamma) = holt_win_init(
        x, hw_args
    )
    lvl = hw_args.lvl
    s = hw_args.s
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = (y_alpha[i - 1] / s[i - 1]) + (alphac * (lvl[i - 1]))
        s[i + m - 1] = (y_gamma[i - 1] / (lvl[i - 1])) + (gammac * s[i - 1])
    return hw_args.y - lvl * s[: -(m - 1)]


def holt_win__add(x, hw_args: HoltWintersArgs):
    """
    Additive Seasonal
    Minimization Function
    (,A)
    """
    (alpha, _, gamma, _, alphac, _, gammac, y_alpha, y_gamma) = holt_win_init(
        x, hw_args
    )
    lvl = hw_args.lvl
    s = hw_args.s
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = (
            (y_alpha[i - 1]) - (alpha * s[i - 1]) + (alphac * (lvl[i - 1]))
        )
        s[i + m - 1] = (
            y_gamma[i - 1] - (gamma * (lvl[i - 1])) + (gammac * s[i - 1])
        )
    return hw_args.y - lvl - s[: -(m - 1)]


def holt_win_add_mul_dam(x, hw_args: HoltWintersArgs):
    """
    Additive and Additive Damped with Multiplicative Seasonal
    Minimization Function
    (A,M) & (Ad,M)
    """
    (
        _,
        beta,
        _,
        phi,
        alphac,
        betac,
        gammac,
        y_alpha,
        y_gamma,
    ) = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    b = hw_args.b
    s = hw_args.s
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = (y_alpha[i - 1] / s[i - 1]) + (
            alphac * (lvl[i - 1] + phi * b[i - 1])
        )
        b[i] = (beta * (lvl[i] - lvl[i - 1])) + (betac * phi * b[i - 1])
        s[i + m - 1] = (y_gamma[i - 1] / (lvl[i - 1] + phi * b[i - 1])) + (
            gammac * s[i - 1]
        )
    return hw_args.y - (lvl + phi * b) * s[: -(m - 1)]


def holt_win_mul_mul_dam(x, hw_args: HoltWintersArgs):
    """
    Multiplicative and Multiplicative Damped with Multiplicative Seasonal
    Minimization Function
    (M,M) & (Md,M)
    """
    (
        _,
        beta,
        _,
        phi,
        alphac,
        betac,
        gammac,
        y_alpha,
        y_gamma,
    ) = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    s = hw_args.s
    b = hw_args.b
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = (y_alpha[i - 1] / s[i - 1]) + (
            alphac * (lvl[i - 1] * b[i - 1] ** phi)
        )
        b[i] = (beta * (lvl[i] / lvl[i - 1])) + (betac * b[i - 1] ** phi)
        s[i + m - 1] = (y_gamma[i - 1] / (lvl[i - 1] * b[i - 1] ** phi)) + (
            gammac * s[i - 1]
        )
    return hw_args.y - (lvl * b**phi) * s[: -(m - 1)]


def holt_win_add_add_dam(x, hw_args: HoltWintersArgs):
    """
    Additive and Additive Damped with Additive Seasonal
    Minimization Function
    (A,A) & (Ad,A)
    """
    (
        alpha,
        beta,
        gamma,
        phi,
        alphac,
        betac,
        gammac,
        y_alpha,
        y_gamma,
    ) = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    s = hw_args.s
    b = hw_args.b
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = (
            (y_alpha[i - 1])
            - (alpha * s[i - 1])
            + (alphac * (lvl[i - 1] + phi * b[i - 1]))
        )
        b[i] = (beta * (lvl[i] - lvl[i - 1])) + (betac * phi * b[i - 1])
        s[i + m - 1] = (
            y_gamma[i - 1]
            - (gamma * (lvl[i - 1] + phi * b[i - 1]))
            + (gammac * s[i - 1])
        )
    return hw_args.y - ((lvl + phi * b) + s[: -(m - 1)])


def holt_win_mul_add_dam(x, hw_args: HoltWintersArgs):
    """
    Multiplicative and Multiplicative Damped with Additive Seasonal
    Minimization Function
    (M,A) & (M,Ad)
    """
    (
        alpha,
        beta,
        gamma,
        phi,
        alphac,
        betac,
        gammac,
        y_alpha,
        y_gamma,
    ) = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    s = hw_args.s
    b = hw_args.b
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = (
            (y_alpha[i - 1])
            - (alpha * s[i - 1])
            + (alphac * (lvl[i - 1] * b[i - 1] ** phi))
        )
        b[i] = (beta * (lvl[i] / lvl[i - 1])) + (betac * b[i - 1] ** phi)
        s[i + m - 1] = (
            y_gamma[i - 1]
            - (gamma * (lvl[i - 1] * b[i - 1] ** phi))
            + (gammac * s[i - 1])
        )
    return hw_args.y - ((lvl * phi * b) + s[: -(m - 1)])
