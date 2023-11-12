# -*- coding: utf-8 -*-
"""Asymmetric kernels for R+ and unit interval

References
----------

.. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of
   Asymmetric Kernel Density Estimators and Smoothed Histograms with
   Application to Income Data.” Econometric Theory 21 (2): 390–412.

.. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”
   Computational Statistics & Data Analysis 31 (2): 131–45.
   https://doi.org/10.1016/S0167-9473(99)00010-9.

.. [3] Chen, Song Xi. 2000. “Probability Density Function Estimation Using
   Gamma Kernels.”
   Annals of the Institute of Statistical Mathematics 52 (3): 471–80.
   https://doi.org/10.1023/A:1004165218295.

.. [4] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and
   Lognormal Kernel Estimators for Modelling Durations in High Frequency
   Financial Data.” Annals of Economics and Finance 4: 103–24.

.. [5] Micheaux, Pierre Lafaye de, and Frédéric Ouimet. 2020. “A Study of Seven
   Asymmetric Kernels for the Estimation of Cumulative Distribution Functions,”
   November. https://arxiv.org/abs/2011.14893v1.

.. [6] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.
   “Asymmetric Kernels for Boundary Modification in Distribution Function
   Estimation.” REVSTAT, 1–27.

.. [7] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal
   Inverse Gaussian Kernels.”
   Journal of Nonparametric Statistics 16 (1–2): 217–26.
   https://doi.org/10.1080/10485250310001624819.


Created on Mon Mar  8 11:12:24 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import special, stats

doc_params = """\
Parameters
    ----------
    x : array_like, float
        Points for which density is evaluated. ``x`` can be scalar or 1-dim.
    sample : ndarray, 1-d
        Sample from which kde is computed.
    bw : float
        Bandwidth parameter, there is currently no default value for it.

    Returns
    -------
    Components for kernel estimation"""


def pdf_kernel_asym(x, sample, bw, kernel_type, weights=None, batch_size=10):
    """Density estimate based on asymmetric kernel.

    Parameters
    ----------
    x : array_like, float
        Points for which density is evaluated. ``x`` can be scalar or 1-dim.
    sample : ndarray, 1-d
        Sample from which kernel estimate is computed.
    bw : float
        Bandwidth parameter, there is currently no default value for it.
    kernel_type : str or callable
        Kernel name or kernel function.
        Currently supported kernel names are "beta", "beta2", "gamma",
        "gamma2", "bs", "invgamma", "invgauss", "lognorm", "recipinvgauss" and
        "weibull".
    weights : None or ndarray
        If weights is not None, then kernel for sample points are weighted
        by it. No weights corresponds to uniform weighting of each component
        with 1 / nobs, where nobs is the size of `sample`.
    batch_size : float
        If x is an 1-dim array, then points can be evaluated in vectorized
        form. To limit the amount of memory, a loop can work in batches.
        The number of batches is determined so that the intermediate array
        sizes are limited by

        ``np.size(batch) * len(sample) < batch_size * 1000``.

        Default is to have at most 10000 elements in intermediate arrays.

    Returns
    -------
    pdf : float or ndarray
        Estimate of pdf at points x. ``pdf`` has the same size or shape as x.
    """

    if callable(kernel_type):
        kfunc = kernel_type
    else:
        kfunc = kernel_dict_pdf[kernel_type]

    batch_size = batch_size * 1000

    if np.size(x) * len(sample) < batch_size:
        # no batch-loop
        if np.size(x) > 1:
            x = np.asarray(x)[:, None]

        pdfi = kfunc(x, sample, bw)
        if weights is None:
            pdf = pdfi.mean(-1)
        else:
            pdf = pdfi @ weights
    else:
        # batch, designed for 1-d x
        if weights is None:
            weights = np.ones(len(sample)) / len(sample)

        k = batch_size // len(sample)
        n = len(x) // k
        x_split = np.array_split(x, n)
        pdf = np.concatenate([(kfunc(xi[:, None], sample, bw) @ weights)
                              for xi in x_split])

    return pdf


def cdf_kernel_asym(x, sample, bw, kernel_type, weights=None, batch_size=10):
    """Estimate of cumulative distribution based on asymmetric kernel.

    Parameters
    ----------
    x : array_like, float
        Points for which density is evaluated. ``x`` can be scalar or 1-dim.
    sample : ndarray, 1-d
        Sample from which kernel estimate is computed.
    bw : float
        Bandwidth parameter, there is currently no default value for it.
    kernel_type : str or callable
        Kernel name or kernel function.
        Currently supported kernel names are "beta", "beta2", "gamma",
        "gamma2", "bs", "invgamma", "invgauss", "lognorm", "recipinvgauss" and
        "weibull".
    weights : None or ndarray
        If weights is not None, then kernel for sample points are weighted
        by it. No weights corresponds to uniform weighting of each component
        with 1 / nobs, where nobs is the size of `sample`.
    batch_size : float
        If x is an 1-dim array, then points can be evaluated in vectorized
        form. To limit the amount of memory, a loop can work in batches.
        The number of batches is determined so that the intermediate array
        sizes are limited by

        ``np.size(batch) * len(sample) < batch_size * 1000``.

        Default is to have at most 10000 elements in intermediate arrays.

    Returns
    -------
    cdf : float or ndarray
        Estimate of cdf at points x. ``cdf`` has the same size or shape as x.
    """

    if callable(kernel_type):
        kfunc = kernel_type
    else:
        kfunc = kernel_dict_cdf[kernel_type]

    batch_size = batch_size * 1000

    if np.size(x) * len(sample) < batch_size:
        # no batch-loop
        if np.size(x) > 1:
            x = np.asarray(x)[:, None]

        cdfi = kfunc(x, sample, bw)
        if weights is None:
            cdf = cdfi.mean(-1)
        else:
            cdf = cdfi @ weights
    else:
        # batch, designed for 1-d x
        if weights is None:
            weights = np.ones(len(sample)) / len(sample)

        k = batch_size // len(sample)
        n = len(x) // k
        x_split = np.array_split(x, n)
        cdf = np.concatenate([(kfunc(xi[:, None], sample, bw) @ weights)
                              for xi in x_split])

    return cdf


def kernel_pdf_beta(x, sample, bw):
    # Beta kernel for density, pdf, estimation
    return stats.beta.pdf(sample, x / bw + 1, (1 - x) / bw + 1)


kernel_pdf_beta.__doc__ = """\
    Beta kernel for density, pdf, estimation.

    {doc_params}

    References
    ----------
    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of
       Asymmetric Kernel Density Estimators and Smoothed Histograms with
       Application to Income Data.” Econometric Theory 21 (2): 390–412.

    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”
       Computational Statistics & Data Analysis 31 (2): 131–45.
       https://doi.org/10.1016/S0167-9473(99)00010-9.
    """.format(doc_params=doc_params)


def kernel_cdf_beta(x, sample, bw):
    # Beta kernel for cumulative distribution, cdf, estimation
    return stats.beta.sf(sample, x / bw + 1, (1 - x) / bw + 1)


kernel_cdf_beta.__doc__ = """\
    Beta kernel for cumulative distribution, cdf, estimation.

    {doc_params}

    References
    ----------
    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of
       Asymmetric Kernel Density Estimators and Smoothed Histograms with
       Application to Income Data.” Econometric Theory 21 (2): 390–412.

    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”
       Computational Statistics & Data Analysis 31 (2): 131–45.
       https://doi.org/10.1016/S0167-9473(99)00010-9.
    """.format(doc_params=doc_params)


def kernel_pdf_beta2(x, sample, bw):
    # Beta kernel for density, pdf, estimation with boundary corrections

    # a = 2 * bw**2 + 2.5 -
    #     np.sqrt(4 * bw**4 + 6 * bw**2 + 2.25 - x**2 - x / bw)
    # terms a1 and a2 are independent of x
    a1 = 2 * bw**2 + 2.5
    a2 = 4 * bw**4 + 6 * bw**2 + 2.25

    if np.size(x) == 1:
        # without vectorizing:
        if x < 2 * bw:
            a = a1 - np.sqrt(a2 - x**2 - x / bw)
            pdf = stats.beta.pdf(sample, a, (1 - x) / bw)
        elif x > (1 - 2 * bw):
            x_ = 1 - x
            a = a1 - np.sqrt(a2 - x_**2 - x_ / bw)
            pdf = stats.beta.pdf(sample, x / bw, a)
        else:
            pdf = stats.beta.pdf(sample, x / bw, (1 - x) / bw)
    else:
        alpha = x / bw
        beta = (1 - x) / bw

        mask_low = x < 2 * bw
        x_ = x[mask_low]
        alpha[mask_low] = a1 - np.sqrt(a2 - x_**2 - x_ / bw)

        mask_upp = x > (1 - 2 * bw)
        x_ = 1 - x[mask_upp]
        beta[mask_upp] = a1 - np.sqrt(a2 - x_**2 - x_ / bw)

        pdf = stats.beta.pdf(sample, alpha, beta)

    return pdf


kernel_pdf_beta2.__doc__ = """\
    Beta kernel for density, pdf, estimation with boundary corrections.

    {doc_params}

    References
    ----------
    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of
       Asymmetric Kernel Density Estimators and Smoothed Histograms with
       Application to Income Data.” Econometric Theory 21 (2): 390–412.

    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”
       Computational Statistics & Data Analysis 31 (2): 131–45.
       https://doi.org/10.1016/S0167-9473(99)00010-9.
    """.format(doc_params=doc_params)


def kernel_cdf_beta2(x, sample, bw):
    # Beta kernel for cdf estimation with boundary correction

    # a = 2 * bw**2 + 2.5 -
    #     np.sqrt(4 * bw**4 + 6 * bw**2 + 2.25 - x**2 - x / bw)
    # terms a1 and a2 are independent of x
    a1 = 2 * bw**2 + 2.5
    a2 = 4 * bw**4 + 6 * bw**2 + 2.25

    if np.size(x) == 1:
        # without vectorizing:
        if x < 2 * bw:
            a = a1 - np.sqrt(a2 - x**2 - x / bw)
            pdf = stats.beta.sf(sample, a, (1 - x) / bw)
        elif x > (1 - 2 * bw):
            x_ = 1 - x
            a = a1 - np.sqrt(a2 - x_**2 - x_ / bw)
            pdf = stats.beta.sf(sample, x / bw, a)
        else:
            pdf = stats.beta.sf(sample, x / bw, (1 - x) / bw)
    else:
        alpha = x / bw
        beta = (1 - x) / bw
        mask_low = x < 2 * bw

        x_ = x[mask_low]
        alpha[mask_low] = a1 - np.sqrt(a2 - x_**2 - x_ / bw)

        mask_upp = x > (1 - 2 * bw)
        x_ = 1 - x[mask_upp]
        beta[mask_upp] = a1 - np.sqrt(a2 - x_**2 - x_ / bw)

        pdf = stats.beta.sf(sample, alpha, beta)

    return pdf


kernel_cdf_beta2.__doc__ = """\
    Beta kernel for cdf estimation with boundary correction.

    {doc_params}

    References
    ----------
    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of
       Asymmetric Kernel Density Estimators and Smoothed Histograms with
       Application to Income Data.” Econometric Theory 21 (2): 390–412.

    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”
       Computational Statistics & Data Analysis 31 (2): 131–45.
       https://doi.org/10.1016/S0167-9473(99)00010-9.
    """.format(doc_params=doc_params)


def kernel_pdf_gamma(x, sample, bw):
    # Gamma kernel for density, pdf, estimation
    pdfi = stats.gamma.pdf(sample, x / bw + 1, scale=bw)
    return pdfi


kernel_pdf_gamma.__doc__ = """\
    Gamma kernel for density, pdf, estimation.

    {doc_params}

    References
    ----------
    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of
       Asymmetric Kernel Density Estimators and Smoothed Histograms with
       Application to Income Data.” Econometric Theory 21 (2): 390–412.

    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using
       Gamma Krnels.”
       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.
       https://doi.org/10.1023/A:1004165218295.
    """.format(doc_params=doc_params)


def kernel_cdf_gamma(x, sample, bw):
    # Gamma kernel for density, pdf, estimation
    # kernel cdf uses the survival function, but I don't know why.
    cdfi = stats.gamma.sf(sample, x / bw + 1, scale=bw)
    return cdfi


kernel_cdf_gamma.__doc__ = """\
    Gamma kernel for cumulative distribution, cdf, estimation.

    {doc_params}

    References
    ----------
    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of
       Asymmetric Kernel Density Estimators and Smoothed Histograms with
       Application to Income Data.” Econometric Theory 21 (2): 390–412.

    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using
       Gamma Krnels.”
       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.
       https://doi.org/10.1023/A:1004165218295.
    """.format(doc_params=doc_params)


def _kernel_pdf_gamma(x, sample, bw):
    """Gamma kernel for pdf, without boundary corrected part.

    drops `+ 1` in shape parameter

    It should be possible to use this if probability in
    neighborhood of zero boundary is small.

    """
    return stats.gamma.pdf(sample, x / bw, scale=bw)


def _kernel_cdf_gamma(x, sample, bw):
    """Gamma kernel for cdf, without boundary corrected part.

    drops `+ 1` in shape parameter

    It should be possible to use this if probability in
    neighborhood of zero boundary is small.

    """
    return stats.gamma.sf(sample, x / bw, scale=bw)


def kernel_pdf_gamma2(x, sample, bw):
    # Gamma kernel for density, pdf, estimation with boundary correction
    if np.size(x) == 1:
        # without vectorizing, easier to read
        if x < 2 * bw:
            a = (x / bw)**2 + 1
        else:
            a = x / bw
    else:
        a = x / bw
        mask = x < 2 * bw
        a[mask] = a[mask]**2 + 1
    pdf = stats.gamma.pdf(sample, a, scale=bw)

    return pdf


kernel_pdf_gamma2.__doc__ = """\
    Gamma kernel for density, pdf, estimation with boundary correction.

    {doc_params}

    References
    ----------
    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of
       Asymmetric Kernel Density Estimators and Smoothed Histograms with
       Application to Income Data.” Econometric Theory 21 (2): 390–412.

    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using
       Gamma Krnels.”
       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.
       https://doi.org/10.1023/A:1004165218295.
    """.format(doc_params=doc_params)


def kernel_cdf_gamma2(x, sample, bw):
    # Gamma kernel for cdf estimation with boundary correction
    if np.size(x) == 1:
        # without vectorizing
        if x < 2 * bw:
            a = (x / bw)**2 + 1
        else:
            a = x / bw
    else:
        a = x / bw
        mask = x < 2 * bw
        a[mask] = a[mask]**2 + 1
    pdf = stats.gamma.sf(sample, a, scale=bw)

    return pdf


kernel_cdf_gamma2.__doc__ = """\
    Gamma kernel for cdf estimation with boundary correction.

    {doc_params}

    References
    ----------
    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of
       Asymmetric Kernel Density Estimators and Smoothed Histograms with
       Application to Income Data.” Econometric Theory 21 (2): 390–412.

    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using
       Gamma Krnels.”
       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.
       https://doi.org/10.1023/A:1004165218295.
    """.format(doc_params=doc_params)


def kernel_pdf_invgamma(x, sample, bw):
    # Inverse gamma kernel for density, pdf, estimation
    return stats.invgamma.pdf(sample, 1 / bw + 1, scale=x / bw)


kernel_pdf_invgamma.__doc__ = """\
    Inverse gamma kernel for density, pdf, estimation.

    Based on cdf kernel by Micheaux and Ouimet (2020)

    {doc_params}

    References
    ----------
    .. [1] Micheaux, Pierre Lafaye de, and Frédéric Ouimet. 2020. “A Study of
       Seven Asymmetric Kernels for the Estimation of Cumulative Distribution
       Functions,” November. https://arxiv.org/abs/2011.14893v1.
    """.format(doc_params=doc_params)


def kernel_cdf_invgamma(x, sample, bw):
    # Inverse gamma kernel for cumulative distribution, cdf, estimation
    return stats.invgamma.sf(sample, 1 / bw + 1, scale=x / bw)


kernel_cdf_invgamma.__doc__ = """\
    Inverse gamma kernel for cumulative distribution, cdf, estimation.

    {doc_params}

    References
    ----------
    .. [1] Micheaux, Pierre Lafaye de, and Frédéric Ouimet. 2020. “A Study of
       Seven Asymmetric Kernels for the Estimation of Cumulative Distribution
       Functions,” November. https://arxiv.org/abs/2011.14893v1.
    """.format(doc_params=doc_params)


def kernel_pdf_invgauss(x, sample, bw):
    # Inverse gaussian kernel for density, pdf, estimation
    m = x
    lam = 1 / bw
    return stats.invgauss.pdf(sample, m / lam, scale=lam)


kernel_pdf_invgauss.__doc__ = """\
    Inverse gaussian kernel for density, pdf, estimation.

    {doc_params}

    References
    ----------
    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal
       Inverse Gaussian Kernels.”
       Journal of Nonparametric Statistics 16 (1–2): 217–26.
       https://doi.org/10.1080/10485250310001624819.
    """.format(doc_params=doc_params)


def kernel_pdf_invgauss_(x, sample, bw):
    """Inverse gaussian kernel density, explicit formula.

    Scaillet 2004
    """
    pdf = (1 / np.sqrt(2 * np.pi * bw * sample**3) *
           np.exp(- 1 / (2 * bw * x) * (sample / x - 2 + x / sample)))
    return pdf.mean(-1)


def kernel_cdf_invgauss(x, sample, bw):
    # Inverse gaussian kernel for cumulative distribution, cdf, estimation
    m = x
    lam = 1 / bw
    return stats.invgauss.sf(sample, m / lam, scale=lam)


kernel_cdf_invgauss.__doc__ = """\
    Inverse gaussian kernel for cumulative distribution, cdf, estimation.

    {doc_params}

    References
    ----------
    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal
       Inverse Gaussian Kernels.”
       Journal of Nonparametric Statistics 16 (1–2): 217–26.
       https://doi.org/10.1080/10485250310001624819.
    """.format(doc_params=doc_params)


def kernel_pdf_recipinvgauss(x, sample, bw):
    # Reciprocal inverse gaussian kernel for density, pdf, estimation

    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    m = 1 / (x - bw)
    lam = 1 / bw
    return stats.recipinvgauss.pdf(sample, m / lam, scale=1 / lam)


kernel_pdf_recipinvgauss.__doc__ = """\
    Reciprocal inverse gaussian kernel for density, pdf, estimation.

    {doc_params}

    References
    ----------
    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal
       Inverse Gaussian Kernels.”
       Journal of Nonparametric Statistics 16 (1–2): 217–26.
       https://doi.org/10.1080/10485250310001624819.
    """.format(doc_params=doc_params)


def kernel_pdf_recipinvgauss_(x, sample, bw):
    """Reciprocal inverse gaussian kernel density, explicit formula.

    Scaillet 2004
    """

    pdf = (1 / np.sqrt(2 * np.pi * bw * sample) *
           np.exp(- (x - bw) / (2 * bw) * sample / (x - bw) - 2 +
                  (x - bw) / sample))
    return pdf


def kernel_cdf_recipinvgauss(x, sample, bw):
    # Reciprocal inverse gaussian kernel for cdf estimation

    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    m = 1 / (x - bw)
    lam = 1 / bw
    return stats.recipinvgauss.sf(sample, m / lam, scale=1 / lam)


kernel_cdf_recipinvgauss.__doc__ = """\
    Reciprocal inverse gaussian kernel for cdf estimation.

    {doc_params}

    References
    ----------
    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal
       Inverse Gaussian Kernels.”
       Journal of Nonparametric Statistics 16 (1–2): 217–26.
       https://doi.org/10.1080/10485250310001624819.
    """.format(doc_params=doc_params)


def kernel_pdf_bs(x, sample, bw):
    # Birnbaum Saunders (normal) kernel for density, pdf, estimation
    return stats.fatiguelife.pdf(sample, bw, scale=x)


kernel_pdf_bs.__doc__ = """\
    Birnbaum Saunders (normal) kernel for density, pdf, estimation.

    {doc_params}

    References
    ----------
    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and
       Lognormal Kernel Estimators for Modelling Durations in High Frequency
       Financial Data.” Annals of Economics and Finance 4: 103–24.
    """.format(doc_params=doc_params)


def kernel_cdf_bs(x, sample, bw):
    # Birnbaum Saunders (normal) kernel for cdf estimation
    return stats.fatiguelife.sf(sample, bw, scale=x)


kernel_cdf_bs.__doc__ = """\
    Birnbaum Saunders (normal) kernel for cdf estimation.

    {doc_params}

    References
    ----------
    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and
       Lognormal Kernel Estimators for Modelling Durations in High Frequency
       Financial Data.” Annals of Economics and Finance 4: 103–24.
    .. [2] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.
       “Asymmetric Kernels for Boundary Modification in Distribution Function
       Estimation.” REVSTAT, 1–27.
    """.format(doc_params=doc_params)


def kernel_pdf_lognorm(x, sample, bw):
    # Log-normal kernel for density, pdf, estimation

    # need shape-scale parameterization for scipy
    # not sure why JK picked this normalization, makes required bw small
    # maybe we should skip this transformation and just use bw
    # Funke and Kawka 2015 (table 1) use bw (or bw**2) corresponding to
    #    variance of normal pdf
    # bw = np.exp(bw_**2 / 4) - 1  # this is inverse transformation
    bw_ = np.sqrt(4*np.log(1+bw))
    return stats.lognorm.pdf(sample, bw_, scale=x)


kernel_pdf_lognorm.__doc__ = """\
    Log-normal kernel for density, pdf, estimation.

    {doc_params}

    Notes
    -----
    Warning: parameterization of bandwidth will likely be changed

    References
    ----------
    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and
       Lognormal Kernel Estimators for Modelling Durations in High Frequency
       Financial Data.” Annals of Economics and Finance 4: 103–24.
    """.format(doc_params=doc_params)


def kernel_cdf_lognorm(x, sample, bw):
    # Log-normal kernel for cumulative distribution, cdf, estimation

    # need shape-scale parameterization for scipy
    # not sure why JK picked this normalization, makes required bw small
    # maybe we should skip this transformation and just use bw
    # Funke and Kawka 2015 (table 1) use bw (or bw**2) corresponding to
    #    variance of normal pdf
    # bw = np.exp(bw_**2 / 4) - 1  # this is inverse transformation
    bw_ = np.sqrt(4*np.log(1+bw))
    return stats.lognorm.sf(sample, bw_, scale=x)


kernel_cdf_lognorm.__doc__ = """\
    Log-normal kernel for cumulative distribution, cdf, estimation.

    {doc_params}

    Notes
    -----
    Warning: parameterization of bandwidth will likely be changed

    References
    ----------
    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and
       Lognormal Kernel Estimators for Modelling Durations in High Frequency
       Financial Data.” Annals of Economics and Finance 4: 103–24.
    """.format(doc_params=doc_params)


def kernel_pdf_lognorm_(x, sample, bw):
    """Log-normal kernel for density, pdf, estimation, explicit formula.

    Jin, Kawczak 2003
    """
    term = 8 * np.log(1 + bw)  # this is 2 * variance in normal pdf
    pdf = (1 / np.sqrt(term * np.pi) / sample *
           np.exp(- (np.log(x) - np.log(sample))**2 / term))
    return pdf.mean(-1)


def kernel_pdf_weibull(x, sample, bw):
    # Weibull kernel for density, pdf, estimation

    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    return stats.weibull_min.pdf(sample, 1 / bw,
                                 scale=x / special.gamma(1 + bw))


kernel_pdf_weibull.__doc__ = """\
    Weibull kernel for density, pdf, estimation.

    Based on cdf kernel by Mombeni et al. (2019)

    {doc_params}

    References
    ----------
    .. [1] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.
       “Asymmetric Kernels for Boundary Modification in Distribution Function
       Estimation.” REVSTAT, 1–27.
    """.format(doc_params=doc_params)


def kernel_cdf_weibull(x, sample, bw):
    # Weibull kernel for cumulative distribution, cdf, estimation

    # need shape-scale parameterization for scipy
    # references use m, lambda parameterization
    return stats.weibull_min.sf(sample, 1 / bw,
                                scale=x / special.gamma(1 + bw))


kernel_cdf_weibull.__doc__ = """\
    Weibull kernel for cumulative distribution, cdf, estimation.

    {doc_params}

    References
    ----------
    .. [1] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.
       “Asymmetric Kernels for Boundary Modification in Distribution Function
       Estimation.” REVSTAT, 1–27.
    """.format(doc_params=doc_params)


# produced wth
# print("\n".join(['"%s": %s,' % (i.split("_")[-1], i) for i in dir(kern)
#                  if "kernel" in i and not i.endswith("_")]))
kernel_dict_cdf = {
    "beta": kernel_cdf_beta,
    "beta2": kernel_cdf_beta2,
    "bs": kernel_cdf_bs,
    "gamma": kernel_cdf_gamma,
    "gamma2": kernel_cdf_gamma2,
    "invgamma": kernel_cdf_invgamma,
    "invgauss": kernel_cdf_invgauss,
    "lognorm": kernel_cdf_lognorm,
    "recipinvgauss": kernel_cdf_recipinvgauss,
    "weibull": kernel_cdf_weibull,
    }

kernel_dict_pdf = {
    "beta": kernel_pdf_beta,
    "beta2": kernel_pdf_beta2,
    "bs": kernel_pdf_bs,
    "gamma": kernel_pdf_gamma,
    "gamma2": kernel_pdf_gamma2,
    "invgamma": kernel_pdf_invgamma,
    "invgauss": kernel_pdf_invgauss,
    "lognorm": kernel_pdf_lognorm,
    "recipinvgauss": kernel_pdf_recipinvgauss,
    "weibull": kernel_pdf_weibull,
    }
