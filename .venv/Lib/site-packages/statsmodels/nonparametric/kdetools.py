#### Convenience Functions to be moved to kerneltools ####
import numpy as np

def forrt(X, m=None):
    """
    RFFT with order like Munro (1976) FORTT routine.
    """
    if m is None:
        m = len(X)
    y = np.fft.rfft(X, m) / m
    return np.r_[y.real, y[1:-1].imag]

def revrt(X, m=None):
    """
    Inverse of forrt. Equivalent to Munro (1976) REVRT routine.
    """
    if m is None:
        m = len(X)
    i = int(m // 2 + 1)
    y = X[:i] + np.r_[0, X[i:], 0] * 1j
    return np.fft.irfft(y)*m

def silverman_transform(bw, M, RANGE):
    """
    FFT of Gaussian kernel following to Silverman AS 176.

    Notes
    -----
    Underflow is intentional as a dampener.
    """
    J = np.arange(M/2+1)
    FAC1 = 2*(np.pi*bw/RANGE)**2
    JFAC = J**2*FAC1
    BC = 1 - 1. / 3 * (J * 1./M*np.pi)**2
    FAC = np.exp(-JFAC)/BC
    kern_est = np.r_[FAC, FAC[1:-1]]
    return kern_est

def counts(x, v):
    """
    Counts the number of elements of x that fall within the grid points v

    Notes
    -----
    Using np.digitize and np.bincount
    """
    idx = np.digitize(x, v)
    try: # numpy 1.6
        return np.bincount(idx, minlength=len(v))
    except:
        bc = np.bincount(idx)
        return np.r_[bc, np.zeros(len(v) - len(bc))]

def kdesum(x, axis=0):
    return np.asarray([np.sum(x[i] - x, axis) for i in range(len(x))])
