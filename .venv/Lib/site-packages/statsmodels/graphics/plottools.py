import numpy as np


def rainbow(n):
    """
    Returns a list of colors sampled at equal intervals over the spectrum.

    Parameters
    ----------
    n : int
        The number of colors to return

    Returns
    -------
    R : (n,3) array
        An of rows of RGB color values

    Notes
    -----
    Converts from HSV coordinates (0, 1, 1) to (1, 1, 1) to RGB. Based on
    the Sage function of the same name.
    """
    from matplotlib import colors
    R = np.ones((1,n,3))
    R[0,:,0] = np.linspace(0, 1, n, endpoint=False)
    #Note: could iterate and use colorsys.hsv_to_rgb
    return colors.hsv_to_rgb(R).squeeze()
