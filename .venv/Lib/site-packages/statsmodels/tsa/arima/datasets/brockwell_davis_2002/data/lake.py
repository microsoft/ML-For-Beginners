"""
Lake level of Lake Huron in feet (reduced by 570), 1875--1972.

Dataset described in [1]_ and included as a part of the ITSM2000 software [2]_.
Downloaded on April 22, 2019 from:
http://www.springer.com/cda/content/document/cda_downloaddocument/ITSM2000.zip

References
----------
.. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
   Introduction to Time Series and Forecasting. Springer.
.. [2] Brockwell, Peter J., and Richard A. Davis. n.d. ITSM2000.
"""

import pandas as pd

lake = pd.Series([
    10.38, 11.86, 10.97, 10.8, 9.79, 10.39, 10.42, 10.82, 11.4, 11.32, 11.44,
    11.68, 11.17, 10.53, 10.01, 9.91, 9.14, 9.16, 9.55, 9.67, 8.44, 8.24, 9.1,
    9.09, 9.35, 8.82, 9.32, 9.01, 9, 9.8, 9.83, 9.72, 9.89, 10.01, 9.37, 8.69,
    8.19, 8.67, 9.55, 8.92, 8.09, 9.37, 10.13, 10.14, 9.51, 9.24, 8.66, 8.86,
    8.05, 7.79, 6.75, 6.75, 7.82, 8.64, 10.58, 9.48, 7.38, 6.9, 6.94, 6.24,
    6.84, 6.85, 6.9, 7.79, 8.18, 7.51, 7.23, 8.42, 9.61, 9.05, 9.26, 9.22,
    9.38, 9.1, 7.95, 8.12, 9.75, 10.85, 10.41, 9.96, 9.61, 8.76, 8.18, 7.21,
    7.13, 9.1, 8.25, 7.91, 6.89, 5.96, 6.8, 7.68, 8.38, 8.52, 9.74, 9.31,
    9.89, 9.96],
    index=pd.period_range(start='1875', end='1972', freq='A').to_timestamp())
