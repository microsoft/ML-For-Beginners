"""
Dow-Jones Utilities Index, Aug.28--Dec.18, 1972.

Dataset described in [1]_ and included as a part of the ITSM2000 software [2]_.
Downloaded on April 22, 2019 from:
http://www.springer.com/cda/content/document/cda_downloaddocument/ITSM2000.zip

See also https://finance.yahoo.com/quote/%5EDJU/history?period1=83822400&period2=93502800&interval=1d&filter=history&frequency=1d

TODO: Add the correct business days index for this data (freq='B' does not work)

References
----------
.. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
   Introduction to Time Series and Forecasting. Springer.
.. [2] Brockwell, Peter J., and Richard A. Davis. n.d. ITSM2000.
"""  # noqa:E501

import pandas as pd

dowj = pd.Series([
    110.94, 110.69, 110.43, 110.56, 110.75, 110.84, 110.46, 110.56, 110.46,
    110.05, 109.6, 109.31, 109.31, 109.25, 109.02, 108.54, 108.77, 109.02,
    109.44, 109.38, 109.53, 109.89, 110.56, 110.56, 110.72, 111.23, 111.48,
    111.58, 111.9, 112.19, 112.06, 111.96, 111.68, 111.36, 111.42, 112,
    112.22, 112.7, 113.15, 114.36, 114.65, 115.06, 115.86, 116.4, 116.44,
    116.88, 118.07, 118.51, 119.28, 119.79, 119.7, 119.28, 119.66, 120.14,
    120.97, 121.13, 121.55, 121.96, 122.26, 123.79, 124.11, 124.14, 123.37,
    123.02, 122.86, 123.02, 123.11, 123.05, 123.05, 122.83, 123.18, 122.67,
    122.73, 122.86, 122.67, 122.09, 122, 121.23])
