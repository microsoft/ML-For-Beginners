"""
57 consecutive daily overshorts from an underground gasoline tank at a filling
station in Colorado

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

oshorts = pd.Series([
    78, -58, 53, -65, 13, -6, -16, -14, 3, -72, 89, -48, -14, 32, 56, -86,
    -66, 50, 26, 59, -47, -83, 2, -1, 124, -106, 113, -76, -47, -32, 39,
    -30, 6, -73, 18, 2, -24, 23, -38, 91, -56, -58, 1, 14, -4, 77, -127, 97,
    10, -28, -17, 23, -2, 48, -131, 65, -17])
