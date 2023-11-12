"""
The number of car drivers killed or seriously injured monthly in Great Britain
for ten years beginning in January 1975

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

sbl = pd.Series([
    1577, 1356, 1652, 1382, 1519, 1421, 1442, 1543, 1656, 1561, 1905, 2199,
    1473, 1655, 1407, 1395, 1530, 1309, 1526, 1327, 1627, 1748, 1958, 2274,
    1648, 1401, 1411, 1403, 1394, 1520, 1528, 1643, 1515, 1685, 2000, 2215,
    1956, 1462, 1563, 1459, 1446, 1622, 1657, 1638, 1643, 1683, 2050, 2262,
    1813, 1445, 1762, 1461, 1556, 1431, 1427, 1554, 1645, 1653, 2016, 2207,
    1665, 1361, 1506, 1360, 1453, 1522, 1460, 1552, 1548, 1827, 1737, 1941,
    1474, 1458, 1542, 1404, 1522, 1385, 1641, 1510, 1681, 1938, 1868, 1726,
    1456, 1445, 1456, 1365, 1487, 1558, 1488, 1684, 1594, 1850, 1998, 2079,
    1494, 1057, 1218, 1168, 1236, 1076, 1174, 1139, 1427, 1487, 1483, 1513,
    1357, 1165, 1282, 1110, 1297, 1185, 1222, 1284, 1444, 1575, 1737, 1763],
    index=pd.date_range(start='1975-01-01', end='1984-12-01', freq='MS'))
