""" support pyarrow compatibility across versions """

from __future__ import annotations

from pandas.util.version import Version

try:
    import pyarrow as pa

    _palv = Version(Version(pa.__version__).base_version)
    pa_version_under7p0 = _palv < Version("7.0.0")
    pa_version_under8p0 = _palv < Version("8.0.0")
    pa_version_under9p0 = _palv < Version("9.0.0")
    pa_version_under10p0 = _palv < Version("10.0.0")
    pa_version_under11p0 = _palv < Version("11.0.0")
    pa_version_under12p0 = _palv < Version("12.0.0")
    pa_version_under13p0 = _palv < Version("13.0.0")
except ImportError:
    pa_version_under7p0 = True
    pa_version_under8p0 = True
    pa_version_under9p0 = True
    pa_version_under10p0 = True
    pa_version_under11p0 = True
    pa_version_under12p0 = True
    pa_version_under13p0 = True
