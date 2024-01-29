"""
This test script is adopted from:
    https://github.com/numpy/numpy/blob/main/numpy/tests/test_public_api.py
"""

import pkgutil
import types
import importlib
import warnings
from importlib import import_module

import pytest

import scipy


def test_dir_testing():
    """Assert that output of dir has only one "testing/tester"
    attribute without duplicate"""
    assert len(dir(scipy)) == len(set(dir(scipy)))


# Historically SciPy has not used leading underscores for private submodules
# much.  This has resulted in lots of things that look like public modules
# (i.e. things that can be imported as `import scipy.somesubmodule.somefile`),
# but were never intended to be public.  The PUBLIC_MODULES list contains
# modules that are either public because they were meant to be, or because they
# contain public functions/objects that aren't present in any other namespace
# for whatever reason and therefore should be treated as public.
PUBLIC_MODULES = ["scipy." + s for s in [
    "cluster",
    "cluster.vq",
    "cluster.hierarchy",
    "constants",
    "datasets",
    "fft",
    "fftpack",
    "integrate",
    "interpolate",
    "io",
    "io.arff",
    "io.matlab",
    "io.wavfile",
    "linalg",
    "linalg.blas",
    "linalg.cython_blas",
    "linalg.lapack",
    "linalg.cython_lapack",
    "linalg.interpolative",
    "misc",
    "ndimage",
    "odr",
    "optimize",
    "signal",
    "signal.windows",
    "sparse",
    "sparse.linalg",
    "sparse.csgraph",
    "spatial",
    "spatial.distance",
    "spatial.transform",
    "special",
    "stats",
    "stats.contingency",
    "stats.distributions",
    "stats.mstats",
    "stats.qmc",
    "stats.sampling"
]]

# The PRIVATE_BUT_PRESENT_MODULES list contains modules that lacked underscores
# in their name and hence looked public, but weren't meant to be. All these
# namespace were deprecated in the 1.8.0 release - see "clear split between
# public and private API" in the 1.8.0 release notes.
# These private modules support will be removed in SciPy v2.0.0, as the
# deprecation messages emitted by each of these modules say.
PRIVATE_BUT_PRESENT_MODULES = [
    'scipy.constants.codata',
    'scipy.constants.constants',
    'scipy.fftpack.basic',
    'scipy.fftpack.convolve',
    'scipy.fftpack.helper',
    'scipy.fftpack.pseudo_diffs',
    'scipy.fftpack.realtransforms',
    'scipy.integrate.dop',
    'scipy.integrate.lsoda',
    'scipy.integrate.odepack',
    'scipy.integrate.quadpack',
    'scipy.integrate.vode',
    'scipy.interpolate.dfitpack',
    'scipy.interpolate.fitpack',
    'scipy.interpolate.fitpack2',
    'scipy.interpolate.interpnd',
    'scipy.interpolate.interpolate',
    'scipy.interpolate.ndgriddata',
    'scipy.interpolate.polyint',
    'scipy.interpolate.rbf',
    'scipy.io.arff.arffread',
    'scipy.io.harwell_boeing',
    'scipy.io.idl',
    'scipy.io.matlab.byteordercodes',
    'scipy.io.matlab.mio',
    'scipy.io.matlab.mio4',
    'scipy.io.matlab.mio5',
    'scipy.io.matlab.mio5_params',
    'scipy.io.matlab.mio5_utils',
    'scipy.io.matlab.mio_utils',
    'scipy.io.matlab.miobase',
    'scipy.io.matlab.streams',
    'scipy.io.mmio',
    'scipy.io.netcdf',
    'scipy.linalg.basic',
    'scipy.linalg.decomp',
    'scipy.linalg.decomp_cholesky',
    'scipy.linalg.decomp_lu',
    'scipy.linalg.decomp_qr',
    'scipy.linalg.decomp_schur',
    'scipy.linalg.decomp_svd',
    'scipy.linalg.flinalg',
    'scipy.linalg.matfuncs',
    'scipy.linalg.misc',
    'scipy.linalg.special_matrices',
    'scipy.misc.common',
    'scipy.misc.doccer',
    'scipy.ndimage.filters',
    'scipy.ndimage.fourier',
    'scipy.ndimage.interpolation',
    'scipy.ndimage.measurements',
    'scipy.ndimage.morphology',
    'scipy.odr.models',
    'scipy.odr.odrpack',
    'scipy.optimize.cobyla',
    'scipy.optimize.cython_optimize',
    'scipy.optimize.lbfgsb',
    'scipy.optimize.linesearch',
    'scipy.optimize.minpack',
    'scipy.optimize.minpack2',
    'scipy.optimize.moduleTNC',
    'scipy.optimize.nonlin',
    'scipy.optimize.optimize',
    'scipy.optimize.slsqp',
    'scipy.optimize.tnc',
    'scipy.optimize.zeros',
    'scipy.signal.bsplines',
    'scipy.signal.filter_design',
    'scipy.signal.fir_filter_design',
    'scipy.signal.lti_conversion',
    'scipy.signal.ltisys',
    'scipy.signal.signaltools',
    'scipy.signal.spectral',
    'scipy.signal.spline',
    'scipy.signal.waveforms',
    'scipy.signal.wavelets',
    'scipy.signal.windows.windows',
    'scipy.sparse.base',
    'scipy.sparse.bsr',
    'scipy.sparse.compressed',
    'scipy.sparse.construct',
    'scipy.sparse.coo',
    'scipy.sparse.csc',
    'scipy.sparse.csr',
    'scipy.sparse.data',
    'scipy.sparse.dia',
    'scipy.sparse.dok',
    'scipy.sparse.extract',
    'scipy.sparse.lil',
    'scipy.sparse.linalg.dsolve',
    'scipy.sparse.linalg.eigen',
    'scipy.sparse.linalg.interface',
    'scipy.sparse.linalg.isolve',
    'scipy.sparse.linalg.matfuncs',
    'scipy.sparse.sparsetools',
    'scipy.sparse.spfuncs',
    'scipy.sparse.sputils',
    'scipy.spatial.ckdtree',
    'scipy.spatial.kdtree',
    'scipy.spatial.qhull',
    'scipy.spatial.transform.rotation',
    'scipy.special.add_newdocs',
    'scipy.special.basic',
    'scipy.special.cython_special',
    'scipy.special.orthogonal',
    'scipy.special.sf_error',
    'scipy.special.specfun',
    'scipy.special.spfun_stats',
    'scipy.stats.biasedurn',
    'scipy.stats.kde',
    'scipy.stats.morestats',
    'scipy.stats.mstats_basic',
    'scipy.stats.mstats_extras',
    'scipy.stats.mvn',
    'scipy.stats.stats',
]


def is_unexpected(name):
    """Check if this needs to be considered."""
    if '._' in name or '.tests' in name or '.setup' in name:
        return False

    if name in PUBLIC_MODULES:
        return False

    if name in PRIVATE_BUT_PRESENT_MODULES:
        return False

    return True


SKIP_LIST = [
    'scipy.conftest',
    'scipy.version',
]


def test_all_modules_are_expected():
    """
    Test that we don't add anything that looks like a new public module by
    accident.  Check is based on filenames.
    """

    modnames = []
    for _, modname, ispkg in pkgutil.walk_packages(path=scipy.__path__,
                                                   prefix=scipy.__name__ + '.',
                                                   onerror=None):
        if is_unexpected(modname) and modname not in SKIP_LIST:
            # We have a name that is new.  If that's on purpose, add it to
            # PUBLIC_MODULES.  We don't expect to have to add anything to
            # PRIVATE_BUT_PRESENT_MODULES.  Use an underscore in the name!
            modnames.append(modname)

    if modnames:
        raise AssertionError(f'Found unexpected modules: {modnames}')


# Stuff that clearly shouldn't be in the API and is detected by the next test
# below
SKIP_LIST_2 = [
    'scipy.char',
    'scipy.rec',
    'scipy.emath',
    'scipy.math',
    'scipy.random',
    'scipy.ctypeslib',
    'scipy.ma'
]


def test_all_modules_are_expected_2():
    """
    Method checking all objects. The pkgutil-based method in
    `test_all_modules_are_expected` does not catch imports into a namespace,
    only filenames.
    """

    def find_unexpected_members(mod_name):
        members = []
        module = importlib.import_module(mod_name)
        if hasattr(module, '__all__'):
            objnames = module.__all__
        else:
            objnames = dir(module)

        for objname in objnames:
            if not objname.startswith('_'):
                fullobjname = mod_name + '.' + objname
                if isinstance(getattr(module, objname), types.ModuleType):
                    if is_unexpected(fullobjname) and fullobjname not in SKIP_LIST_2:
                        members.append(fullobjname)

        return members

    unexpected_members = find_unexpected_members("scipy")
    for modname in PUBLIC_MODULES:
        unexpected_members.extend(find_unexpected_members(modname))

    if unexpected_members:
        raise AssertionError("Found unexpected object(s) that look like "
                             f"modules: {unexpected_members}")


def test_api_importable():
    """
    Check that all submodules listed higher up in this file can be imported
    Note that if a PRIVATE_BUT_PRESENT_MODULES entry goes missing, it may
    simply need to be removed from the list (deprecation may or may not be
    needed - apply common sense).
    """
    def check_importable(module_name):
        try:
            importlib.import_module(module_name)
        except (ImportError, AttributeError):
            return False

        return True

    module_names = []
    for module_name in PUBLIC_MODULES:
        if not check_importable(module_name):
            module_names.append(module_name)

    if module_names:
        raise AssertionError("Modules in the public API that cannot be "
                             f"imported: {module_names}")

    with warnings.catch_warnings(record=True):
        warnings.filterwarnings('always', category=DeprecationWarning)
        warnings.filterwarnings('always', category=ImportWarning)
        for module_name in PRIVATE_BUT_PRESENT_MODULES:
            if not check_importable(module_name):
                module_names.append(module_name)

    if module_names:
        raise AssertionError("Modules that are not really public but looked "
                             "public and can not be imported: "
                             f"{module_names}")


@pytest.mark.parametrize(("module_name", "correct_module"),
                         [('scipy.constants.codata', None),
                          ('scipy.constants.constants', None),
                          ('scipy.fftpack.basic', None),
                          ('scipy.fftpack.helper', None),
                          ('scipy.fftpack.pseudo_diffs', None),
                          ('scipy.fftpack.realtransforms', None),
                          ('scipy.integrate.dop', None),
                          ('scipy.integrate.lsoda', None),
                          ('scipy.integrate.odepack', None),
                          ('scipy.integrate.quadpack', None),
                          ('scipy.integrate.vode', None),
                          ('scipy.interpolate.fitpack', None),
                          ('scipy.interpolate.fitpack2', None),
                          ('scipy.interpolate.interpolate', None),
                          ('scipy.interpolate.ndgriddata', None),
                          ('scipy.interpolate.polyint', None),
                          ('scipy.interpolate.rbf', None),
                          ('scipy.io.harwell_boeing', None),
                          ('scipy.io.idl', None),
                          ('scipy.io.mmio', None),
                          ('scipy.io.netcdf', None),
                          ('scipy.io.arff.arffread', 'arff'),
                          ('scipy.io.matlab.byteordercodes', 'matlab'),
                          ('scipy.io.matlab.mio_utils', 'matlab'),
                          ('scipy.io.matlab.mio', 'matlab'),
                          ('scipy.io.matlab.mio4', 'matlab'),
                          ('scipy.io.matlab.mio5_params', 'matlab'),
                          ('scipy.io.matlab.mio5_utils', 'matlab'),
                          ('scipy.io.matlab.mio5', 'matlab'),
                          ('scipy.io.matlab.miobase', 'matlab'),
                          ('scipy.io.matlab.streams', 'matlab'),
                          ('scipy.linalg.basic', None),
                          ('scipy.linalg.decomp', None),
                          ('scipy.linalg.decomp_cholesky', None),
                          ('scipy.linalg.decomp_lu', None),
                          ('scipy.linalg.decomp_qr', None),
                          ('scipy.linalg.decomp_schur', None),
                          ('scipy.linalg.decomp_svd', None),
                          ('scipy.linalg.flinalg', None),
                          ('scipy.linalg.matfuncs', None),
                          ('scipy.linalg.misc', None),
                          ('scipy.linalg.special_matrices', None),
                          ('scipy.misc.common', None),
                          ('scipy.ndimage.filters', None),
                          ('scipy.ndimage.fourier', None),
                          ('scipy.ndimage.interpolation', None),
                          ('scipy.ndimage.measurements', None),
                          ('scipy.ndimage.morphology', None),
                          ('scipy.odr.models', None),
                          ('scipy.odr.odrpack', None),
                          ('scipy.optimize.cobyla', None),
                          ('scipy.optimize.lbfgsb', None),
                          ('scipy.optimize.linesearch', None),
                          ('scipy.optimize.minpack', None),
                          ('scipy.optimize.minpack2', None),
                          ('scipy.optimize.moduleTNC', None),
                          ('scipy.optimize.nonlin', None),
                          ('scipy.optimize.optimize', None),
                          ('scipy.optimize.slsqp', None),
                          ('scipy.optimize.tnc', None),
                          ('scipy.optimize.zeros', None),
                          ('scipy.signal.bsplines', None),
                          ('scipy.signal.filter_design', None),
                          ('scipy.signal.fir_filter_design', None),
                          ('scipy.signal.lti_conversion', None),
                          ('scipy.signal.ltisys', None),
                          ('scipy.signal.signaltools', None),
                          ('scipy.signal.spectral', None),
                          ('scipy.signal.waveforms', None),
                          ('scipy.signal.wavelets', None),
                          ('scipy.signal.windows.windows', 'windows'),
                          ('scipy.sparse.lil', None),
                          ('scipy.sparse.linalg.dsolve', 'linalg'),
                          ('scipy.sparse.linalg.eigen', 'linalg'),
                          ('scipy.sparse.linalg.interface', 'linalg'),
                          ('scipy.sparse.linalg.isolve', 'linalg'),
                          ('scipy.sparse.linalg.matfuncs', 'linalg'),
                          ('scipy.sparse.sparsetools', None),
                          ('scipy.sparse.spfuncs', None),
                          ('scipy.sparse.sputils', None),
                          ('scipy.spatial.ckdtree', None),
                          ('scipy.spatial.kdtree', None),
                          ('scipy.spatial.qhull', None),
                          ('scipy.spatial.transform.rotation', 'transform'),
                          ('scipy.special.add_newdocs', None),
                          ('scipy.special.basic', None),
                          ('scipy.special.orthogonal', None),
                          ('scipy.special.sf_error', None),
                          ('scipy.special.specfun', None),
                          ('scipy.special.spfun_stats', None),
                          ('scipy.stats.biasedurn', None),
                          ('scipy.stats.kde', None),
                          ('scipy.stats.morestats', None),
                          ('scipy.stats.mstats_basic', 'mstats'),
                          ('scipy.stats.mstats_extras', 'mstats'),
                          ('scipy.stats.mvn', None),
                          ('scipy.stats.stats', None)])
def test_private_but_present_deprecation(module_name, correct_module):
    # gh-18279, gh-17572, gh-17771 noted that deprecation warnings
    # for imports from private modules
    # were misleading. Check that this is resolved.
    module = import_module(module_name)
    if correct_module is None:
        import_name = f'scipy.{module_name.split(".")[1]}'
    else:
        import_name = f'scipy.{module_name.split(".")[1]}.{correct_module}'

    correct_import = import_module(import_name)

    # Attributes that were formerly in `module_name` can still be imported from
    # `module_name`, albeit with a deprecation warning. The specific message
    # depends on whether the attribute is public in `scipy.xxx` or not.
    for attr_name in module.__all__:
        attr = getattr(correct_import, attr_name, None)
        if attr is None:
            message = f"`{module_name}.{attr_name}` is deprecated..."
        else:
            message = f"Please import `{attr_name}` from the `{import_name}`..."
        with pytest.deprecated_call(match=message):
            getattr(module, attr_name)

    # Attributes that were not in `module_name` get an error notifying the user
    # that the attribute is not in `module_name` and that `module_name` is deprecated.
    message = f"`{module_name}` is deprecated..."
    with pytest.raises(AttributeError, match=message):
        getattr(module, "ekki")


def test_misc_doccer_deprecation():
    # gh-18279, gh-17572, gh-17771 noted that deprecation warnings
    # for imports from private modules were misleading.
    # Check that this is resolved.
    # `test_private_but_present_deprecation` cannot be used since `correct_import`
    # is a different subpackage (`_lib` instead of `misc`).
    module = import_module('scipy.misc.doccer')
    correct_import = import_module('scipy._lib.doccer')

    # Attributes that were formerly in `scipy.misc.doccer` can still be imported from
    # `scipy.misc.doccer`, albeit with a deprecation warning. The specific message
    # depends on whether the attribute is in `scipy._lib.doccer` or not.
    for attr_name in module.__all__:
        attr = getattr(correct_import, attr_name, None)
        if attr is None:
            message = f"`scipy.misc.{attr_name}` is deprecated..."
        else:
            message = f"Please import `{attr_name}` from the `scipy._lib.doccer`..."
        with pytest.deprecated_call(match=message):
            getattr(module, attr_name)

    # Attributes that were not in `scipy.misc.doccer` get an error
    # notifying the user that the attribute is not in `scipy.misc.doccer` 
    # and that `scipy.misc.doccer` is deprecated.
    message = "`scipy.misc.doccer` is deprecated..."
    with pytest.raises(AttributeError, match=message):
        getattr(module, "ekki")
