from IPython.utils.shimmodule import ShimModule
import IPython


def test_shimmodule_repr_does_not_fail_on_import_error():
    shim_module = ShimModule("shim_module", mirror="mirrored_module_does_not_exist")
    repr(shim_module)


def test_shimmodule_repr_forwards_to_module():
    shim_module = ShimModule("shim_module", mirror="IPython")
    assert repr(shim_module) == repr(IPython)
