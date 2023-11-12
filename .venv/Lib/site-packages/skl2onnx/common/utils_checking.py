# SPDX-License-Identifier: Apache-2.0


from inspect import signature
from collections import OrderedDict


def check_signature(fct, reference, skip=None):
    """
    Checks that two functions have the same signature
    (same parameter names).
    Raises an exception otherwise.
    """

    def select_parameters(pars):
        new_pars = OrderedDict()
        for i, (name, p) in enumerate(pars.items()):
            if i >= 3 and name in ("op_type", "op_domain", "op_version"):
                if p.default is not None:
                    # Parameters op_type and op_domain are skipped.
                    continue
            new_pars[name] = p
        return new_pars

    sig = signature(fct)
    sig_ref = signature(reference)
    fct_pars = select_parameters(sig.parameters)
    ref_pars = select_parameters(sig_ref.parameters)
    if len(fct_pars) != len(ref_pars):
        raise TypeError(
            "Function '{}' must have {} parameters but has {}."
            "".format(fct.__name__, len(ref_pars), len(fct_pars))
        )
    for i, (a, b) in enumerate(zip(fct_pars, ref_pars)):
        if a != b and skip is not None and b not in skip and a not in skip:
            raise NameError(
                "Parameter name mismatch at position {}."
                "Function '{}' has '{}' but '{}' is expected."
                "".format(i + 1, fct.__name__, a, b)
            )
