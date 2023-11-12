# This file is part of Patsy
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Regression tests for fixed bugs (when not otherwise better covered somewhere
# else)

from patsy import (EvalEnvironment, dmatrix, build_design_matrices,
                   PatsyError, Origin)

def test_issue_11():
    # Give a sensible error message for level mismatches
    # (At some points we've failed to put an origin= on these errors)
    env = EvalEnvironment.capture()
    data = {"X" : [0,1,2,3], "Y" : [1,2,3,4]}
    formula = "C(X) + Y"
    new_data = {"X" : [0,0,1,2,3,3,4], "Y" : [1,2,3,4,5,6,7]}
    info = dmatrix(formula, data)
    try:
        build_design_matrices([info.design_info], new_data)
    except PatsyError as e:
        assert e.origin == Origin(formula, 0, 4)
    else:
        assert False
