import pytest

from joblib._utils import eval_expr


@pytest.mark.parametrize(
    "expr",
    ["exec('import os')", "print(1)", "import os", "1+1; import os", "1^1"],
)
def test_eval_expr_invalid(expr):
    with pytest.raises(
        ValueError, match="is not a valid or supported arithmetic"
    ):
        eval_expr(expr)


@pytest.mark.parametrize(
    "expr, result",
    [
        ("2*6", 12),
        ("2**6", 64),
        ("1 + 2*3**(4) / (6 + -7)", -161.0),
        ("(20 // 3) % 5", 1),
    ],
)
def test_eval_expr_valid(expr, result):
    assert eval_expr(expr) == result
