from __future__ import annotations

from typing import Any
from unittest import TestCase

from traitlets import TraitError


class TraitTestBase(TestCase):
    """A best testing class for basic trait types."""

    def assign(self, value: Any) -> None:
        self.obj.value = value  # type:ignore[attr-defined]

    def coerce(self, value: Any) -> Any:
        return value

    def test_good_values(self) -> None:
        if hasattr(self, "_good_values"):
            for value in self._good_values:
                self.assign(value)
                self.assertEqual(self.obj.value, self.coerce(value))  # type:ignore[attr-defined]

    def test_bad_values(self) -> None:
        if hasattr(self, "_bad_values"):
            for value in self._bad_values:
                try:
                    self.assertRaises(TraitError, self.assign, value)
                except AssertionError:
                    raise AssertionError(value) from None

    def test_default_value(self) -> None:
        if hasattr(self, "_default_value"):
            self.assertEqual(self._default_value, self.obj.value)  # type:ignore[attr-defined]

    def test_allow_none(self) -> None:
        if (
            hasattr(self, "_bad_values")
            and hasattr(self, "_good_values")
            and None in self._bad_values
        ):
            trait = self.obj.traits()["value"]  # type:ignore[attr-defined]
            try:
                trait.allow_none = True
                self._bad_values.remove(None)
                # skip coerce. Allow None casts None to None.
                self.assign(None)
                self.assertEqual(self.obj.value, None)  # type:ignore[attr-defined]
                self.test_good_values()
                self.test_bad_values()
            finally:
                # tear down
                trait.allow_none = False
                self._bad_values.append(None)

    def tearDown(self) -> None:
        # restore default value after tests, if set
        if hasattr(self, "_default_value"):
            self.obj.value = self._default_value  # type:ignore[attr-defined]
