# pylint: disable=missing-docstring, too-few-public-methods
"""
Test the trait-type ``UseEnum``.
"""

import enum
import unittest

from traitlets import CaselessStrEnum, Enum, FuzzyEnum, HasTraits, TraitError, UseEnum

# -----------------------------------------------------------------------------
# TEST SUPPORT:
# -----------------------------------------------------------------------------


class Color(enum.Enum):
    red = 1
    green = 2
    blue = 3
    yellow = 4


class OtherColor(enum.Enum):
    red = 0
    green = 1


class CSColor(enum.Enum):
    red = 1
    Green = 2
    BLUE = 3
    YeLLoW = 4


color_choices = "red Green  BLUE YeLLoW".split()


# -----------------------------------------------------------------------------
# TESTSUITE:
# -----------------------------------------------------------------------------
class TestUseEnum(unittest.TestCase):
    # pylint: disable=invalid-name

    class Example(HasTraits):
        color = UseEnum(Color, help="Color enum")

    def test_assign_enum_value(self):
        example = self.Example()
        example.color = Color.green
        self.assertEqual(example.color, Color.green)

    def test_assign_all_enum_values(self):
        # pylint: disable=no-member
        enum_values = list(Color.__members__.values())
        for value in enum_values:
            self.assertIsInstance(value, Color)
            example = self.Example()
            example.color = value
            self.assertEqual(example.color, value)
            self.assertIsInstance(value, Color)

    def test_assign_enum_value__with_other_enum_raises_error(self):
        example = self.Example()
        with self.assertRaises(TraitError):
            example.color = OtherColor.green

    def test_assign_enum_name_1(self):
        # -- CONVERT: string => Enum value (item)
        example = self.Example()
        example.color = "red"
        self.assertEqual(example.color, Color.red)

    def test_assign_enum_value_name(self):
        # -- CONVERT: string => Enum value (item)
        # pylint: disable=no-member
        enum_names = [enum_val.name for enum_val in Color.__members__.values()]
        for value in enum_names:
            self.assertIsInstance(value, str)
            example = self.Example()
            enum_value = Color.__members__.get(value)
            example.color = value
            self.assertIs(example.color, enum_value)
            self.assertEqual(example.color.name, value)  # type:ignore

    def test_assign_scoped_enum_value_name(self):
        # -- CONVERT: string => Enum value (item)
        scoped_names = ["Color.red", "Color.green", "Color.blue", "Color.yellow"]
        for value in scoped_names:
            example = self.Example()
            example.color = value
            self.assertIsInstance(example.color, Color)
            self.assertEqual(str(example.color), value)

    def test_assign_bad_enum_value_name__raises_error(self):
        # -- CONVERT: string => Enum value (item)
        bad_enum_names = ["UNKNOWN_COLOR", "RED", "Green", "blue2"]
        for value in bad_enum_names:
            example = self.Example()
            with self.assertRaises(TraitError):
                example.color = value

    def test_assign_enum_value_number_1(self):
        # -- CONVERT: number => Enum value (item)
        example = self.Example()
        example.color = 1  # == Color.red.value
        example.color = Color.red.value
        self.assertEqual(example.color, Color.red)

    def test_assign_enum_value_number(self):
        # -- CONVERT: number => Enum value (item)
        # pylint: disable=no-member
        enum_numbers = [enum_val.value for enum_val in Color.__members__.values()]
        for value in enum_numbers:
            self.assertIsInstance(value, int)
            example = self.Example()
            example.color = value
            self.assertIsInstance(example.color, Color)
            self.assertEqual(example.color.value, value)  # type:ignore

    def test_assign_bad_enum_value_number__raises_error(self):
        # -- CONVERT: number => Enum value (item)
        bad_numbers = [-1, 0, 5]
        for value in bad_numbers:
            self.assertIsInstance(value, int)
            assert UseEnum(Color).select_by_number(value, None) is None
            example = self.Example()
            with self.assertRaises(TraitError):
                example.color = value

    def test_ctor_without_default_value(self):
        # -- IMPLICIT: default_value = Color.red (first enum-value)
        class Example2(HasTraits):
            color = UseEnum(Color)

        example = Example2()
        self.assertEqual(example.color, Color.red)

    def test_ctor_with_default_value_as_enum_value(self):
        # -- CONVERT: number => Enum value (item)
        class Example2(HasTraits):
            color = UseEnum(Color, default_value=Color.green)

        example = Example2()
        self.assertEqual(example.color, Color.green)

    def test_ctor_with_default_value_none_and_not_allow_none(self):
        # -- IMPLICIT: default_value = Color.red (first enum-value)
        class Example2(HasTraits):
            color1 = UseEnum(Color, default_value=None, allow_none=False)
            color2 = UseEnum(Color, default_value=None)

        example = Example2()
        self.assertEqual(example.color1, Color.red)
        self.assertEqual(example.color2, Color.red)

    def test_ctor_with_default_value_none_and_allow_none(self):
        class Example2(HasTraits):
            color1 = UseEnum(Color, default_value=None, allow_none=True)
            color2 = UseEnum(Color, allow_none=True)

        example = Example2()
        self.assertIs(example.color1, None)
        self.assertIs(example.color2, None)

    def test_assign_none_without_allow_none_resets_to_default_value(self):
        class Example2(HasTraits):
            color1 = UseEnum(Color, allow_none=False)
            color2 = UseEnum(Color)

        example = Example2()
        example.color1 = None
        example.color2 = None
        self.assertIs(example.color1, Color.red)
        self.assertIs(example.color2, Color.red)

    def test_assign_none_to_enum_or_none(self):
        class Example2(HasTraits):
            color = UseEnum(Color, allow_none=True)

        example = Example2()
        example.color = None
        self.assertIs(example.color, None)

    def test_assign_bad_value_with_to_enum_or_none(self):
        class Example2(HasTraits):
            color = UseEnum(Color, allow_none=True)

        example = Example2()
        with self.assertRaises(TraitError):
            example.color = "BAD_VALUE"

    def test_info(self):
        choices = color_choices

        class Example(HasTraits):
            enum1 = Enum(choices, allow_none=False)
            enum2 = CaselessStrEnum(choices, allow_none=False)
            enum3 = FuzzyEnum(choices, allow_none=False)
            enum4 = UseEnum(CSColor, allow_none=False)

        for i in range(1, 5):
            attr = "enum%s" % i
            enum = getattr(Example, attr)

            enum.allow_none = True

            info = enum.info()
            self.assertEqual(len(info.split(", ")), len(choices), info.split(", "))
            self.assertIn("or None", info)

            info = enum.info_rst()
            self.assertEqual(len(info.split("|")), len(choices), info.split("|"))
            self.assertIn("or `None`", info)
            # Check no single `\` exists.
            self.assertNotRegex(info, r"\b\\\b")

            enum.allow_none = False

            info = enum.info()
            self.assertEqual(len(info.split(", ")), len(choices), info.split(", "))
            self.assertNotIn("None", info)

            info = enum.info_rst()
            self.assertEqual(len(info.split("|")), len(choices), info.split("|"))
            self.assertNotIn("None", info)
            # Check no single `\` exists.
            self.assertNotRegex(info, r"\b\\\b")


# -----------------------------------------------------------------------------
# TESTSUITE:
# -----------------------------------------------------------------------------


class TestFuzzyEnum(unittest.TestCase):
    # Check mostly `validate()`, Ctor must be checked on generic `Enum`
    # or `CaselessStrEnum`.

    def test_search_all_prefixes__overwrite(self):
        class FuzzyExample(HasTraits):
            color = FuzzyEnum(color_choices, help="Color enum")

        example = FuzzyExample()
        for color in color_choices:
            for wlen in range(1, len(color)):
                value = color[:wlen]

                example.color = value
                self.assertEqual(example.color, color)

                example.color = value.upper()
                self.assertEqual(example.color, color)

                example.color = value.lower()
                self.assertEqual(example.color, color)

    def test_search_all_prefixes__ctor(self):
        class FuzzyExample(HasTraits):
            color = FuzzyEnum(color_choices, help="Color enum")

        for color in color_choices:
            for wlen in range(1, len(color)):
                value = color[:wlen]

                example = FuzzyExample()
                example.color = value
                self.assertEqual(example.color, color)

                example = FuzzyExample()
                example.color = value.upper()
                self.assertEqual(example.color, color)

                example = FuzzyExample()
                example.color = value.lower()
                self.assertEqual(example.color, color)

    def test_search_substrings__overwrite(self):
        class FuzzyExample(HasTraits):
            color = FuzzyEnum(color_choices, help="Color enum", substring_matching=True)

        example = FuzzyExample()
        for color in color_choices:
            for wlen in range(0, 2):
                value = color[wlen:]

                example.color = value
                self.assertEqual(example.color, color)

                example.color = value.upper()
                self.assertEqual(example.color, color)

                example.color = value.lower()
                self.assertEqual(example.color, color)

    def test_search_substrings__ctor(self):
        class FuzzyExample(HasTraits):
            color = FuzzyEnum(color_choices, help="Color enum", substring_matching=True)

        color = color_choices[-1]  # 'YeLLoW'
        for end in (-1, len(color)):
            for start in range(1, len(color) - 2):
                value = color[start:end]

                example = FuzzyExample()
                example.color = value
                self.assertEqual(example.color, color)

                example = FuzzyExample()
                example.color = value.upper()
                self.assertEqual(example.color, color)

    def test_assign_other_raises(self):
        def new_trait_class(case_sensitive, substring_matching):
            class Example(HasTraits):
                color = FuzzyEnum(
                    color_choices,
                    case_sensitive=case_sensitive,
                    substring_matching=substring_matching,
                )

            return Example

        example = new_trait_class(case_sensitive=False, substring_matching=False)()
        with self.assertRaises(TraitError):
            example.color = ""
        with self.assertRaises(TraitError):
            example.color = "BAD COLOR"
        with self.assertRaises(TraitError):
            example.color = "ed"

        example = new_trait_class(case_sensitive=True, substring_matching=False)()
        with self.assertRaises(TraitError):
            example.color = ""
        with self.assertRaises(TraitError):
            example.color = "Red"  # not 'red'

        example = new_trait_class(case_sensitive=True, substring_matching=True)()
        with self.assertRaises(TraitError):
            example.color = ""
        with self.assertRaises(TraitError):
            example.color = "BAD COLOR"
        with self.assertRaises(TraitError):
            example.color = "green"  # not 'Green'
        with self.assertRaises(TraitError):
            example.color = "lue"  # not (b)'LUE'
        with self.assertRaises(TraitError):
            example.color = "lUE"  # not (b)'LUE'

        example = new_trait_class(case_sensitive=False, substring_matching=True)()
        with self.assertRaises(TraitError):
            example.color = ""
        with self.assertRaises(TraitError):
            example.color = "BAD COLOR"

    def test_ctor_with_default_value(self):
        def new_trait_class(default_value, case_sensitive, substring_matching):
            class Example(HasTraits):
                color = FuzzyEnum(
                    color_choices,
                    default_value=default_value,
                    case_sensitive=case_sensitive,
                    substring_matching=substring_matching,
                )

            return Example

        for color in color_choices:
            example = new_trait_class(color, False, False)()
            self.assertEqual(example.color, color)

            example = new_trait_class(color.upper(), False, False)()
            self.assertEqual(example.color, color)

        color = color_choices[-1]  # 'YeLLoW'
        example = new_trait_class(color, True, False)()
        self.assertEqual(example.color, color)

        # FIXME: default value not validated!
        # with self.assertRaises(TraitError):
        #    example = new_trait_class(color.lower(), True, False)
