from inspect import Parameter, signature
from unittest import TestCase

from ...traitlets import HasTraits, Int, Unicode
from ..decorators import signature_has_traits


class TestExpandSignature(TestCase):
    def test_no_init(self):
        @signature_has_traits
        class Foo(HasTraits):
            number1 = Int()
            number2 = Int()
            value = Unicode("Hello")

        parameters = signature(Foo).parameters
        parameter_names = list(parameters)

        self.assertIs(parameters["args"].kind, Parameter.VAR_POSITIONAL)
        self.assertEqual("args", parameter_names[0])

        self.assertIs(parameters["number1"].kind, Parameter.KEYWORD_ONLY)
        self.assertIs(parameters["number2"].kind, Parameter.KEYWORD_ONLY)
        self.assertIs(parameters["value"].kind, Parameter.KEYWORD_ONLY)

        self.assertIs(parameters["kwargs"].kind, Parameter.VAR_KEYWORD)
        self.assertEqual("kwargs", parameter_names[-1])

        f = Foo(number1=32, value="World")
        self.assertEqual(f.number1, 32)
        self.assertEqual(f.number2, 0)
        self.assertEqual(f.value, "World")

    def test_partial_init(self):
        @signature_has_traits
        class Foo(HasTraits):
            number1 = Int()
            number2 = Int()
            value = Unicode("Hello")

            def __init__(self, arg1, **kwargs):
                self.arg1 = arg1

                super().__init__(**kwargs)

        parameters = signature(Foo).parameters
        parameter_names = list(parameters)

        self.assertIs(parameters["arg1"].kind, Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual("arg1", parameter_names[0])

        self.assertIs(parameters["number1"].kind, Parameter.KEYWORD_ONLY)
        self.assertIs(parameters["number2"].kind, Parameter.KEYWORD_ONLY)
        self.assertIs(parameters["value"].kind, Parameter.KEYWORD_ONLY)

        self.assertIs(parameters["kwargs"].kind, Parameter.VAR_KEYWORD)
        self.assertEqual("kwargs", parameter_names[-1])

        f = Foo(1, number1=32, value="World")
        self.assertEqual(f.arg1, 1)
        self.assertEqual(f.number1, 32)
        self.assertEqual(f.number2, 0)
        self.assertEqual(f.value, "World")

    def test_duplicate_init(self):
        @signature_has_traits
        class Foo(HasTraits):
            number1 = Int()
            number2 = Int()

            def __init__(self, number1, **kwargs):
                self.test = number1

                super().__init__(number1=number1, **kwargs)

        parameters = signature(Foo).parameters
        parameter_names = list(parameters)

        self.assertListEqual(parameter_names, ["number1", "number2", "kwargs"])

        f = Foo(number1=32, number2=36)
        self.assertEqual(f.test, 32)
        self.assertEqual(f.number1, 32)
        self.assertEqual(f.number2, 36)

    def test_full_init(self):
        @signature_has_traits
        class Foo(HasTraits):
            number1 = Int()
            number2 = Int()
            value = Unicode("Hello")

            def __init__(self, arg1, arg2=None, *pos_args, **kw_args):
                self.arg1 = arg1
                self.arg2 = arg2
                self.pos_args = pos_args
                self.kw_args = kw_args

                super().__init__(*pos_args, **kw_args)

        parameters = signature(Foo).parameters
        parameter_names = list(parameters)

        self.assertIs(parameters["arg1"].kind, Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual("arg1", parameter_names[0])

        self.assertIs(parameters["arg2"].kind, Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual("arg2", parameter_names[1])

        self.assertIs(parameters["pos_args"].kind, Parameter.VAR_POSITIONAL)
        self.assertEqual("pos_args", parameter_names[2])

        self.assertIs(parameters["number1"].kind, Parameter.KEYWORD_ONLY)
        self.assertIs(parameters["number2"].kind, Parameter.KEYWORD_ONLY)
        self.assertIs(parameters["value"].kind, Parameter.KEYWORD_ONLY)

        self.assertIs(parameters["kw_args"].kind, Parameter.VAR_KEYWORD)
        self.assertEqual("kw_args", parameter_names[-1])

        f = Foo(1, 3, 45, "hey", number1=32, value="World")
        self.assertEqual(f.arg1, 1)
        self.assertEqual(f.arg2, 3)
        self.assertTupleEqual(f.pos_args, (45, "hey"))
        self.assertEqual(f.number1, 32)
        self.assertEqual(f.number2, 0)
        self.assertEqual(f.value, "World")

    def test_no_kwargs(self):
        with self.assertRaises(RuntimeError):

            @signature_has_traits
            class Foo(HasTraits):
                number1 = Int()
                number2 = Int()

                def __init__(self, arg1, arg2=None):
                    pass
