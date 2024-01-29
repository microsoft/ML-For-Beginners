# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.arff` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'MetaData', 'loadarff', 'ArffError', 'ParseArffError',
    'r_meta', 'r_comment', 'r_empty', 'r_headerline',
    'r_datameta', 'r_relation', 'r_attribute', 'r_nominal',
    'r_date', 'r_comattrval', 'r_wcomattrval', 'Attribute',
    'NominalAttribute', 'NumericAttribute', 'StringAttribute',
    'DateAttribute', 'RelationalAttribute', 'to_attribute',
    'csv_sniffer_has_bug_last_field', 'workaround_csv_sniffer_bug_last_field',
    'split_data_line', 'tokenize_attribute', 'tokenize_single_comma',
    'tokenize_single_wcomma', 'read_relational_attribute', 'read_header',
    'basic_stats', 'print_attribute', 'test_weka'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.arff", module="arffread",
                                   private_modules=["_arffread"], all=__all__,
                                   attribute=name)
