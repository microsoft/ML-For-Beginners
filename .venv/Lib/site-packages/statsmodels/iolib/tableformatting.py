"""
Summary Table formating
This is here to help keep the formating consistent across the different models
"""
import copy

gen_fmt = {
    "data_fmts": ["%s", "%s", "%s", "%s", "%s"],
    "empty_cell": '',
    "colwidths": 7,
    "colsep": '   ',
    "row_pre": '  ',
    "row_post": '  ',
    "table_dec_above": '": ',
    "table_dec_below": None,
    "header_dec_below": None,
    "header_fmt": '%s',
    "stub_fmt": '%s',
    "title_align": 'c',
    "header_align": 'r',
    "data_aligns": "r",
    "stubs_align": "l",
    "fmt": 'txt'
}

# Note table_1l_fmt over rides the below formating unless it is not
# appended to table_1l
fmt_1_right = {
    "data_fmts": ["%s", "%s", "%s", "%s", "%s"],
    "empty_cell": '',
    "colwidths": 16,
    "colsep": '   ',
    "row_pre": '',
    "row_post": '',
    "table_dec_above": '": ',
    "table_dec_below": None,
    "header_dec_below": None,
    "header_fmt": '%s',
    "stub_fmt": '%s',
    "title_align": 'c',
    "header_align": 'r',
    "data_aligns": "r",
    "stubs_align": "l",
    "fmt": 'txt'
}

fmt_2 = {
    "data_fmts": ["%s", "%s", "%s", "%s"],
    "empty_cell": '',
    "colwidths": 10,
    "colsep": ' ',
    "row_pre": '  ',
    "row_post": '   ',
    "table_dec_above": '": ',
    "table_dec_below": '": ',
    "header_dec_below": '-',
    "header_fmt": '%s',
    "stub_fmt": '%s',
    "title_align": 'c',
    "header_align": 'r',
    "data_aligns": 'r',
    "stubs_align": 'l',
    "fmt": 'txt'
}


# new version  # TODO: as of when?  compared to what?  is old version needed?
fmt_base = {
    "data_fmts": ["%s", "%s", "%s", "%s", "%s"],
    "empty_cell": '',
    "colwidths": 10,
    "colsep": ' ',
    "row_pre": '',
    "row_post": '',
    "table_dec_above": '=',
    "table_dec_below": '=',  # TODO need '=' at the last subtable
    "header_dec_below": '-',
    "header_fmt": '%s',
    "stub_fmt": '%s',
    "title_align": 'c',
    "header_align": 'r',
    "data_aligns": 'r',
    "stubs_align": 'l',
    "fmt": 'txt'
}

fmt_2cols = copy.deepcopy(fmt_base)

fmt2 = {
    "data_fmts": ["%18s", "-%19s", "%18s", "%19s"],  # TODO: TODO: what?
    "colsep": ' ',
    "colwidths": 18,
    "stub_fmt": '-%21s',
}
fmt_2cols.update(fmt2)

fmt_params = copy.deepcopy(fmt_base)

fmt3 = {
    "data_fmts": ["%s", "%s", "%8s", "%s", "%11s", "%11s"],
}
fmt_params.update(fmt3)

"""
Summary Table formating
This is here to help keep the formating consistent across the different models
"""
fmt_latex = {
    'colsep': ' & ',
    'colwidths': None,
    'data_aligns': 'r',
    'data_fmt': '%s',
    'data_fmts': ['%s'],
    'empty': '',
    'empty_cell': '',
    'fmt': 'ltx',
    'header': '%s',
    'header_align': 'c',
    'header_dec_below': '\\hline',
    'header_fmt': '%s',
    'missing': '--',
    'row_dec_below': None,
    'row_post': '  \\\\',
    'strip_backslash': True,
    'stub': '%s',
    'stub_align': 'l',
    'stub_fmt': '%s',
    'table_dec_above': '\\hline',
    'table_dec_below': '\\hline'}

fmt_txt = {
    'colsep': ' ',
    'colwidths': None,
    'data_aligns': 'r',
    'data_fmts': ['%s'],
    'empty': '',
    'empty_cell': '',
    'fmt': 'txt',
    'header': '%s',
    'header_align': 'c',
    'header_dec_below': '-',
    'header_fmt': '%s',
    'missing': '--',
    'row_dec_below': None,
    'row_post': '',
    'row_pre': '',
    'stub': '%s',
    'stub_align': 'l',
    'stub_fmt': '%s',
    'table_dec_above': '-',
    'table_dec_below': None,
    'title_align': 'c'}
