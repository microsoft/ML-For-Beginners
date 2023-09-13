_algos_take_helper = custom_target('algos_take_helper_pxi',
    output: 'algos_take_helper.pxi',
    input: 'algos_take_helper.pxi.in',
    command: [
        py, tempita, '@INPUT@', '-o', '@OUTDIR@'
    ]
)
_algos_common_helper = custom_target('algos_common_helper_pxi',
    output: 'algos_common_helper.pxi',
    input: 'algos_common_helper.pxi.in',
    command: [
        py, tempita, '@INPUT@', '-o', '@OUTDIR@'
    ]
)
_khash_primitive_helper = custom_target('khash_primitive_helper_pxi',
    output: 'khash_for_primitive_helper.pxi',
    input: 'khash_for_primitive_helper.pxi.in',
    command: [
        py, tempita, '@INPUT@', '-o', '@OUTDIR@'
    ]
)
_hashtable_class_helper = custom_target('hashtable_class_helper_pxi',
    output: 'hashtable_class_helper.pxi',
    input: 'hashtable_class_helper.pxi.in',
    command: [
        py, tempita, '@INPUT@', '-o', '@OUTDIR@'
    ]
)
_hashtable_func_helper = custom_target('hashtable_func_helper_pxi',
    output: 'hashtable_func_helper.pxi',
    input: 'hashtable_func_helper.pxi.in',
    command: [
        py, tempita, '@INPUT@', '-o', '@OUTDIR@'
    ]
)
_index_class_helper = custom_target('index_class_helper_pxi',
    output: 'index_class_helper.pxi',
    input: 'index_class_helper.pxi.in',
    command: [
        py, tempita, '@INPUT@', '-o', '@OUTDIR@'
    ]
)
_sparse_op_helper = custom_target('sparse_op_helper_pxi',
    output: 'sparse_op_helper.pxi',
    input: 'sparse_op_helper.pxi.in',
    command: [
        py, tempita, '@INPUT@', '-o', '@OUTDIR@'
    ]
)
_intervaltree_helper = custom_target('intervaltree_helper_pxi',
    output: 'intervaltree.pxi',
    input: 'intervaltree.pxi.in',
    command: [
        py, tempita, '@INPUT@', '-o', '@OUTDIR@'
    ]
)
_khash_primitive_helper_dep = declare_dependency(sources: _khash_primitive_helper)

subdir('tslibs')

libs_sources = {
    # Dict of extension name -> dict of {sources, include_dirs, and deps}
    # numpy include dir is implicitly included
    'algos': {'sources': ['algos.pyx', _algos_common_helper, _algos_take_helper, _khash_primitive_helper]},
    'arrays': {'sources': ['arrays.pyx']},
    'groupby': {'sources': ['groupby.pyx']},
    'hashing': {'sources': ['hashing.pyx']},
    'hashtable': {'sources': ['hashtable.pyx', _khash_primitive_helper, _hashtable_class_helper, _hashtable_func_helper]},
    'index': {'sources': ['index.pyx', _index_class_helper]},
    'indexing': {'sources': ['indexing.pyx']},
    'internals': {'sources': ['internals.pyx']},
    'interval': {'sources': ['interval.pyx', _intervaltree_helper]},
    'join': {'sources': ['join.pyx', _khash_primitive_helper],
             'deps': _khash_primitive_helper_dep},
    'lib': {'sources': ['lib.pyx', 'src/parser/tokenizer.c']},
    'missing': {'sources': ['missing.pyx']},
    'pandas_datetime': {'sources': ['src/vendored/numpy/datetime/np_datetime.c',
                                    'src/vendored/numpy/datetime/np_datetime_strings.c',
                                    'src/datetime/date_conversions.c',
                                    'src/datetime/pd_datetime.c']},
    'pandas_parser': {'sources': ['src/parser/tokenizer.c',
                                  'src/parser/io.c',
                                  'src/parser/pd_parser.c']},
    'parsers': {'sources': ['parsers.pyx', 'src/parser/tokenizer.c', 'src/parser/io.c'],
                'deps': _khash_primitive_helper_dep},
    'json': {'sources': ['src/vendored/ujson/python/ujson.c',
                         'src/vendored/ujson/python/objToJSON.c',
                         'src/vendored/ujson/python/JSONtoObj.c',
                         'src/vendored/ujson/lib/ultrajsonenc.c',
                         'src/vendored/ujson/lib/ultrajsondec.c']},
    'ops': {'sources': ['ops.pyx']},
    'ops_dispatch': {'sources': ['ops_dispatch.pyx']},
    'properties': {'sources': ['properties.pyx']},
    'reshape': {'sources': ['reshape.pyx']},
    'sas': {'sources': ['sas.pyx']},
    'byteswap': {'sources': ['byteswap.pyx']},
    'sparse': {'sources': ['sparse.pyx', _sparse_op_helper]},
    'tslib': {'sources': ['tslib.pyx']},
    'testing': {'sources': ['testing.pyx']},
    'writers': {'sources': ['writers.pyx']}
}


foreach ext_name, ext_dict : libs_sources
    py.extension_module(
        ext_name,
        ext_dict.get('sources'),
        cython_args: ['--include-dir', meson.current_build_dir(), '-X always_allow_keywords=true'],
        include_directories: [inc_np, inc_pd],
        dependencies: ext_dict.get('deps', ''),
        subdir: 'pandas/_libs',
        install: true
    )
endforeach

py.install_sources('__init__.py',
                    pure: false,
                    subdir: 'pandas/_libs')

subdir('window')
