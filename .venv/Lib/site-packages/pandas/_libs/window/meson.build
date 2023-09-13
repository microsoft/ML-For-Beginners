py.extension_module(
    'aggregations',
    ['aggregations.pyx'],
    cython_args: ['-X always_allow_keywords=true'],
    include_directories: [inc_np, inc_pd],
    subdir: 'pandas/_libs/window',
    override_options : ['cython_language=cpp'],
    install: true
)

py.extension_module(
    'indexers',
    ['indexers.pyx'],
    cython_args: ['-X always_allow_keywords=true'],
    include_directories: [inc_np, inc_pd],
    subdir: 'pandas/_libs/window',
    install: true
)
