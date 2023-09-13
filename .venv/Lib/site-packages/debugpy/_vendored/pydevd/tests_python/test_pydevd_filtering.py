from _pydevd_bundle.pydevd_constants import IS_WINDOWS, IS_MAC


def test_in_project_roots_prefix_01(tmpdir):
    from _pydevd_bundle.pydevd_filtering import FilesFiltering
    files_filtering = FilesFiltering()

    another = str(tmpdir.join('another'))
    assert not another.endswith('/') and not another.endswith('\\')

    files_filtering.set_library_roots([another])
    files_filtering.set_project_roots([])
    assert not files_filtering.in_project_roots(another + '/f.py')
    if IS_WINDOWS:
        assert not files_filtering.in_project_roots(another + '\\f.py')
    else:
        assert files_filtering.in_project_roots(another + '\\f.py')

    assert files_filtering.in_project_roots(another + 'f.py')


def test_in_project_roots_prefix_02(tmpdir):
    from _pydevd_bundle.pydevd_filtering import FilesFiltering
    files_filtering = FilesFiltering()

    another = str(tmpdir.join('another'))
    assert not another.endswith('/') and not another.endswith('\\')

    files_filtering.set_library_roots([])
    files_filtering.set_project_roots([another])
    assert files_filtering.in_project_roots(another + '/f.py')
    if IS_WINDOWS:
        assert files_filtering.in_project_roots(another + '\\f.py')
    else:
        assert not files_filtering.in_project_roots(another + '\\f.py')

    assert not files_filtering.in_project_roots(another + 'f.py')


def test_in_project_roots(tmpdir):
    from _pydevd_bundle.pydevd_filtering import FilesFiltering
    files_filtering = FilesFiltering()

    import os.path
    import sys

    if IS_WINDOWS:
        assert files_filtering._get_library_roots() == [
            os.path.normcase(x) + '\\' for x in files_filtering._get_default_library_roots()]
    elif IS_MAC:
        assert files_filtering._get_library_roots() == [
            x.lower() + '/' for x in files_filtering._get_default_library_roots()]
    else:
        assert files_filtering._get_library_roots() == [
            os.path.normcase(x) + '/' for x in files_filtering._get_default_library_roots()]

    site_packages = tmpdir.mkdir('site-packages')
    project_dir = tmpdir.mkdir('project')

    project_dir_inside_site_packages = str(site_packages.mkdir('project'))
    site_packages_inside_project_dir = str(project_dir.mkdir('site-packages'))

    # Convert from pytest paths to str.
    site_packages = str(site_packages)
    project_dir = str(project_dir)
    tmpdir = str(tmpdir)

    # Test permutations of project dir inside site packages and vice-versa.
    files_filtering.set_project_roots([project_dir, project_dir_inside_site_packages])
    files_filtering.set_library_roots([site_packages, site_packages_inside_project_dir])

    check = [
        (tmpdir, False),
        (site_packages, False),
        (site_packages_inside_project_dir, False),
        (project_dir, True),
        (project_dir_inside_site_packages, True),
    ]
    for (check_path, find) in check[:]:
        filename_inside = os.path.join(check_path, 'a.py')
        with open(filename_inside, 'w') as stream:
            # Note: on Github actions, tmpdir may be something as:
            # c:\\users\\runner~1\\appdata\\local\\temp\\pytest-of-runneradmin\\pytest-0\\test_in_project_roots0
            # internally this may be set as:
            # c:\\users\\runneradmin\\appdata\\local\\temp\\pytest-of-runneradmin\\pytest-0\\test_in_project_roots0
            # So, when getting the absolute path, `runner~1` will be properly expanded to `runneradmin` if the
            # file exists, but if it doesn't it's not (which may make the test fail), so, make sure
            # that we actually create the file so that things work as expected.
            stream.write('...')
        check.append((filename_inside, find))

    for check_path, find in check:
        if files_filtering.in_project_roots(check_path) != find:
            if find:
                msg = 'Expected %s to be in the project roots.\nProject roots: %s\nLibrary roots: %s\n'
            else:
                msg = 'Expected %s NOT to be in the project roots.\nProject roots: %s\nLibrary roots: %s\n'

            raise AssertionError(msg % (
                check_path,
                files_filtering._get_project_roots(),
                files_filtering._get_library_roots(),
                )
            )

    files_filtering.set_project_roots([])
    files_filtering.set_library_roots([site_packages, site_packages_inside_project_dir])

    # If the IDE did not set the project roots, consider anything not in the site
    # packages as being in a project root (i.e.: we can calculate default values for
    # site-packages but not for project roots).
    check = [
        (tmpdir, True),
        (site_packages, False),
        (site_packages_inside_project_dir, False),
        (project_dir, True),
        (project_dir_inside_site_packages, False),
        ('<foo>', False),
        ('<ipython>', True),
        ('<frozen importlib._bootstrap>', False),
    ]

    for check_path, find in check:
        assert files_filtering.in_project_roots(check_path) == find, \
            'Expected: %s to be a part of the project: %s' % (check_path, find)

    sys.path.append(str(site_packages))
    try:
        default_library_roots = files_filtering._get_default_library_roots()
        assert len(set(default_library_roots)) == len(default_library_roots), \
            'Duplicated library roots found in: %s' % (default_library_roots,)

        assert str(site_packages) in default_library_roots
        for path in sys.path:
            if os.path.exists(path) and path.endswith('site-packages'):
                assert path in default_library_roots
    finally:
        sys.path.remove(str(site_packages))


def test_filtering(tmpdir):
    from _pydevd_bundle.pydevd_filtering import FilesFiltering
    from _pydevd_bundle.pydevd_filtering import ExcludeFilter
    files_filtering = FilesFiltering()

    site_packages = tmpdir.mkdir('site-packages')
    project_dir = tmpdir.mkdir('project')

    project_dir_inside_site_packages = str(site_packages.mkdir('project'))
    site_packages_inside_project_dir = str(project_dir.mkdir('site-packages'))

    files_filtering.set_exclude_filters([
        ExcludeFilter('**/project*', True, True),
        ExcludeFilter('**/bar*', False, True),
    ])
    assert files_filtering.exclude_by_filter('/foo/project', None) is True
    assert files_filtering.exclude_by_filter('/foo/unmatched', None) is None
    assert files_filtering.exclude_by_filter('/foo/bar', None) is False


def test_glob_matching():
    from _pydevd_bundle.pydevd_filtering import glob_matches_path

    # Linux
    for sep, altsep in (('\\', '/'), ('/', None)):

        def build(path):
            if sep == '/':
                return path
            else:
                return ('c:' + path).replace('/', '\\')

        assert glob_matches_path(build('/a'), r'*', sep, altsep)

        assert not glob_matches_path(build('/a/b/c/some.py'), '/a/**/c/so?.py', sep, altsep)

        assert glob_matches_path('/a/b/c', '/a/b/*')
        assert not glob_matches_path('/a/b', '/*')
        assert glob_matches_path('/a/b', '/*/b')
        assert glob_matches_path('/a/b', '**/*')
        assert not glob_matches_path('/a/b', '**/a')

        assert glob_matches_path(build('/a/b/c/d'), '**/d', sep, altsep)
        assert not glob_matches_path(build('/a/b/c/d'), '**/c', sep, altsep)
        assert glob_matches_path(build('/a/b/c/d'), '**/c/d', sep, altsep)
        assert glob_matches_path(build('/a/b/c/d'), '**/b/c/d', sep, altsep)
        assert glob_matches_path(build('/a/b/c/d'), '/*/b/*/d', sep, altsep)
        assert glob_matches_path(build('/a/b/c/d'), '**/c/*', sep, altsep)
        assert glob_matches_path(build('/a/b/c/d'), '/a/**/c/*', sep, altsep)

        # I.e. directories are expected to end with '/', so, it'll match
        # something as **/directory/**
        assert glob_matches_path(build('/a/b/c/'), '**/c/**', sep, altsep)
        assert glob_matches_path(build('/a/b/c/'), '**/c/', sep, altsep)
        # But not something as **/directory (that'd be a file match).
        assert not glob_matches_path(build('/a/b/c/'), '**/c', sep, altsep)
        assert not glob_matches_path(build('/a/b/c'), '**/c/', sep, altsep)

        assert glob_matches_path(build('/a/b/c/d.py'), '/a/**/c/*', sep, altsep)
        assert glob_matches_path(build('/a/b/c/d.py'), '/a/**/c/*.py', sep, altsep)
        assert glob_matches_path(build('/a/b/c/some.py'), '/a/**/c/so*.py', sep, altsep)
        assert glob_matches_path(build('/a/b/c/some.py'), '/a/**/c/som?.py', sep, altsep)
        assert glob_matches_path(build('/a/b/c/d'), '/**', sep, altsep)
        assert glob_matches_path(build('/a/b/c/d'), '/**/d', sep, altsep)
        assert glob_matches_path(build('/a/b/c/d.py'), '/**/*.py', sep, altsep)
        assert glob_matches_path(build('/a/b/c/d.py'), '**/c/*.py', sep, altsep)

        if IS_WINDOWS:
            assert glob_matches_path(build('/a/b/c/d.py'), '**/C/*.py', sep, altsep)
            assert glob_matches_path(build('/a/b/C/d.py'), '**/c/*.py', sep, altsep)

        # Expected not to match.
        assert not glob_matches_path(build('/a/b/c/d'), '/**/d.py', sep, altsep)
        assert not glob_matches_path(build('/a/b/c/d.pyx'), '/a/**/c/*.py', sep, altsep)
        assert not glob_matches_path(build('/a/b/c/d'), '/*/d', sep, altsep)

        if sep == '/':
            assert not glob_matches_path(build('/a/b/c/d'), r'**\d', sep, altsep)  # Match with \ doesn't work on linux...
            assert not glob_matches_path(build('/a/b/c/d'), r'c:\**\d', sep, altsep)  # Match with drive doesn't work on linux...
        else:
            # Works in Windows.
            assert glob_matches_path(build('/a/b/c/d'), r'**\d', sep, altsep)
            assert glob_matches_path(build('/a/b/c/d'), r'c:\**\d', sep, altsep)

        # Corner cases
        assert not glob_matches_path(build('/'), r'', sep, altsep)
        assert glob_matches_path(build(''), r'', sep, altsep)
        assert not glob_matches_path(build(''), r'**', sep, altsep)
        assert glob_matches_path(build('/'), r'**', sep, altsep)
        assert glob_matches_path(build('/'), r'*', sep, altsep)


def test_rules_to_exclude_filter(tmpdir):
    from _pydevd_bundle.pydevd_process_net_command_json import _convert_rules_to_exclude_filters
    from _pydevd_bundle.pydevd_filtering import ExcludeFilter
    from random import shuffle
    dira = tmpdir.mkdir('a')
    dirb = dira.mkdir('b')
    fileb = dirb.join('fileb.py')
    fileb2 = dirb.join('fileb2.py')
    with fileb.open('w') as stream:
        stream.write('')

    def on_error(msg):
        raise AssertionError(msg)

    rules = [
        {'path': str(dira), 'include': False},
        {'path': str(dirb), 'include': True},
        {'path': str(fileb), 'include': True},
        {'path': str(fileb2), 'include': True},
        {'path': '**/foo/*.py', 'include': True},
        {'module': 'bar', 'include': False},
        {'module': 'bar.foo', 'include': True},
    ]
    shuffle(rules)
    exclude_filters = _convert_rules_to_exclude_filters(rules, on_error)
    assert exclude_filters == [
        ExcludeFilter(name=str(fileb2), exclude=False, is_path=True),
        ExcludeFilter(name=str(fileb), exclude=False, is_path=True),
        ExcludeFilter(name=str(dirb) + '/**', exclude=False, is_path=True),
        ExcludeFilter(name=str(dira) + '/**', exclude=True, is_path=True),
        ExcludeFilter(name='**/foo/*.py', exclude=False, is_path=True),
        ExcludeFilter(name='bar.foo', exclude=False, is_path=False),
        ExcludeFilter(name='bar', exclude=True, is_path=False),
    ]
