# _build_config.py.in is converted into _build_config.py during the meson build process.

from __future__ import annotations


def build_config() -> dict[str, str]:
    """
    Return a dictionary containing build configuration settings.

    All dictionary keys and values are strings, for example ``False`` is
    returned as ``"False"``.

        .. versionadded:: 1.1.0
    """
    return dict(
        # Python settings
        python_version="3.10",
        python_install_dir=r"c:/Lib/site-packages/",
        python_path=r"C:/Users/runneradmin/AppData/Local/Temp/build-env-pvxu370h/Scripts/python.exe",

        # Package versions
        contourpy_version="1.2.0",
        meson_version="1.2.3",
        mesonpy_version="0.15.0",
        pybind11_version="2.11.1",

        # Misc meson settings
        meson_backend="ninja",
        build_dir=r"D:/a/contourpy/contourpy/.mesonpy-nbc_4neo/lib/contourpy/util",
        source_dir=r"D:/a/contourpy/contourpy/lib/contourpy/util",
        cross_build="False",

        # Build options
        build_options=r"-Dbuildtype=release -Db_ndebug=if-release -Db_vscrt=md -Dvsenv=True '--native-file=D:/a/contourpy/contourpy/.mesonpy-nbc_4neo/meson-python-native-file.ini'",
        buildtype="release",
        cpp_std="c++17",
        debug="False",
        optimization="3",
        vsenv="True",
        b_ndebug="if-release",
        b_vscrt="md",

        # C++ compiler
        compiler_name="msvc",
        compiler_version="19.29.30152",
        linker_id="link",
        compile_command="cl",

        # Host machine
        host_cpu="x86_64",
        host_cpu_family="x86_64",
        host_cpu_endian="little",
        host_cpu_system="windows",

        # Build machine, same as host machine if not a cross_build
        build_cpu="x86_64",
        build_cpu_family="x86_64",
        build_cpu_endian="little",
        build_cpu_system="windows",
    )
