Extensions allow extending the debugger without modifying the debugger code. This is implemented with explicit namespace
packages.

To implement your own extension:

1. Ensure that the root folder of your extension is in sys.path (add it to PYTHONPATH) 
2. Ensure that your module follows the directory structure below
3. The ``__init__.py`` files inside the pydevd_plugin and extension folder must contain the preamble below,
and nothing else.
Preamble: 
```python
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    import pkgutil
    __path__ = pkgutil.extend_path(__path__, __name__)
```
4. Your plugin name inside the extensions folder must start with `"pydevd_plugin"`
5. Implement one or more of the abstract base classes defined in `_pydevd_bundle.pydevd_extension_api`. This can be done
by either inheriting from them or registering with the abstract base class.

* Directory structure:
```
|--  root_directory-> must be on python path
|    |-- pydevd_plugins
|    |   |-- __init__.py -> must contain preamble
|    |   |-- extensions
|    |   |   |-- __init__.py -> must contain preamble
|    |   |   |-- pydevd_plugin_plugin_name.py
```