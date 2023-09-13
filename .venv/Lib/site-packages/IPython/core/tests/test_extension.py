import os.path

from tempfile import TemporaryDirectory

import IPython.testing.tools as tt
from IPython.utils.syspathcontext import prepended_to_syspath

ext1_content = """
def load_ipython_extension(ip):
    print("Running ext1 load")

def unload_ipython_extension(ip):
    print("Running ext1 unload")
"""

ext2_content = """
def load_ipython_extension(ip):
    print("Running ext2 load")
"""

ext3_content = """
def load_ipython_extension(ip):
    ip2 = get_ipython()
    print(ip is ip2)
"""

def test_extension_loading():
    em = get_ipython().extension_manager
    with TemporaryDirectory() as td:
        ext1 = os.path.join(td, "ext1.py")
        with open(ext1, "w", encoding="utf-8") as f:
            f.write(ext1_content)

        ext2 = os.path.join(td, "ext2.py")
        with open(ext2, "w", encoding="utf-8") as f:
            f.write(ext2_content)
        
        with prepended_to_syspath(td):
            assert 'ext1' not in em.loaded
            assert 'ext2' not in em.loaded
            
            # Load extension
            with tt.AssertPrints("Running ext1 load"):
                assert em.load_extension('ext1') is None
            assert 'ext1' in em.loaded
            
            # Should refuse to load it again
            with tt.AssertNotPrints("Running ext1 load"):
                assert em.load_extension('ext1') == 'already loaded'
            
            # Reload
            with tt.AssertPrints("Running ext1 unload"):
                with tt.AssertPrints("Running ext1 load", suppress=False):
                    em.reload_extension('ext1')
            
            # Unload
            with tt.AssertPrints("Running ext1 unload"):
                assert em.unload_extension('ext1') is None
            
            # Can't unload again
            with tt.AssertNotPrints("Running ext1 unload"):
                assert em.unload_extension('ext1') == 'not loaded'
            assert em.unload_extension('ext2') == 'not loaded'
            
            # Load extension 2
            with tt.AssertPrints("Running ext2 load"):
                assert em.load_extension('ext2') is None
            
            # Can't unload this
            assert em.unload_extension('ext2') == 'no unload function'
            
            # But can reload it
            with tt.AssertPrints("Running ext2 load"):
                em.reload_extension('ext2')


def test_extension_builtins():
    em = get_ipython().extension_manager
    with TemporaryDirectory() as td:
        ext3 = os.path.join(td, "ext3.py")
        with open(ext3, "w", encoding="utf-8") as f:
            f.write(ext3_content)
        
        assert 'ext3' not in em.loaded
        
        with prepended_to_syspath(td):
            # Load extension
            with tt.AssertPrints("True"):
                assert em.load_extension('ext3') is None
            assert 'ext3' in em.loaded


def test_non_extension():
    em = get_ipython().extension_manager
    assert em.load_extension("sys") == "no load function"
