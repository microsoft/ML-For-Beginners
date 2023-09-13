"""Module with bad __all__

To test https://github.com/ipython/ipython/issues/9678
"""

def evil():
    pass

def puppies():
    pass

__all__ = [evil,       # Bad
           'puppies',  # Good
          ]
