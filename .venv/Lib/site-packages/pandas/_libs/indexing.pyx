cdef class NDFrameIndexerBase:
    """
    A base class for _NDFrameIndexer for fast instantiation and attribute access.
    """
    cdef:
        Py_ssize_t _ndim

    cdef public:
        str name
        object obj

    def __init__(self, name: str, obj):
        self.obj = obj
        self.name = name
        self._ndim = -1

    @property
    def ndim(self) -> int:
        # Delay `ndim` instantiation until required as reading it
        # from `obj` isn't entirely cheap.
        ndim = self._ndim
        if ndim == -1:
            ndim = self._ndim = self.obj.ndim
            if ndim > 2:
                raise ValueError(  # pragma: no cover
                    "NDFrameIndexer does not support NDFrame objects with ndim > 2"
                )
        return ndim
