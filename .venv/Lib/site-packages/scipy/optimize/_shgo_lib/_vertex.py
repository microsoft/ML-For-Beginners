import collections
from abc import ABC, abstractmethod

import numpy as np

from scipy._lib._util import MapWrapper


class VertexBase(ABC):
    """
    Base class for a vertex.
    """
    def __init__(self, x, nn=None, index=None):
        """
        Initiation of a vertex object.

        Parameters
        ----------
        x : tuple or vector
            The geometric location (domain).
        nn : list, optional
            Nearest neighbour list.
        index : int, optional
            Index of vertex.
        """
        self.x = x
        self.hash = hash(self.x)  # Save precomputed hash

        if nn is not None:
            self.nn = set(nn)  # can use .indexupdate to add a new list
        else:
            self.nn = set()

        self.index = index

    def __hash__(self):
        return self.hash

    def __getattr__(self, item):
        if item not in ['x_a']:
            raise AttributeError(f"{type(self)} object has no attribute "
                                 f"'{item}'")
        if item == 'x_a':
            self.x_a = np.array(self.x)
            return self.x_a

    @abstractmethod
    def connect(self, v):
        raise NotImplementedError("This method is only implemented with an "
                                  "associated child of the base class.")

    @abstractmethod
    def disconnect(self, v):
        raise NotImplementedError("This method is only implemented with an "
                                  "associated child of the base class.")

    def star(self):
        """Returns the star domain ``st(v)`` of the vertex.

        Parameters
        ----------
        v :
            The vertex ``v`` in ``st(v)``

        Returns
        -------
        st : set
            A set containing all the vertices in ``st(v)``
        """
        self.st = self.nn
        self.st.add(self)
        return self.st


class VertexScalarField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R associated with
    the geometry built from the VertexBase class
    """

    def __init__(self, x, field=None, nn=None, index=None, field_args=(),
                 g_cons=None, g_cons_args=()):
        """
        Parameters
        ----------
        x : tuple,
            vector of vertex coordinates
        field : callable, optional
            a scalar field f: R^n --> R associated with the geometry
        nn : list, optional
            list of nearest neighbours
        index : int, optional
            index of the vertex
        field_args : tuple, optional
            additional arguments to be passed to field
        g_cons : callable, optional
            constraints on the vertex
        g_cons_args : tuple, optional
            additional arguments to be passed to g_cons

        """
        super().__init__(x, nn=nn, index=index)

        # Note Vertex is only initiated once for all x so only
        # evaluated once
        # self.feasible = None

        # self.f is externally defined by the cache to allow parallel
        # processing
        # None type that will break arithmetic operations unless defined
        # self.f = None

        self.check_min = True
        self.check_max = True

    def connect(self, v):
        """Connects self to another vertex object v.

        Parameters
        ----------
        v : VertexBase or VertexScalarField object
        """
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

            # Flags for checking homology properties:
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)

            # Flags for checking homology properties:
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def minimiser(self):
        """Check whether this vertex is strictly less than all its
           neighbours"""
        if self.check_min:
            self._min = all(self.f < v.f for v in self.nn)
            self.check_min = False

        return self._min

    def maximiser(self):
        """
        Check whether this vertex is strictly greater than all its
        neighbours.
        """
        if self.check_max:
            self._max = all(self.f > v.f for v in self.nn)
            self.check_max = False

        return self._max


class VertexVectorField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R^m associated with
    the geometry built from the VertexBase class.
    """

    def __init__(self, x, sfield=None, vfield=None, field_args=(),
                 vfield_args=(), g_cons=None,
                 g_cons_args=(), nn=None, index=None):
        super().__init__(x, nn=nn, index=index)

        raise NotImplementedError("This class is still a work in progress")


class VertexCacheBase:
    """Base class for a vertex cache for a simplicial complex."""
    def __init__(self):

        self.cache = collections.OrderedDict()
        self.nfev = 0  # Feasible points
        self.index = -1

    def __iter__(self):
        for v in self.cache:
            yield self.cache[v]
        return

    def size(self):
        """Returns the size of the vertex cache."""
        return self.index + 1

    def print_out(self):
        headlen = len(f"Vertex cache of size: {len(self.cache)}:")
        print('=' * headlen)
        print(f"Vertex cache of size: {len(self.cache)}:")
        print('=' * headlen)
        for v in self.cache:
            self.cache[v].print_out()


class VertexCube(VertexBase):
    """Vertex class to be used for a pure simplicial complex with no associated
    differential geometry (single level domain that exists in R^n)"""
    def __init__(self, x, nn=None, index=None):
        super().__init__(x, nn=nn, index=index)

    def connect(self, v):
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)


class VertexCacheIndex(VertexCacheBase):
    def __init__(self):
        """
        Class for a vertex cache for a simplicial complex without an associated
        field. Useful only for building and visualising a domain complex.

        Parameters
        ----------
        """
        super().__init__()
        self.Vertex = VertexCube

    def __getitem__(self, x, nn=None):
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, index=self.index)
            # logging.info("New generated vertex at x = {}".format(x))
            # NOTE: Surprisingly high performance increase if logging
            # is commented out
            self.cache[x] = xval
            return self.cache[x]


class VertexCacheField(VertexCacheBase):
    def __init__(self, field=None, field_args=(), g_cons=None, g_cons_args=(),
                 workers=1):
        """
        Class for a vertex cache for a simplicial complex with an associated
        field.

        Parameters
        ----------
        field : callable
            Scalar or vector field callable.
        field_args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            field function
        g_cons : dict or sequence of dict, optional
            Constraints definition.
            Function(s) ``R**n`` in the form::
        g_cons_args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            constraint functions
        workers : int  optional
            Uses `multiprocessing.Pool <multiprocessing>`) to compute the field
             functions in parallel.

        """
        super().__init__()
        self.index = -1
        self.Vertex = VertexScalarField
        self.field = field
        self.field_args = field_args
        self.wfield = FieldWrapper(field, field_args)  # if workers is not 1

        self.g_cons = g_cons
        self.g_cons_args = g_cons_args
        self.wgcons = ConstraintWrapper(g_cons, g_cons_args)
        self.gpool = set()  # A set of tuples to process for feasibility

        # Field processing objects
        self.fpool = set()  # A set of tuples to process for scalar function
        self.sfc_lock = False  # True if self.fpool is non-Empty

        self.workers = workers
        self._mapwrapper = MapWrapper(workers)

        if workers == 1:
            self.process_gpool = self.proc_gpool
            if g_cons is None:
                self.process_fpool = self.proc_fpool_nog
            else:
                self.process_fpool = self.proc_fpool_g
        else:
            self.process_gpool = self.pproc_gpool
            if g_cons is None:
                self.process_fpool = self.pproc_fpool_nog
            else:
                self.process_fpool = self.pproc_fpool_g

    def __getitem__(self, x, nn=None):
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, field=self.field, nn=nn, index=self.index,
                               field_args=self.field_args,
                               g_cons=self.g_cons,
                               g_cons_args=self.g_cons_args)

            self.cache[x] = xval  # Define in cache
            self.gpool.add(xval)  # Add to pool for processing feasibility
            self.fpool.add(xval)  # Add to pool for processing field values
            return self.cache[x]

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def process_pools(self):
        if self.g_cons is not None:
            self.process_gpool()
        self.process_fpool()
        self.proc_minimisers()

    def feasibility_check(self, v):
        v.feasible = True
        for g, args in zip(self.g_cons, self.g_cons_args):
            # constraint may return more than 1 value.
            if np.any(g(v.x_a, *args) < 0.0):
                v.f = np.inf
                v.feasible = False
                break

    def compute_sfield(self, v):
        """Compute the scalar field values of a vertex object `v`.

        Parameters
        ----------
        v : VertexBase or VertexScalarField object
        """
        try:
            v.f = self.field(v.x_a, *self.field_args)
            self.nfev += 1
        except AttributeError:
            v.f = np.inf
            # logging.warning(f"Field function not found at x = {self.x_a}")
        if np.isnan(v.f):
            v.f = np.inf

    def proc_gpool(self):
        """Process all constraints."""
        if self.g_cons is not None:
            for v in self.gpool:
                self.feasibility_check(v)
        # Clean the pool
        self.gpool = set()

    def pproc_gpool(self):
        """Process all constraints in parallel."""
        gpool_l = []
        for v in self.gpool:
            gpool_l.append(v.x_a)

        G = self._mapwrapper(self.wgcons.gcons, gpool_l)
        for v, g in zip(self.gpool, G):
            v.feasible = g  # set vertex object attribute v.feasible = g (bool)

    def proc_fpool_g(self):
        """Process all field functions with constraints supplied."""
        for v in self.fpool:
            if v.feasible:
                self.compute_sfield(v)
        # Clean the pool
        self.fpool = set()

    def proc_fpool_nog(self):
        """Process all field functions with no constraints supplied."""
        for v in self.fpool:
            self.compute_sfield(v)
        # Clean the pool
        self.fpool = set()

    def pproc_fpool_g(self):
        """
        Process all field functions with constraints supplied in parallel.
        """
        self.wfield.func
        fpool_l = []
        for v in self.fpool:
            if v.feasible:
                fpool_l.append(v.x_a)
            else:
                v.f = np.inf
        F = self._mapwrapper(self.wfield.func, fpool_l)
        for va, f in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f  # set vertex object attribute v.f = f
            self.nfev += 1
        # Clean the pool
        self.fpool = set()

    def pproc_fpool_nog(self):
        """
        Process all field functions with no constraints supplied in parallel.
        """
        self.wfield.func
        fpool_l = []
        for v in self.fpool:
            fpool_l.append(v.x_a)
        F = self._mapwrapper(self.wfield.func, fpool_l)
        for va, f in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f  # set vertex object attribute v.f = f
            self.nfev += 1
        # Clean the pool
        self.fpool = set()

    def proc_minimisers(self):
        """Check for minimisers."""
        for v in self:
            v.minimiser()
            v.maximiser()


class ConstraintWrapper:
    """Object to wrap constraints to pass to `multiprocessing.Pool`."""
    def __init__(self, g_cons, g_cons_args):
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args

    def gcons(self, v_x_a):
        vfeasible = True
        for g, args in zip(self.g_cons, self.g_cons_args):
            # constraint may return more than 1 value.
            if np.any(g(v_x_a, *args) < 0.0):
                vfeasible = False
                break
        return vfeasible


class FieldWrapper:
    """Object to wrap field to pass to `multiprocessing.Pool`."""
    def __init__(self, field, field_args):
        self.field = field
        self.field_args = field_args

    def func(self, v_x_a):
        try:
            v_f = self.field(v_x_a, *self.field_args)
        except Exception:
            v_f = np.inf
        if np.isnan(v_f):
            v_f = np.inf

        return v_f
