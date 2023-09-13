# This is a dummy for code-completion purposes.

def __unicode__(self):
    """
    Return "app_label.model_label.manager_name". 
    """

def _copy_to_model(self, model):
    """
    Makes a copy of the manager and assigns it to 'model', which should be
    a child of the existing model (used when inheriting a manager from an
    abstract base class).
    """


def _db(self):
    """

    """


def _get_queryset_methods(cls, queryset_class):
    """

    """


def _hints(self):
    """
    dict() -> new empty dictionary
    dict(mapping) -> new dictionary initialized from a mapping object's
        (key, value) pairs
    dict(iterable) -> new dictionary initialized as if via:
        d = {}
        for k, v in iterable:
            d[k] = v
    dict(**kwargs) -> new dictionary initialized with the name=value pairs
        in the keyword argument list.  For example:  dict(one=1, two=2)
    """


def _inherited(self):
    """

    """


def _insert(self, *args, **kwargs):
    """
    Inserts a new record for the given model. This provides an interface to
    the InsertQuery class and is how Model.save() is implemented.
    """


def _queryset_class(self):
    """
    Represents a lazy database lookup for a set of objects.
    """


def _set_creation_counter(self):
    """
    Sets the creation counter value for this instance and increments the
    class-level copy.
    """


def _update(self, *args, **kwargs):
    """
    A version of update that accepts field objects instead of field names.
    Used primarily for model saving and not intended for use by general
    code (it requires too much poking around at model internals to be
    useful at that level).
    """


def aggregate(self, *args, **kwargs):
    """
    Returns a dictionary containing the calculations (aggregation)
    over the current queryset
    
    If args is present the expression is passed as a kwarg using
    the Aggregate object's default alias.
    """


def all(self):
    """
    @rtype: django.db.models.query.QuerySet
    """


def annotate(self, *args, **kwargs):
    """
    Return a query set in which the returned objects have been annotated
    with data aggregated from related fields.
    """


def bulk_create(self, *args, **kwargs):
    """
    Inserts each of the instances into the database. This does *not* call
    save() on each of the instances, does not send any pre/post save
    signals, and does not set the primary key attribute if it is an
    autoincrement field.
    """


def check(self, **kwargs):
    """

    """


def complex_filter(self, *args, **kwargs):
    """
    Returns a new QuerySet instance with filter_obj added to the filters.
    
    filter_obj can be a Q object (or anything with an add_to_query()
    method) or a dictionary of keyword lookup arguments.
    
    This exists to support framework features such as 'limit_choices_to',
    and usually it will be more natural to use other methods.
    
    @rtype: django.db.models.query.QuerySet
    """


def contribute_to_class(self, model, name):
    """

    """


def count(self, *args, **kwargs):
    """
    Performs a SELECT COUNT() and returns the number of records as an
    integer.
    
    If the QuerySet is already fully cached this simply returns the length
    of the cached results set to avoid multiple SELECT COUNT(*) calls.
    """


def create(self, *args, **kwargs):
    """
    Creates a new object with the given kwargs, saving it to the database
    and returning the created object.
    """


def creation_counter(self):
    """

    """


def dates(self, *args, **kwargs):
    """
    Returns a list of date objects representing all available dates for
    the given field_name, scoped to 'kind'.
    """


def datetimes(self, *args, **kwargs):
    """
    Returns a list of datetime objects representing all available
    datetimes for the given field_name, scoped to 'kind'.
    """


def db(self):
    """

    """


def db_manager(self, using=None, hints=None):
    """

    """


def defer(self, *args, **kwargs):
    """
    Defers the loading of data for certain fields until they are accessed.
    The set of fields to defer is added to any existing set of deferred
    fields. The only exception to this is if None is passed in as the only
    parameter, in which case all deferrals are removed (None acts as a
    reset option).
    """


def distinct(self, *args, **kwargs):
    """
    Returns a new QuerySet instance that will select only distinct results.
    
    @rtype: django.db.models.query.QuerySet
    """


def earliest(self, *args, **kwargs):
    """

    """


def exclude(self, *args, **kwargs):
    """
    Returns a new QuerySet instance with NOT (args) ANDed to the existing
    set.
    
    @rtype: django.db.models.query.QuerySet
    """


def exists(self, *args, **kwargs):
    """

    """


def extra(self, *args, **kwargs):
    """
    Adds extra SQL fragments to the query.
    """


def filter(self, *args, **kwargs):
    """
    Returns a new QuerySet instance with the args ANDed to the existing
    set.
    
    @rtype: django.db.models.query.QuerySet
    """


def first(self, *args, **kwargs):
    """
    Returns the first object of a query, returns None if no match is found.
    """


def from_queryset(cls, queryset_class, class_name=None):
    """

    """


def get(self, *args, **kwargs):
    """
    Performs the query and returns a single object matching the given
    keyword arguments.
    """


def get_or_create(self, *args, **kwargs):
    """
    Looks up an object with the given kwargs, creating one if necessary.
    Returns a tuple of (object, created), where created is a boolean
    specifying whether an object was created.
    """


def get_queryset(self):
    """
    Returns a new QuerySet object.  Subclasses can override this method to
    easily customize the behavior of the Manager.
    
    @rtype: django.db.models.query.QuerySet
    """


def in_bulk(self, *args, **kwargs):
    """
    Returns a dictionary mapping each of the given IDs to the object with
    that ID.
    """


def iterator(self, *args, **kwargs):
    """
    An iterator over the results from applying this QuerySet to the
    database.
    """


def last(self, *args, **kwargs):
    """
    Returns the last object of a query, returns None if no match is found.
    """


def latest(self, *args, **kwargs):
    """

    """


def model(self):
    """
    MyModel(id)
    """


def none(self, *args, **kwargs):
    """
    Returns an empty QuerySet.
    
    @rtype: django.db.models.query.QuerySet
    """


def only(self, *args, **kwargs):
    """
    Essentially, the opposite of defer. Only the fields passed into this
    method and that are not already specified as deferred are loaded
    immediately when the queryset is evaluated.
    """


def order_by(self, *args, **kwargs):
    """
    Returns a new QuerySet instance with the ordering changed.
    
    @rtype: django.db.models.query.QuerySet
    """


def prefetch_related(self, *args, **kwargs):
    """
    Returns a new QuerySet instance that will prefetch the specified
    Many-To-One and Many-To-Many related objects when the QuerySet is
    evaluated.
    
    When prefetch_related() is called more than once, the list of lookups to
    prefetch is appended to. If prefetch_related(None) is called, the list
    is cleared.
    
    @rtype: django.db.models.query.QuerySet
    """


def raw(self, *args, **kwargs):
    """

    """


def reverse(self, *args, **kwargs):
    """
    Reverses the ordering of the QuerySet.
    
    @rtype: django.db.models.query.QuerySet
    """


def select_for_update(self, *args, **kwargs):
    """
    Returns a new QuerySet instance that will select objects with a
    FOR UPDATE lock.
    
    @rtype: django.db.models.query.QuerySet
    """


def select_related(self, *args, **kwargs):
    """
    Returns a new QuerySet instance that will select related objects.
    
    If fields are specified, they must be ForeignKey fields and only those
    related objects are included in the selection.
    
    If select_related(None) is called, the list is cleared.
    
    @rtype: django.db.models.query.QuerySet
    """


def update(self, *args, **kwargs):
    """
    Updates all elements in the current QuerySet, setting all the given
    fields to the appropriate values.
    """


def update_or_create(self, *args, **kwargs):
    """
    Looks up an object with the given kwargs, updating one with defaults
    if it exists, otherwise creates a new one.
    Returns a tuple (object, created), where created is a boolean
    specifying whether an object was created.
    """


def using(self, *args, **kwargs):
    """
    Selects which database this QuerySet should execute its query against.
    
    @rtype: django.db.models.query.QuerySet
    """


def values(self, *args, **kwargs):
    """

    """


def values_list(self, *args, **kwargs):
    """

    """

