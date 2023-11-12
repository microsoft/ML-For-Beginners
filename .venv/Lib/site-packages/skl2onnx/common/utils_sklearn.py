# SPDX-License-Identifier: Apache-2.0

import copy
from collections import OrderedDict
import warnings
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def enumerate_model_names(model, prefix="", short=True):
    """
    Enumerates ``tuple (name, model)`` associated
    to the model itself.
    """
    if isinstance(model, (list, tuple)):
        if all(map(lambda x: isinstance(x, tuple) and len(x) in (2, 3), model)):
            for i, named_mod in enumerate(model):
                name, mod = named_mod[:2]
                p = name if short and prefix == "" else "{}__{}".format(prefix, name)
                for t in enumerate_model_names(mod, p, short=short):
                    yield t
        else:
            for i, mod in enumerate(model):
                p = i if short and prefix == "" else "{}__{}".format(prefix, i)
                for t in enumerate_model_names(mod, p, short=short):
                    yield t
    elif isinstance(model, (dict, OrderedDict)):
        for name, mod in model.items():
            p = name if short and prefix == "" else "{}__{}".format(prefix, name)
            for t in enumerate_model_names(mod, p, short=short):
                yield t
    else:
        yield (prefix, model)
        reserved_atts = {
            "transformers",
            "steps",
            "transformer_list",
            "named_estimators_",
            "named_transformers_",
            "transformer_",
            "estimator_",
        }
        for key in dir(model):
            if key in ("estimators_", "estimator") and hasattr(
                model, "named_estimators_"
            ):
                continue
            if key in ("transformers_", "transformers") and hasattr(
                model, "named_transformers_"
            ):
                continue
            if key in reserved_atts or (
                key.endswith("_") and not key.endswith("__") and not key.startswith("_")
            ):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        obj = getattr(model, key)
                except AttributeError:
                    continue
                if hasattr(obj, "get_params") and isinstance(obj, BaseEstimator):
                    prefix = (
                        key if short and prefix == "" else "{}__{}".format(prefix, key)
                    )
                    yield (prefix, obj)
                elif isinstance(obj, (list, tuple, dict, OrderedDict)):
                    if not short or key not in reserved_atts:
                        prefix = (
                            key
                            if short and prefix == ""
                            else "{}__{}".format(prefix, key)
                        )
                    for t in enumerate_model_names(obj, prefix, short=short):
                        yield t


def has_pipeline(model):
    """
    Tells if a model contains a pipeline.
    """
    return any(map(lambda x: isinstance(x[1], Pipeline), enumerate_model_names(model)))


def _process_options(model, options):
    """
    Converts options defined as string into options
    ``id(model): options``. The second format is not
    pickable.
    """
    if options is None:
        return None
    if all(map(lambda x: not isinstance(x, str), options)):
        # No need to transform.
        return _process_pipeline_options(model, options)

    new_options = copy.deepcopy(options)
    names = dict(enumerate_model_names(model))
    for k, v in options.items():
        if k in names:
            new_options[id(names[k])] = v
            continue
        try:
            ri = k.rindex("__")
            m2, k2 = k[:ri], k[ri + 2 :]
        except ValueError:
            key = id(model)
            if key not in new_options:
                new_options[key] = {}
            new_options[key][k] = v
            continue
        if m2 in names:
            key = id(names[m2])
            if key not in new_options:
                new_options[key] = {}
            new_options[key][k2] = v
            continue
        raise RuntimeError(
            "Unable to find model name '{}' or '{}' in \n{}".format(
                k, m2, list(sorted(names))
            )
        )

    return _process_pipeline_options(model, new_options)


def _process_pipeline_options(model, options):
    """
    Tells the final classifier of a pipeline that
    options 'zipmap', 'nocl' or 'output_class_labels'
    were attached to the pipeline.
    """
    new_options = None
    names = dict(enumerate_model_names(model))
    for k, v in names.items():
        if id(v) in options and isinstance(v, Pipeline):
            opts = options[id(v)]
            last = v.steps[-1][1]
            key = id(last)
            for opt, val in opts.items():
                if opt not in {"zipmap", "nocl", "output_class_labels"}:
                    continue
                if new_options is None:
                    new_options = copy.deepcopy(options)
                if key not in new_options:
                    new_options[key] = {}
                if opt not in new_options[key]:
                    new_options[key][opt] = val
    return new_options or options
