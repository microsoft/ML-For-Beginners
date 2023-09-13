from _pydevd_bundle._debug_adapter.pydevd_schema_log import debug_exception
import json
import itertools
from functools import partial


class BaseSchema(object):

    @staticmethod
    def initialize_ids_translation():
        BaseSchema._dap_id_to_obj_id = {0:0, None:None}
        BaseSchema._obj_id_to_dap_id = {0:0, None:None}
        BaseSchema._next_dap_id = partial(next, itertools.count(1))

    def to_json(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def _translate_id_to_dap(obj_id):
        if obj_id == '*':
            return '*'
        # Note: we don't invalidate ids, so, if some object starts using the same id
        # of another object, the same id will be used.
        dap_id = BaseSchema._obj_id_to_dap_id.get(obj_id)
        if dap_id is None:
            dap_id = BaseSchema._obj_id_to_dap_id[obj_id] = BaseSchema._next_dap_id()
            BaseSchema._dap_id_to_obj_id[dap_id] = obj_id
        return dap_id

    @staticmethod
    def _translate_id_from_dap(dap_id):
        if dap_id == '*':
            return '*'
        try:
            return BaseSchema._dap_id_to_obj_id[dap_id]
        except:
            raise KeyError('Wrong ID sent from the client: %s' % (dap_id,))

    @staticmethod
    def update_dict_ids_to_dap(dct):
        return dct

    @staticmethod
    def update_dict_ids_from_dap(dct):
        return dct


BaseSchema.initialize_ids_translation()

_requests_to_types = {}
_responses_to_types = {}
_event_to_types = {}
_all_messages = {}


def register(cls):
    _all_messages[cls.__name__] = cls
    return cls


def register_request(command):

    def do_register(cls):
        _requests_to_types[command] = cls
        return cls

    return do_register


def register_response(command):

    def do_register(cls):
        _responses_to_types[command] = cls
        return cls

    return do_register


def register_event(event):

    def do_register(cls):
        _event_to_types[event] = cls
        return cls

    return do_register


def from_dict(dct, update_ids_from_dap=False):
    msg_type = dct.get('type')
    if msg_type is None:
        raise ValueError('Unable to make sense of message: %s' % (dct,))

    if msg_type == 'request':
        to_type = _requests_to_types
        use = dct['command']

    elif msg_type == 'response':
        to_type = _responses_to_types
        use = dct['command']

    else:
        to_type = _event_to_types
        use = dct['event']

    cls = to_type.get(use)
    if cls is None:
        raise ValueError('Unable to create message from dict: %s. %s not in %s' % (dct, use, sorted(to_type.keys())))
    try:
        return cls(update_ids_from_dap=update_ids_from_dap, **dct)
    except:
        msg = 'Error creating %s from %s' % (cls, dct)
        debug_exception(msg)
        raise


def from_json(json_msg, update_ids_from_dap=False, on_dict_loaded=lambda dct:None):
    if isinstance(json_msg, bytes):
        json_msg = json_msg.decode('utf-8')

    as_dict = json.loads(json_msg)
    on_dict_loaded(as_dict)
    try:
        return from_dict(as_dict, update_ids_from_dap=update_ids_from_dap)
    except:
        if as_dict.get('type') == 'response' and not as_dict.get('success'):
            # Error messages may not have required body (return as a generic Response).
            Response = _all_messages['Response']
            return Response(**as_dict)
        else:
            raise


def get_response_class(request):
    if request.__class__ == dict:
        return _responses_to_types[request['command']]
    return _responses_to_types[request.command]


def build_response(request, kwargs=None):
    if kwargs is None:
        kwargs = {'success':True}
    else:
        if 'success' not in kwargs:
            kwargs['success'] = True
    response_class = _responses_to_types[request.command]
    kwargs.setdefault('seq', -1)  # To be overwritten before sending
    return response_class(command=request.command, request_seq=request.seq, **kwargs)
