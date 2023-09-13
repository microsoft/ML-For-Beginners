import sys
from _pydevd_bundle.pydevd_constants import int_types, GENERATED_LEN_ATTR_NAME
from _pydevd_bundle.pydevd_resolver import TOO_LARGE_ATTR
from _pydevd_bundle import pydevd_resolver, pydevd_constants
from _pydevd_bundle import pydevd_frame_utils
import pytest


def get_frame():
    var1 = 1
    var2 = [var1]
    var3 = {33: [var1]}
    return sys._getframe()


def check_vars_dict_expected(as_dict, expected):
    assert as_dict == expected


class _DummyPyDB(object):

    def __init__(self):
        from _pydevd_bundle.pydevd_api import PyDevdAPI
        self.variable_presentation = PyDevdAPI.VariablePresentation()


def test_suspended_frames_manager():
    from _pydevd_bundle.pydevd_suspended_frames import SuspendedFramesManager
    from _pydevd_bundle.pydevd_utils import DAPGrouper
    suspended_frames_manager = SuspendedFramesManager()
    py_db = _DummyPyDB()
    with suspended_frames_manager.track_frames(py_db) as tracker:
        # : :type tracker: _FramesTracker
        thread_id = 'thread1'
        frame = get_frame()
        tracker.track(thread_id, pydevd_frame_utils.create_frames_list_from_frame(frame))

        assert suspended_frames_manager.get_thread_id_for_variable_reference(id(frame)) == thread_id

        variable = suspended_frames_manager.get_variable(id(frame))

        # Should be properly sorted.
        assert ['var1', 'var2', 'var3'] == [x.get_name()for x in variable.get_children_variables()]

        as_dict = dict((x.get_name(), x.get_var_data()) for x in variable.get_children_variables())
        var_reference = as_dict['var2'].pop('variablesReference')
        assert isinstance(var_reference, int_types)  # The variable reference is always a new int.
        assert isinstance(as_dict['var3'].pop('variablesReference'), int_types)  # The variable reference is always a new int.

        check_vars_dict_expected(as_dict, {
            'var1': {'name': 'var1', 'value': '1', 'type': 'int', 'evaluateName': 'var1', 'variablesReference': 0},
            'var2': {'name': 'var2', 'value': '[1]', 'type': 'list', 'evaluateName': 'var2'},
            'var3': {'name': 'var3', 'value': '{33: [1]}', 'type': 'dict', 'evaluateName': 'var3'}
        })

        # Now, same thing with a different format.
        as_dict = dict((x.get_name(), x.get_var_data(fmt={'hex': True})) for x in variable.get_children_variables())
        var_reference = as_dict['var2'].pop('variablesReference')
        assert isinstance(var_reference, int_types)  # The variable reference is always a new int.
        assert isinstance(as_dict['var3'].pop('variablesReference'), int_types)  # The variable reference is always a new int.

        check_vars_dict_expected(as_dict, {
            'var1': {'name': 'var1', 'value': '0x1', 'type': 'int', 'evaluateName': 'var1', 'variablesReference': 0},
            'var2': {'name': 'var2', 'value': '[0x1]', 'type': 'list', 'evaluateName': 'var2'},
            'var3': {'name': 'var3', 'value': '{0x21: [0x1]}', 'type': 'dict', 'evaluateName': 'var3'}
        })

        var2 = dict((x.get_name(), x) for x in variable.get_children_variables())['var2']
        children_vars = var2.get_children_variables()
        as_dict = (dict([x.get_name(), x.get_var_data()] for x in children_vars if x.get_name() not in DAPGrouper.SCOPES_SORTED))
        assert as_dict == {
            '0': {'name': '0', 'value': '1', 'type': 'int', 'evaluateName': 'var2[0]', 'variablesReference': 0 },
            GENERATED_LEN_ATTR_NAME: {'name': GENERATED_LEN_ATTR_NAME, 'value': '1', 'type': 'int', 'evaluateName': 'len(var2)', 'variablesReference': 0, 'presentationHint': {'attributes': ['readOnly']}, },
        }

        var3 = dict((x.get_name(), x) for x in variable.get_children_variables())['var3']
        children_vars = var3.get_children_variables()
        as_dict = (dict([x.get_name(), x.get_var_data()] for x in children_vars if x.get_name() not in DAPGrouper.SCOPES_SORTED))
        assert isinstance(as_dict['33'].pop('variablesReference'), int_types)  # The variable reference is always a new int.

        check_vars_dict_expected(as_dict, {
            '33': {'name': '33', 'value': "[1]", 'type': 'list', 'evaluateName': 'var3[33]'},
            GENERATED_LEN_ATTR_NAME: {'name': GENERATED_LEN_ATTR_NAME, 'value': '1', 'type': 'int', 'evaluateName': 'len(var3)', 'variablesReference': 0, 'presentationHint': {'attributes': ['readOnly']}, }
        })


def get_dict_large_frame():
    obj = {}
    for idx in range(pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS + +300):
        obj[idx] = (1)
    return sys._getframe()


def get_set_large_frame():
    obj = set()
    for idx in range(pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS + +300):
        obj.add(idx)
    return sys._getframe()


def test_get_child_variables():
    from _pydevd_bundle.pydevd_suspended_frames import SuspendedFramesManager
    suspended_frames_manager = SuspendedFramesManager()
    py_db = _DummyPyDB()
    for frame in (
        get_dict_large_frame(),
        get_set_large_frame(),
        ):
        with suspended_frames_manager.track_frames(py_db) as tracker:
            # : :type tracker: _FramesTracker
            thread_id = 'thread1'
            tracker.track(thread_id, pydevd_frame_utils.create_frames_list_from_frame(frame))

            assert suspended_frames_manager.get_thread_id_for_variable_reference(id(frame)) == thread_id

            variable = suspended_frames_manager.get_variable(id(frame))

            children_variables = variable.get_child_variable_named('obj').get_children_variables()

            found_too_large = False
            found_len = False
            for x in children_variables:
                if x.name == TOO_LARGE_ATTR:
                    var_data = x.get_var_data()
                    assert 'readOnly' in var_data['presentationHint']['attributes']
                    found_too_large = True
                elif x.name == GENERATED_LEN_ATTR_NAME:
                    found_len = True

            if not found_too_large:
                raise AssertionError('Expected to find variable named: %s' % (TOO_LARGE_ATTR,))
            if not found_len:
                raise AssertionError('Expected to find variable named: len()')

