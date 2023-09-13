from contextlib import contextmanager
import sys

from _pydevd_bundle.pydevd_constants import get_frame, RETURN_VALUES_DICT, \
    ForkSafeLock, GENERATED_LEN_ATTR_NAME, silence_warnings_decorator
from _pydevd_bundle.pydevd_xml import get_variable_details, get_type
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_resolver import sorted_attributes_key, TOO_LARGE_ATTR, get_var_scope
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_vars
from _pydev_bundle.pydev_imports import Exec
from _pydevd_bundle.pydevd_frame_utils import FramesList
from _pydevd_bundle.pydevd_utils import ScopeRequest, DAPGrouper, Timer
from typing import Optional


class _AbstractVariable(object):

    # Default attributes in class, set in instance.

    name = None
    value = None
    evaluate_name = None

    def __init__(self, py_db):
        assert py_db is not None
        self.py_db = py_db

    def get_name(self):
        return self.name

    def get_value(self):
        return self.value

    def get_variable_reference(self):
        return id(self.value)

    def get_var_data(self, fmt: Optional[dict]=None, context: Optional[str]=None, **safe_repr_custom_attrs):
        '''
        :param dict fmt:
            Format expected by the DAP (keys: 'hex': bool, 'rawString': bool)

        :param context:
            This is the context in which the variable is being requested. Valid values:
                "watch",
                "repl",
                "hover",
                "clipboard"
        '''
        timer = Timer()
        safe_repr = SafeRepr()
        if fmt is not None:
            safe_repr.convert_to_hex = fmt.get('hex', False)
            safe_repr.raw_value = fmt.get('rawString', False)
        for key, val in safe_repr_custom_attrs.items():
            setattr(safe_repr, key, val)

        type_name, _type_qualifier, _is_exception_on_eval, resolver, value = get_variable_details(
            self.value, to_string=safe_repr, context=context)

        is_raw_string = type_name in ('str', 'bytes', 'bytearray')

        attributes = []

        if is_raw_string:
            attributes.append('rawString')

        name = self.name

        if self._is_return_value:
            attributes.append('readOnly')
            name = '(return) %s' % (name,)

        elif name in (TOO_LARGE_ATTR, GENERATED_LEN_ATTR_NAME):
            attributes.append('readOnly')

        try:
            if self.value.__class__ == DAPGrouper:
                type_name = ''
        except:
            pass  # Ignore errors accessing __class__.

        var_data = {
            'name': name,
            'value': value,
            'type': type_name,
        }

        if self.evaluate_name is not None:
            var_data['evaluateName'] = self.evaluate_name

        if resolver is not None:  # I.e.: it's a container
            var_data['variablesReference'] = self.get_variable_reference()
        else:
            var_data['variablesReference'] = 0  # It's mandatory (although if == 0 it doesn't have children).

        if len(attributes) > 0:
            var_data['presentationHint'] = {'attributes': attributes}

        timer.report_if_compute_repr_attr_slow('', name, type_name)
        return var_data

    def get_children_variables(self, fmt=None, scope=None):
        raise NotImplementedError()

    def get_child_variable_named(self, name, fmt=None, scope=None):
        for child_var in self.get_children_variables(fmt=fmt, scope=scope):
            if child_var.get_name() == name:
                return child_var
        return None

    def _group_entries(self, lst, handle_return_values):
        scope_to_grouper = {}

        group_entries = []
        if isinstance(self.value, DAPGrouper):
            new_lst = lst
        else:
            new_lst = []
            get_presentation = self.py_db.variable_presentation.get_presentation
            # Now that we have the contents, group items.
            for attr_name, attr_value, evaluate_name in lst:
                scope = get_var_scope(attr_name, attr_value, evaluate_name, handle_return_values)

                entry = (attr_name, attr_value, evaluate_name)
                if scope:
                    presentation = get_presentation(scope)
                    if presentation == 'hide':
                        continue

                    elif presentation == 'inline':
                        new_lst.append(entry)

                    else:  # group
                        if scope not in scope_to_grouper:
                            grouper = DAPGrouper(scope)
                            scope_to_grouper[scope] = grouper
                        else:
                            grouper = scope_to_grouper[scope]

                        grouper.contents_debug_adapter_protocol.append(entry)

                else:
                    new_lst.append(entry)

            for scope in DAPGrouper.SCOPES_SORTED:
                grouper = scope_to_grouper.get(scope)
                if grouper is not None:
                    group_entries.append((scope, grouper, None))

        return new_lst, group_entries


class _ObjectVariable(_AbstractVariable):

    def __init__(self, py_db, name, value, register_variable, is_return_value=False, evaluate_name=None, frame=None):
        _AbstractVariable.__init__(self, py_db)
        self.frame = frame
        self.name = name
        self.value = value
        self._register_variable = register_variable
        self._register_variable(self)
        self._is_return_value = is_return_value
        self.evaluate_name = evaluate_name

    @silence_warnings_decorator
    @overrides(_AbstractVariable.get_children_variables)
    def get_children_variables(self, fmt=None, scope=None):
        _type, _type_name, resolver = get_type(self.value)

        children_variables = []
        if resolver is not None:  # i.e.: it's a container.
            if hasattr(resolver, 'get_contents_debug_adapter_protocol'):
                # The get_contents_debug_adapter_protocol needs to return sorted.
                lst = resolver.get_contents_debug_adapter_protocol(self.value, fmt=fmt)
            else:
                # If there's no special implementation, the default is sorting the keys.
                dct = resolver.get_dictionary(self.value)
                lst = sorted(dct.items(), key=lambda tup: sorted_attributes_key(tup[0]))
                # No evaluate name in this case.
                lst = [(key, value, None) for (key, value) in lst]

            lst, group_entries = self._group_entries(lst, handle_return_values=False)
            if group_entries:
                lst = group_entries + lst
            parent_evaluate_name = self.evaluate_name
            if parent_evaluate_name:
                for key, val, evaluate_name in lst:
                    if evaluate_name is not None:
                        if callable(evaluate_name):
                            evaluate_name = evaluate_name(parent_evaluate_name)
                        else:
                            evaluate_name = parent_evaluate_name + evaluate_name
                    variable = _ObjectVariable(
                        self.py_db, key, val, self._register_variable, evaluate_name=evaluate_name, frame=self.frame)
                    children_variables.append(variable)
            else:
                for key, val, evaluate_name in lst:
                    # No evaluate name
                    variable = _ObjectVariable(self.py_db, key, val, self._register_variable, frame=self.frame)
                    children_variables.append(variable)

        return children_variables

    def change_variable(self, name, value, py_db, fmt=None):

        children_variable = self.get_child_variable_named(name)
        if children_variable is None:
            return None

        var_data = children_variable.get_var_data()
        evaluate_name = var_data.get('evaluateName')

        if not evaluate_name:
            # Note: right now we only pass control to the resolver in the cases where
            # there's no evaluate name (the idea being that if we can evaluate it,
            # we can use that evaluation to set the value too -- if in the future
            # a case where this isn't true is found this logic may need to be changed).
            _type, _type_name, container_resolver = get_type(self.value)
            if hasattr(container_resolver, 'change_var_from_name'):
                try:
                    new_value = eval(value)
                except:
                    return None
                new_key = container_resolver.change_var_from_name(self.value, name, new_value)
                if new_key is not None:
                    return _ObjectVariable(
                        self.py_db, new_key, new_value, self._register_variable, evaluate_name=None, frame=self.frame)

                return None
            else:
                return None

        frame = self.frame
        if frame is None:
            return None

        try:
            # This handles the simple cases (such as dict, list, object)
            Exec('%s=%s' % (evaluate_name, value), frame.f_globals, frame.f_locals)
        except:
            return None

        return self.get_child_variable_named(name, fmt=fmt)


def sorted_variables_key(obj):
    return sorted_attributes_key(obj.name)


class _FrameVariable(_AbstractVariable):

    def __init__(self, py_db, frame, register_variable):
        _AbstractVariable.__init__(self, py_db)
        self.frame = frame

        self.name = self.frame.f_code.co_name
        self.value = frame

        self._register_variable = register_variable
        self._register_variable(self)

    def change_variable(self, name, value, py_db, fmt=None):
        frame = self.frame

        pydevd_vars.change_attr_expression(frame, name, value, py_db)

        return self.get_child_variable_named(name, fmt=fmt)

    @silence_warnings_decorator
    @overrides(_AbstractVariable.get_children_variables)
    def get_children_variables(self, fmt=None, scope=None):
        children_variables = []
        if scope is not None:
            assert isinstance(scope, ScopeRequest)
            scope = scope.scope

        if scope in ('locals', None):
            dct = self.frame.f_locals
        elif scope == 'globals':
            dct = self.frame.f_globals
        else:
            raise AssertionError('Unexpected scope: %s' % (scope,))

        lst, group_entries = self._group_entries([(x[0], x[1], None) for x in list(dct.items()) if x[0] != '_pydev_stop_at_break'], handle_return_values=True)
        group_variables = []

        for key, val, _ in group_entries:
            # Make sure that the contents in the group are also sorted.
            val.contents_debug_adapter_protocol.sort(key=lambda v:sorted_attributes_key(v[0]))
            variable = _ObjectVariable(self.py_db, key, val, self._register_variable, False, key, frame=self.frame)
            group_variables.append(variable)

        for key, val, _ in lst:
            is_return_value = key == RETURN_VALUES_DICT
            if is_return_value:
                for return_key, return_value in val.items():
                    variable = _ObjectVariable(
                        self.py_db, return_key, return_value, self._register_variable, is_return_value, '%s[%r]' % (key, return_key), frame=self.frame)
                    children_variables.append(variable)
            else:
                variable = _ObjectVariable(self.py_db, key, val, self._register_variable, is_return_value, key, frame=self.frame)
                children_variables.append(variable)

        # Frame variables always sorted.
        children_variables.sort(key=sorted_variables_key)
        if group_variables:
            # Groups have priority over other variables.
            children_variables = group_variables + children_variables

        return children_variables


class _FramesTracker(object):
    '''
    This is a helper class to be used to track frames when a thread becomes suspended.
    '''

    def __init__(self, suspended_frames_manager, py_db):
        self._suspended_frames_manager = suspended_frames_manager
        self.py_db = py_db
        self._frame_id_to_frame = {}

        # Note that a given frame may appear in multiple threads when we have custom
        # frames added, but as those are coroutines, this map will point to the actual
        # main thread (which is the one that needs to be suspended for us to get the
        # variables).
        self._frame_id_to_main_thread_id = {}

        # A map of the suspended thread id -> list(frames ids) -- note that
        # frame ids are kept in order (the first one is the suspended frame).
        self._thread_id_to_frame_ids = {}

        self._thread_id_to_frames_list = {}

        # The main suspended thread (if this is a coroutine this isn't the id of the
        # coroutine thread, it's the id of the actual suspended thread).
        self._main_thread_id = None

        # Helper to know if it was already untracked.
        self._untracked = False

        # We need to be thread-safe!
        self._lock = ForkSafeLock()

        self._variable_reference_to_variable = {}

    def _register_variable(self, variable):
        variable_reference = variable.get_variable_reference()
        self._variable_reference_to_variable[variable_reference] = variable

    def obtain_as_variable(self, name, value, evaluate_name=None, frame=None):
        if evaluate_name is None:
            evaluate_name = name

        variable_reference = id(value)
        variable = self._variable_reference_to_variable.get(variable_reference)
        if variable is not None:
            return variable

        # Still not created, let's do it now.
        return _ObjectVariable(
            self.py_db, name, value, self._register_variable, is_return_value=False, evaluate_name=evaluate_name, frame=frame)

    def get_main_thread_id(self):
        return self._main_thread_id

    def get_variable(self, variable_reference):
        return self._variable_reference_to_variable[variable_reference]

    def track(self, thread_id, frames_list, frame_custom_thread_id=None):
        '''
        :param thread_id:
            The thread id to be used for this frame.

        :param FramesList frames_list:
            A list of frames to be tracked (the first is the topmost frame which is suspended at the given thread).

        :param frame_custom_thread_id:
            If None this this is the id of the thread id for the custom frame (i.e.: coroutine).
        '''
        assert frames_list.__class__ == FramesList
        with self._lock:
            coroutine_or_main_thread_id = frame_custom_thread_id or thread_id

            if coroutine_or_main_thread_id in self._suspended_frames_manager._thread_id_to_tracker:
                sys.stderr.write('pydevd: Something is wrong. Tracker being added twice to the same thread id.\n')

            self._suspended_frames_manager._thread_id_to_tracker[coroutine_or_main_thread_id] = self
            self._main_thread_id = thread_id

            frame_ids_from_thread = self._thread_id_to_frame_ids.setdefault(
                coroutine_or_main_thread_id, [])

            self._thread_id_to_frames_list[coroutine_or_main_thread_id] = frames_list
            for frame in frames_list:
                frame_id = id(frame)
                self._frame_id_to_frame[frame_id] = frame
                _FrameVariable(self.py_db, frame, self._register_variable)  # Instancing is enough to register.
                self._suspended_frames_manager._variable_reference_to_frames_tracker[frame_id] = self
                frame_ids_from_thread.append(frame_id)

                self._frame_id_to_main_thread_id[frame_id] = thread_id

            frame = None

    def untrack_all(self):
        with self._lock:
            if self._untracked:
                # Calling multiple times is expected for the set next statement.
                return
            self._untracked = True
            for thread_id in self._thread_id_to_frame_ids:
                self._suspended_frames_manager._thread_id_to_tracker.pop(thread_id, None)

            for frame_id in self._frame_id_to_frame:
                del self._suspended_frames_manager._variable_reference_to_frames_tracker[frame_id]

            self._frame_id_to_frame.clear()
            self._frame_id_to_main_thread_id.clear()
            self._thread_id_to_frame_ids.clear()
            self._thread_id_to_frames_list.clear()
            self._main_thread_id = None
            self._suspended_frames_manager = None
            self._variable_reference_to_variable.clear()

    def get_frames_list(self, thread_id):
        with self._lock:
            return self._thread_id_to_frames_list.get(thread_id)

    def find_frame(self, thread_id, frame_id):
        with self._lock:
            return self._frame_id_to_frame.get(frame_id)

    def create_thread_suspend_command(self, thread_id, stop_reason, message, suspend_type):
        with self._lock:
            # First one is topmost frame suspended.
            frames_list = self._thread_id_to_frames_list[thread_id]

            cmd = self.py_db.cmd_factory.make_thread_suspend_message(
                self.py_db, thread_id, frames_list, stop_reason, message, suspend_type)

            frames_list = None
            return cmd


class SuspendedFramesManager(object):

    def __init__(self):
        self._thread_id_to_fake_frames = {}
        self._thread_id_to_tracker = {}

        # Mappings
        self._variable_reference_to_frames_tracker = {}

    def _get_tracker_for_variable_reference(self, variable_reference):
        tracker = self._variable_reference_to_frames_tracker.get(variable_reference)
        if tracker is not None:
            return tracker

        for _thread_id, tracker in self._thread_id_to_tracker.items():
            try:
                tracker.get_variable(variable_reference)
            except KeyError:
                pass
            else:
                return tracker

        return None

    def get_thread_id_for_variable_reference(self, variable_reference):
        '''
        We can't evaluate variable references values on any thread, only in the suspended
        thread (the main reason for this is that in UI frameworks inspecting a UI object
        from a different thread can potentially crash the application).

        :param int variable_reference:
            The variable reference (can be either a frame id or a reference to a previously
            gotten variable).

        :return str:
            The thread id for the thread to be used to inspect the given variable reference or
            None if the thread was already resumed.
        '''
        frames_tracker = self._get_tracker_for_variable_reference(variable_reference)
        if frames_tracker is not None:
            return frames_tracker.get_main_thread_id()
        return None

    def get_frame_tracker(self, thread_id):
        return self._thread_id_to_tracker.get(thread_id)

    def get_variable(self, variable_reference):
        '''
        :raises KeyError
        '''
        frames_tracker = self._get_tracker_for_variable_reference(variable_reference)
        if frames_tracker is None:
            raise KeyError()
        return frames_tracker.get_variable(variable_reference)

    def get_frames_list(self, thread_id):
        tracker = self._thread_id_to_tracker.get(thread_id)
        if tracker is None:
            return None
        return tracker.get_frames_list(thread_id)

    @contextmanager
    def track_frames(self, py_db):
        tracker = _FramesTracker(self, py_db)
        try:
            yield tracker
        finally:
            tracker.untrack_all()

    def add_fake_frame(self, thread_id, frame_id, frame):
        self._thread_id_to_fake_frames.setdefault(thread_id, {})[int(frame_id)] = frame

    def find_frame(self, thread_id, frame_id):
        try:
            if frame_id == "*":
                return get_frame()  # any frame is specified with "*"
            frame_id = int(frame_id)

            fake_frames = self._thread_id_to_fake_frames.get(thread_id)
            if fake_frames is not None:
                frame = fake_frames.get(frame_id)
                if frame is not None:
                    return frame

            frames_tracker = self._thread_id_to_tracker.get(thread_id)
            if frames_tracker is not None:
                frame = frames_tracker.find_frame(thread_id, frame_id)
                if frame is not None:
                    return frame

            return None
        except:
            pydev_log.exception()
            return None
