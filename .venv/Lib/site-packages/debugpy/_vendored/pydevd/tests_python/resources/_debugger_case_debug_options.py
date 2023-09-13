
import pydevd
# Some hackery to get the PyDevJsonCommandProcessor which is not exposed.
try:
    json_command_processor = pydevd.get_global_debugger().reader.process_net_command_json.__self__
except:
    json_command_processor = pydevd.get_global_debugger().reader.process_net_command_json.im_self

print(json_command_processor._options.to_json())

print('TEST SUCEEDED!')
