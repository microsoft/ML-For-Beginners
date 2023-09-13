import pydevd
from _pydevd_bundle._debug_adapter import pydevd_schema

body = pydevd_schema.OutputEventBody('some output', 'my_category')
event = pydevd_schema.OutputEvent(body)
pydevd.send_json_message(event)

pydevd.send_json_message({
    "type": "event",
    "event": "output",
    "body": {"output": "some output 2", "category": "my_category2"}
})

print('TEST SUCEEDED')
