def call_me_back(callback):
    if callable(callback):
        callback()
