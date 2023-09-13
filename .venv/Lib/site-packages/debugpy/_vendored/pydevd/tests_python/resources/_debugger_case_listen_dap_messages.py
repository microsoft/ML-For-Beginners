import time


def check_output_command(messages_sent):
    for _i in range(50):
        for msg in messages_sent:
            if msg.get('type') == 'event' and msg.get('event') == 'output' and 'Something' in msg['body']['output']:
                return True
        time.sleep(.1)
    return False


def check_continue_request(messages_received):
    for _i in range(50):
        for msg in messages_received:
            if msg.get('type') == 'request' and msg.get('command') == 'continue':
                return True
        time.sleep(.1)
    return False


def main():
    '''
    In this test we'll check that pydevd sends an output event and receives a continue request in
    the dap messages listener.
    '''
    messages_sent = []
    messages_received = []

    import pydevd

    class Listener(pydevd.IDAPMessagesListener):

        def before_send(self, message_as_dict):
            messages_sent.append(message_as_dict)

        def after_receive(self, message_as_dict):
            messages_received.append(message_as_dict)

    pydevd.add_dap_messages_listener(Listener())

    print('Something')  # Break here

    # Note that the message is only received in a thread, so, we have to
    # wait for it to be received.
    if not check_output_command(messages_sent):
        print('Output not received!')
        return False

    if not check_continue_request(messages_received):
        print('Continue not received!')
        return False

    print('TEST SUCEEDED!')
    return True


if __name__ == '__main__':
    assert main()
