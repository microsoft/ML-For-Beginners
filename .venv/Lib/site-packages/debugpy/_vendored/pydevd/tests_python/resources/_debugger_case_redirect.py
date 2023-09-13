# coding:utf-8
if __name__ == '__main__':
    import sys
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name)
        stream.write('text\n')
        stream.write('binary or text\n')
        stream.write('ação1\n')

        if sys.version_info[0] >= 3:
            # sys.stdout.buffer is only available on py3.
            stream.buffer.write(b'binary\n')
            # Note: this will be giberish on the receiving side because when writing bytes
            # we can't be sure what's the encoding and will treat it as utf-8 (i.e.:
            # uses PYTHONIOENCODING).
            stream.buffer.write('ação2\n'.encode(encoding='latin1'))

            # This will be ok
            stream.buffer.write('ação3\n'.encode(encoding='utf-8'))

        if sys.version_info[0] >= 3:
            stream.buffer.write(b'\xe8\xF0\x80\x80\x80\n\n')
        else:
            stream.write(b'\xe8\xF0\x80\x80\x80\n\n')

    print('TEST SUCEEDED!')
