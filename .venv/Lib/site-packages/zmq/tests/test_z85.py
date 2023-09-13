"""Test Z85 encoding

confirm values and roundtrip with test values from the reference implementation.
"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from unittest import TestCase

from zmq.utils import z85


class TestZ85(TestCase):
    def test_client_public(self):
        client_public = (
            b"\xBB\x88\x47\x1D\x65\xE2\x65\x9B"
            b"\x30\xC5\x5A\x53\x21\xCE\xBB\x5A"
            b"\xAB\x2B\x70\xA3\x98\x64\x5C\x26"
            b"\xDC\xA2\xB2\xFC\xB4\x3F\xC5\x18"
        )
        encoded = z85.encode(client_public)

        assert encoded == b"Yne@$w-vo<fVvi]a<NY6T1ed:M$fCG*[IaLV{hID"
        decoded = z85.decode(encoded)
        assert decoded == client_public

    def test_client_secret(self):
        client_secret = (
            b"\x7B\xB8\x64\xB4\x89\xAF\xA3\x67"
            b"\x1F\xBE\x69\x10\x1F\x94\xB3\x89"
            b"\x72\xF2\x48\x16\xDF\xB0\x1B\x51"
            b"\x65\x6B\x3F\xEC\x8D\xFD\x08\x88"
        )
        encoded = z85.encode(client_secret)

        assert encoded == b"D:)Q[IlAW!ahhC2ac:9*A}h:p?([4%wOTJ%JR%cs"
        decoded = z85.decode(encoded)
        assert decoded == client_secret

    def test_server_public(self):
        server_public = (
            b"\x54\xFC\xBA\x24\xE9\x32\x49\x96"
            b"\x93\x16\xFB\x61\x7C\x87\x2B\xB0"
            b"\xC1\xD1\xFF\x14\x80\x04\x27\xC5"
            b"\x94\xCB\xFA\xCF\x1B\xC2\xD6\x52"
        )
        encoded = z85.encode(server_public)

        assert encoded == b"rq:rM>}U?@Lns47E1%kR.o@n%FcmmsL/@{H8]yf7"
        decoded = z85.decode(encoded)
        assert decoded == server_public

    def test_server_secret(self):
        server_secret = (
            b"\x8E\x0B\xDD\x69\x76\x28\xB9\x1D"
            b"\x8F\x24\x55\x87\xEE\x95\xC5\xB0"
            b"\x4D\x48\x96\x3F\x79\x25\x98\x77"
            b"\xB4\x9C\xD9\x06\x3A\xEA\xD3\xB7"
        )
        encoded = z85.encode(server_secret)

        assert encoded == b"JTKVSB%%)wK0E.X)V>+}o?pNmC{O&4W4b!Ni{Lh6"
        decoded = z85.decode(encoded)
        assert decoded == server_secret
