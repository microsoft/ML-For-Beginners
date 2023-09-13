from tornado.ioloop import IOLoop
from tornado.netutil import ThreadedResolver

# When this module is imported, it runs getaddrinfo on a thread. Since
# the hostname is unicode, getaddrinfo attempts to import encodings.idna
# but blocks on the import lock. Verify that ThreadedResolver avoids
# this deadlock.

resolver = ThreadedResolver()
IOLoop.current().run_sync(lambda: resolver.resolve("localhost", 80))
