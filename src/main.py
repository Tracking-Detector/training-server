import os
from xmlrpc.server import SimpleXMLRPCServer

from handle import handler

server = SimpleXMLRPCServer(("localhost", os.environ["PORT"]))
print(f"Listening on port {os.environ['PORT']}...")
server.register_function(handler, "training")
server.serve_forever()
