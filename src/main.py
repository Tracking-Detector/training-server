import os
from handle import handler
from xmlrpc.server import SimpleXMLRPCServer

server = SimpleXMLRPCServer(("localhost", os.environ['PORT']))
print(f"Listening on port {os.environ['PORT']}...")
server.register_function(handler, "training")
server.serve_forever()