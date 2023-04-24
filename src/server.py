import os
from xmlrpc.server import SimpleXMLRPCServer

from handle import handler, status


def start_server():
    print("Hello World")
    server = SimpleXMLRPCServer(("localhost", int(os.environ["PORT"])))
    print(f"Listening on port {os.environ['PORT']}...")
    server.register_function(handler, "training")
    server.register_function(status, "status")
    server.serve_forever()
