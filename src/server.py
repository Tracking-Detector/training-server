import os
from xmlrpc.server import SimpleXMLRPCServer

from src.handle import handler, status


def start_server():
    server = SimpleXMLRPCServer(("localhost", int(os.environ["PORT"])))
    print(f"Listening on port {os.environ['PORT']}...")
    server.register_function(handler, "training")
    server.register_function(status, "status")
    server.serve_forever()


if __name__ == "__main__":
    start_server()
