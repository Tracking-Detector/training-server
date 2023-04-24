import multiprocessing as mp
import os
import xmlrpc.client
from time import sleep

from server import start_server


def test_server_availability():
    # given
    os.environ["PORT"] = "3000"
    server_thread = mp.Process(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    sleep(5)
    server_url = "http://localhost:3000/"
    client = xmlrpc.client.ServerProxy(server_url)
    # when
    result = client.status()

    # then
    server_thread.terminate()
    assert result["status"] == "RUNNING"
