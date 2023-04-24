import xmlrpc.client

server_url = "http://localhost:3000/"
client = xmlrpc.client.ServerProxy(server_url)
result = client.status()
assert result["status"] == "RUNNING"
