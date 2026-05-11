import socket
import struct
import pickle


def _recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf


def send_data(sock, raw):
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def recv_data(sock):
    raw_len = _recvall(sock, 4)
    if not raw_len:
        raise ConnectionError("[ERR] Server closed connection.")
    return _recvall(sock, struct.unpack(">I", raw_len)[0])


def server_handler_interface(server_host: str, server_port: int, data: dict):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(120)
        sock.connect((server_host, server_port))
        send_data(sock, pickle.dumps(data))
        return pickle.loads(recv_data(sock))
