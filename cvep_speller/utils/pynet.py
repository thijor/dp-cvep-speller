import socket

from fire import Fire


def main(server: str = "localhost", port: int = 8080):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((server, port))
        while True:
            data = input("Enter message: ")
            s.sendall(data.encode())
            data = s.recv(1024)
            print("Received", repr(data))


if __name__ == "__main__":
    Fire(main)
