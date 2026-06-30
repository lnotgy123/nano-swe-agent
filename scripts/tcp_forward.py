from __future__ import annotations

import argparse
import selectors
import socket
import socketserver


class ForwardHandler(socketserver.BaseRequestHandler):
    target_host: str
    target_port: int

    def handle(self) -> None:
        with socket.create_connection((self.target_host, self.target_port)) as upstream:
            selector = selectors.DefaultSelector()
            selector.register(self.request, selectors.EVENT_READ, upstream)
            selector.register(upstream, selectors.EVENT_READ, self.request)
            while True:
                events = selector.select()
                for key, _ in events:
                    data = key.fileobj.recv(65536)
                    if not data:
                        return
                    key.data.sendall(data)


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose a local TCP service to Docker bridge containers.")
    parser.add_argument("--listen-host", default="172.17.0.1")
    parser.add_argument("--listen-port", type=int, default=7898)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=7897)
    args = parser.parse_args()

    handler = type(
        "ConfiguredForwardHandler",
        (ForwardHandler,),
        {"target_host": args.target_host, "target_port": args.target_port},
    )
    with ThreadingTCPServer((args.listen_host, args.listen_port), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
