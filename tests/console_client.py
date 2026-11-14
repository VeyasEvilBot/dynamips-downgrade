#!/usr/bin/env python3
"""Check binary pipes, half-close and raw-terminal cleanup in tools/console.py."""

import argparse
import os
from pathlib import Path
import pty
import select
import socket
import subprocess
import sys
import tempfile
import termios
import threading
import time


def listener(path):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.settimeout(10)
    server.bind(str(path))
    server.listen(1)
    return server


def worker(server, handler, errors):
    try:
        with server.accept()[0] as connection:
            connection.settimeout(10)
            handler(connection)
    except Exception as error:
        errors.append(error)


def read_exact(fd, size):
    output = bytearray()
    deadline = time.monotonic() + 10
    while len(output) < size:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "client output timed out"
        assert select.select([fd], [], [], remaining)[0], "client output timed out"
        data = os.read(fd, size - len(output))
        assert data, "client output ended early"
        output.extend(data)
    return bytes(output)


def piped(client, path):
    payload = bytes(range(256)) * 2048
    tail = b"\x00after input EOF\xff"
    errors = []

    def serve(connection):
        while data := connection.recv(4096):
            connection.sendall(data)
        connection.sendall(tail)

    with listener(path) as server:
        thread = threading.Thread(target=worker, args=(server, serve, errors), daemon=True)
        thread.start()
        result = subprocess.run(
            [sys.executable, str(client), str(path)], input=payload,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        thread.join(timeout=12)
        assert not thread.is_alive(), "pipe server did not stop"
    assert not errors, errors
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == payload + tail, "piped bytes or EOF response changed"


def terminal(client, path, terminate=False):
    errors = []
    received = bytearray()

    def serve(connection):
        while data := connection.recv(4096):
            received.extend(data)
            connection.sendall(data)

    master, slave = pty.openpty()
    attributes = termios.tcgetattr(slave)
    process = None
    try:
        with listener(path) as server:
            thread = threading.Thread(target=worker, args=(server, serve, errors), daemon=True)
            thread.start()
            process = subprocess.Popen(
                [sys.executable, str(client), str(path)], stdin=slave,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 10
            while termios.tcgetattr(slave)[3] & termios.ICANON:
                assert process.poll() is None, "terminal client exited before raw mode"
                assert time.monotonic() < deadline, "raw terminal setup timed out"
                time.sleep(0.01)
            if terminate:
                process.terminate()
            else:
                os.write(master, b"\x03A")
                assert read_exact(process.stdout.fileno(), 2) == b"\x03A", "Ctrl-C was not forwarded"
                os.write(master, b"\x1d")
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == (143 if terminate else 0), stderr.decode(errors="replace")
            assert stdout == b"", "unexpected trailing client output"
            assert termios.tcgetattr(slave) == attributes, "terminal attributes were not restored"
            thread.join(timeout=12)
            assert not thread.is_alive(), "terminal server did not stop"
        assert not errors, errors
        assert received == (b"" if terminate else b"\x03A"), "escape leaked to the guest"
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
        os.close(master)
        os.close(slave)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "tools/console.py")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="console-client-", dir="/tmp") as directory:
        root = Path(directory)
        client = args.client.resolve()
        piped(client, root / "pipe.sock")
        terminal(client, root / "tty.sock")
        terminal(client, root / "signal.sock", terminate=True)
    print("console client: binary pipes, half-close, Ctrl-C/escape and terminal cleanup passed")


if __name__ == "__main__":
    main()
