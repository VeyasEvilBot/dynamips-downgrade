#!/usr/bin/env python3
"""Check direct native terminal transport without a socket or helper process."""

import argparse
import fcntl
import os
from pathlib import Path
import pty
import select
import subprocess
import termios
import threading
import time


def read_exact(pipe, count):
    output = bytearray()
    deadline = time.monotonic() + 15
    while len(output) < count:
        remaining = deadline - time.monotonic()
        assert remaining > 0 and select.select([pipe], [], [], remaining)[0], "terminal output timed out"
        data = os.read(pipe.fileno(), count - len(output))
        assert data, "terminal output ended early"
        output.extend(data)
    return bytes(output)


def pipes(binary):
    payload = bytes(range(256)) * 2048
    child = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    failures = []

    def send():
        try:
            child.stdin.write(payload)
            child.stdin.close()
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    try:
        time.sleep(0.1)  # Fill bounded queues before draining the slow reader.
        output = read_exact(child.stdout, len(payload))
        thread.join(timeout=10)
        assert not thread.is_alive() and not failures, failures
        assert output == payload, "binary pipe bytes changed"
        assert child.wait(timeout=10) == 0, child.stderr.read()
        assert child.stdout.read() == b"", "unexpected trailing pipe output"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        thread.join(timeout=10)
        child.stdout.close()
        child.stderr.close()


def terminal(binary, terminate=False):
    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    flags = fcntl.fcntl(slave, fcntl.F_GETFL)
    child = subprocess.Popen([str(binary)], stdin=slave,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while termios.tcgetattr(slave)[3] & termios.ICANON:
            assert child.poll() is None, "native terminal exited before raw mode"
            assert time.monotonic() < deadline, "native terminal did not become raw"
            time.sleep(0.01)
        if terminate:
            child.terminate()
        else:
            os.write(master, b"\x03ABC")
            assert read_exact(child.stdout, 4) == b"\x03ABC", "raw Ctrl-C did not reach guest input"
            os.write(master, b"\x1d")
        output, error = child.communicate(timeout=10)
        assert child.returncode == 0, error
        assert output == b"", "terminal escape leaked to guest output"
        assert termios.tcgetattr(slave) == original, "terminal attributes were not restored"
        assert fcntl.fcntl(slave, fcntl.F_GETFL) == flags, "borrowed descriptor flags were not restored"
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)
        os.close(master)
        os.close(slave)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    pipes(binary)
    terminal(binary)
    terminal(binary, terminate=True)
    print("native terminal: binary pipes, backpressure, EOF, Ctrl-C, escape, signal and restoration passed")


if __name__ == "__main__":
    main()
