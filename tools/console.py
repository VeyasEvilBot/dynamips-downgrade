#!/usr/bin/env python3
"""Connect standard input/output to the standalone emulator's raw Unix console."""

import argparse
import errno
import fcntl
import os
import select
import signal
import socket
import sys
import termios
import tty

CAPACITY = 65536


def relay(connection):
    input_fd = sys.stdin.fileno()
    output_fd = sys.stdout.fileno()
    terminal = os.isatty(input_fd)
    attributes = termios.tcgetattr(input_fd) if terminal else None
    output_flags = fcntl.fcntl(output_fd, fcntl.F_GETFL)
    handlers = {}
    pending_input = bytearray()
    pending_output = bytearray()
    input_closed = False
    write_closed = False
    peer_closed = False

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    try:
        if terminal:
            for signum in (signal.SIGTERM, signal.SIGHUP):
                handlers[signum] = signal.signal(signum, terminate)
            tty.setraw(input_fd, termios.TCSANOW)
        fcntl.fcntl(output_fd, fcntl.F_SETFL, output_flags | os.O_NONBLOCK)
        connection.setblocking(False)
        while not peer_closed or pending_output:
            if input_closed and not pending_input and not write_closed:
                try:
                    connection.shutdown(socket.SHUT_WR)
                except OSError as error:
                    if error.errno not in (errno.ENOTCONN, errno.EPIPE):
                        raise
                write_closed = True

            readers = []
            writers = []
            if not input_closed and not peer_closed and len(pending_input) < CAPACITY:
                readers.append(input_fd)
            if not peer_closed and len(pending_output) < CAPACITY:
                readers.append(connection)
            if pending_input and not peer_closed:
                writers.append(connection)
            if pending_output:
                writers.append(output_fd)
            readable, writable, _ = select.select(readers, writers, [])

            if input_fd in readable:
                data = os.read(input_fd, CAPACITY - len(pending_input))
                if terminal and b"\x1d" in data:
                    return 0
                if data:
                    pending_input.extend(data)
                else:
                    input_closed = True
            if connection in readable:
                try:
                    data = connection.recv(CAPACITY - len(pending_output))
                except BlockingIOError:
                    data = None
                if data:
                    pending_output.extend(data)
                elif data == b"":
                    peer_closed = True
            if connection in writable and not peer_closed:
                try:
                    count = connection.send(pending_input)
                except BlockingIOError:
                    count = 0
                del pending_input[:count]
            if output_fd in writable:
                try:
                    count = os.write(output_fd, pending_output)
                except BlockingIOError:
                    count = 0
                del pending_output[:count]
        return 0
    finally:
        try:
            if attributes is not None:
                termios.tcsetattr(input_fd, termios.TCSANOW, attributes)
        finally:
            try:
                fcntl.fcntl(output_fd, fcntl.F_SETFL, output_flags)
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="On a terminal, Ctrl-C goes to IOS and Ctrl-] disconnects. "
        "Piped input/output remain binary; stdin EOF half-closes the connection.",
    )
    parser.add_argument("socket", help="Unix socket created by c7200-boot --console")
    args = parser.parse_args()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(args.socket)
            if sys.stdin.isatty():
                print(
                    "Connected. IOS startup can be quiet for several minutes.\n"
                    "Press Enter when IOS requests RETURN. Ctrl-C goes to IOS; "
                    "Ctrl-] disconnects.",
                    file=sys.stderr, flush=True,
                )
            return relay(connection)
    except BrokenPipeError:
        return 1
    except OSError as error:
        print(f"console: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
