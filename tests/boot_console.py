#!/usr/bin/env python3
"""Exercise the standalone frontend through its production Unix console."""

import argparse
from pathlib import Path
import signal
import select
import socket
import struct
import subprocess
import tempfile
import threading
import time


def write_echo_rom(path):
    # Take a real TX interrupt, then poll the FPGA UART and echo each byte.
    words = (
        0x3C08BE84,  # LUI t0, kernel alias of I/O FPGA
        0x340A0005,  # Enable FPGA UART receiver and transmitter.
        0xAD0A0414,
        0x3C090040,  # BEV exception vector, external input 5 and IE.
        0x35292001,
        0x40896000,
        0x340A0001,
        0xAD0A042C,  # Enable TX-ready IRQ; handler prints '!'.
        0x340A003E,  # ORI t2, zero, '>'
        0xAD0A041C,  # SW t2, TX(t0)
        0x8D09040C,  # LW t1, status(t0)
        0x31290001,  # ANDI t1, t1, SCN2681 RX-ready
        0x1120FFFD,  # BEQ t1, zero, poll
        0,
        0x8D0A041C,  # LW t2, RX(t0)
        0xAD0A041C,  # SW t2, TX(t0)
        0x1000FFF9,  # B poll
        0,
    )
    handler = (0x341A0021, 0xAD1A041C, 0xAD00042C, 0x42000018)
    code = struct.pack(">" + "I" * len(words), *words).ljust(0x380, b"\0")
    code += struct.pack(">4I", *handler)
    ident = b"\x7fELF\x01\x02\x01" + bytes(9)
    header = struct.pack(
        ">16sHHIIIIIHHHHHH", ident, 2, 8, 1, 0xBFC00000,
        0, 0x1000, 0, 52, 0, 0, 40, 2, 0,
    )
    section = struct.pack(">10I", 0, 1, 6, 0xBFC00000, 0x80, len(code), 0, 0, 4, 0)
    contents = header.ljust(0x80, b"\0") + code
    path.write_bytes(contents.ljust(0x1000, b"\0") + bytes(40) + section)


def wait_for_socket(process, path):
    deadline = time.monotonic() + 10
    diagnostics = bytearray()
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"boot exited {process.returncode} before console: {stdout!r} {bytes(diagnostics) + stderr!r}")
        readable, _, _ = select.select([process.stderr], [], [], 0.01)
        if readable:
            line = process.stderr.readline()
            diagnostics.extend(line)
            if line.startswith(b"Console socket:"):
                assert path.exists(), "console notice preceded socket creation"
                return
    raise AssertionError("boot console listener did not open")


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    return process.communicate(timeout=10)


def echo(binary, rom, path):
    process = subprocess.Popen(
        [str(binary), "--rom", str(rom), "--console", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    peer.settimeout(10)
    sender = None
    errors = []
    try:
        wait_for_socket(process, path)
        time.sleep(0.05)
        peer.connect(str(path))
        payload = bytes(range(256)) * 1024

        def send():
            try:
                peer.sendall(payload)
                peer.shutdown(socket.SHUT_WR)
            except OSError as error:
                errors.append(error)

        sender = threading.Thread(target=send, daemon=True)
        sender.start()
        # Let the bounded transport and device queues encounter a slow reader.
        time.sleep(0.05)
        expected = b"!>" + payload
        output = bytearray()
        deadline = time.monotonic() + 30
        while len(output) < len(expected) and time.monotonic() < deadline:
            data = peer.recv(min(17, len(expected) - len(output)))
            if not data:
                break
            output.extend(data)
        sender.join(timeout=5)
        assert not sender.is_alive(), "console sender did not finish"
        assert not errors, errors
        assert output == expected, "console lost or reordered prompt/input/output bytes"
        stdout, stderr = stop(process)
        assert process.returncode == 0, stderr.decode(errors="replace")
        assert stdout == b"", "socket console output leaked to stdout"
        assert not path.exists(), "console socket survived orderly shutdown"
    finally:
        peer.close()
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)
        if sender:
            sender.join(timeout=5)


def stop_before_accept(binary, rom, path):
    process = subprocess.Popen(
        [str(binary), "--rom", str(rom), "--console", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        wait_for_socket(process, path)
        stdout, stderr = stop(process)
        assert process.returncode == 0, stderr.decode(errors="replace")
        assert stdout == b"", "guest ran before its console connected"
        assert not path.exists(), "unaccepted console socket survived shutdown"
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix="boot-console-", dir="/tmp") as directory:
        root = Path(directory)
        rom = root / "echo.elf"
        write_echo_rom(rom)
        stop_before_accept(binary, rom, root / "waiting.sock")
        echo(binary, rom, root / "echo.sock")
    print("boot console: binary echo, half-close, backpressure and shutdown passed")


if __name__ == "__main__":
    main()
