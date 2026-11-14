#!/usr/bin/env python3
"""Exercise the standalone core's CLI using a user-supplied real IOS image."""

import argparse
from pathlib import Path
import select
import signal
import socket
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    transcript = bytearray()
    sent = set()
    success = False
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ios-cli-", dir="/tmp") as temporary:
        address = Path(temporary) / "console.sock"
        with (args.output / "stderr.log").open("wb") as diagnostics:
            process = subprocess.Popen(
                [str(args.binary.resolve()), "--rom", str(args.rom.resolve()),
                 "--ios", str(args.ios.resolve()), "--console", str(address),
                 *(["--dump-state"] if args.dump_state else [])],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=diagnostics,
            )
            try:
                while not address.exists():
                    if process.poll() is not None:
                        raise RuntimeError("emulator exited before opening console")
                    if time.monotonic() - started > min(15, args.timeout):
                        raise TimeoutError("console creation timed out")
                    time.sleep(0.01)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
                    peer.settimeout(1)
                    peer.connect(str(address))
                    while time.monotonic() - started < args.timeout:
                        readable, _, _ = select.select([peer], [], [], 0.2)
                        if readable:
                            data = peer.recv(65536)
                            if not data:
                                break
                            transcript.extend(data)
                            (args.output / "console.log").write_bytes(transcript)
                        recent = bytes(transcript[-16384:])
                        actions = [
                            ("setup", b"initial configuration dialog", b"no\r"),
                            ("autoinstall", b"terminate autoinstall", b"yes\r"),
                            ("return", b"press return to get started", b"\r"),
                            ("enable", b"router>", b"enable\r"),
                        ]
                        for name, pattern, reply in actions:
                            if name not in sent and pattern in recent.lower():
                                peer.sendall(reply)
                                sent.add(name)
                        if ("startup-return" not in sent
                                and b"Cisco IOS Software" in recent
                                and time.monotonic() - started > 5):
                            peer.sendall(b"\r")
                            sent.add("startup-return")
                        if b"Router#" in recent and "commands" not in sent:
                            peer.sendall(b"terminal length 0\rshow version\r"
                                         b"show ip interface brief\r")
                            sent.add("commands")
                        if ("commands" in sent and b"Router#" in recent
                                and b"uptime is" in recent
                                and b"Configuration register" in recent
                                and b"Interface" in recent
                                and b"IP-Address" in recent
                                and recent.rfind(b"Router#") > recent.rfind(b"IP-Address")):
                            success = True
                            break
                        if process.poll() is not None:
                            break
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                (args.output / "console.log").write_bytes(transcript)
    elapsed = time.monotonic() - started
    if not success or process.returncode != 0:
        raise RuntimeError(
            f"IOS CLI failed after {elapsed:.1f}s (exit {process.returncode}); "
            f"see {args.output.resolve()}"
        )
    print(f"IOS CLI: Router#, show version and interface table passed ({elapsed:.1f}s)")
    print(f"Transcript: {(args.output / 'console.log').resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--ios", type=Path, required=True)
    parser.add_argument("--rom", type=Path, default=ROOT / "stable/mips64_microcode")
    parser.add_argument("--output", type=Path, default=ROOT / "build/ios-cli")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--dump-state", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    run(args)


if __name__ == "__main__":
    main()
