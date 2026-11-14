#!/usr/bin/env python3
"""Exercise the standalone core's CLI using a user-supplied real IOS image."""

import argparse
import contextlib
import json
import os
import resource
from pathlib import Path
import select
import signal
import socket
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


class pipe_console:
    def __init__(self, process):
        self.process = process
    def fileno(self):
        return self.process.stdout.fileno()
    def recv(self, size):
        return os.read(self.fileno(), size)
    def sendall(self, data):
        self.process.stdin.write(data)
        self.process.stdin.flush()


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    transcript = bytearray()
    sent = set()
    success = False
    metrics = {}
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ios-cli-", dir="/tmp") as temporary:
        address = Path(temporary) / "console.sock"
        with (args.output / "stderr.log").open("wb") as diagnostics:
            command = [str(args.binary.resolve())]
            if args.frontend == "cli":
                command += ["run", str(args.ios.resolve())]
            else:
                command += ["--ios", str(args.ios.resolve())]
            command += ["--rom", str(args.rom.resolve())]
            command += ["--stdio"] if args.stdio else ["--console", str(address)]
            if args.dump_state:
                command.append("--dump-state")
            process = subprocess.Popen(
                command, cwd=ROOT,
                stdin=subprocess.PIPE if args.stdio else subprocess.DEVNULL,
                stdout=subprocess.PIPE if args.stdio else subprocess.DEVNULL,
                stderr=diagnostics,
            )
            try:
                while not args.stdio and not address.exists():
                    if process.poll() is not None:
                        raise RuntimeError("emulator exited before opening console")
                    if time.monotonic() - started > min(15, args.timeout):
                        raise TimeoutError("console creation timed out")
                    time.sleep(0.01)
                connection = (contextlib.nullcontext(pipe_console(process)) if args.stdio
                              else socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
                with connection as peer:
                    if not args.stdio:
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
                        for name, marker in [("banner_seconds", b"Cisco IOS Software"),
                                             ("return_seconds", b"Press RETURN"),
                                             ("prompt_seconds", b"Router>")]:
                            if marker in recent and name not in metrics:
                                metrics[name] = time.monotonic() - started
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
                            metrics["commands_seconds"] = time.monotonic() - started
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
                if args.stdio:
                    process.stdin.close()
                    process.stdout.close()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    metrics["cpu_seconds"] = usage.ru_utime + usage.ru_stime - usage_before.ru_utime - usage_before.ru_stime
    metrics["success"] = success
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
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
    parser.add_argument("--frontend", choices=["boot", "cli"], default="boot")
    parser.add_argument("--stdio", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    run(args)


if __name__ == "__main__":
    main()
