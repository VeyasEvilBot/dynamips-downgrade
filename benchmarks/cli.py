#!/usr/bin/env python3
"""Measure real C7200 IOS launch, banner, RETURN and Router> milestones."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import select
import signal
import socket
import statistics
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROMPT = re.compile(rb"(?:^|[\r\n])Router>[ \t]*(?:$|[\r\n])")


def boot(args, number):
    prefix = args.output / f"{args.label}-{number}"
    output = bytearray()
    milestones = {}
    sent = set()
    peer = None
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with tempfile.TemporaryDirectory(prefix="dynamips-cli-bench-", dir="/tmp") as work:
        address = Path(work) / "console.sock"
        if args.core == "legacy":
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            command = [str(args.binary), "-P", "7200", "-t", "npe-400", "-r", "256",
                       "-X", "--noctrl", "-T", str(port), str(args.image)]
        else:
            command = [str(args.binary)]
            command += (["run", str(args.image)] if args.core == "cli"
                        else ["--ios", str(args.image)])
            command += ["--rom", str(args.rom), "--console", str(address)]
        with Path(str(prefix) + "-host.log").open("wb") as host:
            started = time.monotonic()
            process = subprocess.Popen(command, cwd=work, stdout=host, stderr=host,
                                       start_new_session=True)
            try:
                while time.monotonic() - started < min(10, args.timeout):
                    if process.poll() is not None:
                        break
                    try:
                        if args.core == "legacy":
                            peer = socket.create_connection(("127.0.0.1", port), timeout=0.1)
                        else:
                            peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                            peer.settimeout(0.1)
                            peer.connect(str(address))
                        peer.settimeout(1)
                        break
                    except OSError:
                        if peer is not None:
                            peer.close()
                            peer = None
                        time.sleep(0.01)
                if peer is None:
                    raise RuntimeError("console did not open before exit/timeout")
                while time.monotonic() - started < args.timeout:
                    if select.select([peer], [], [], 0.1)[0]:
                        data = peer.recv(65536)
                        if not data:
                            break
                        output.extend(data)
                    elapsed = time.monotonic() - started
                    text = bytes(output)
                    if "banner_seconds" not in milestones and b"Cisco IOS Software" in text:
                        milestones["banner_seconds"] = elapsed
                    if "return_seconds" not in milestones and b"Press RETURN" in text:
                        milestones["return_seconds"] = elapsed
                    for name, pattern, reply in [
                        ("setup", b"initial configuration dialog", b"no\r"),
                        ("autoinstall", b"terminate autoinstall", b"yes\r"),
                        ("return", b"press return", b"\r"),
                    ]:
                        if name not in sent and pattern in text.lower():
                            peer.sendall(reply)
                            sent.add(name)
                    if "startup" not in sent and "banner_seconds" in milestones and elapsed > 5:
                        peer.sendall(b"\r")
                        sent.add("startup")
                    if PROMPT.search(text):
                        milestones["prompt_seconds"] = elapsed
                        break
                    if process.poll() is not None:
                        break
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                if peer is not None:
                    peer.close()
                Path(str(prefix) + "-console.log").write_bytes(output)
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if "prompt_seconds" not in milestones or "banner_seconds" not in milestones:
        raise RuntimeError(f"IOS banner and Router> not reached; see {prefix}-console.log and {prefix}-host.log")
    milestones["banner_to_prompt_seconds"] = (milestones["prompt_seconds"]
                                                - milestones["banner_seconds"])
    return {**milestones,
            "cpu_seconds": (usage_after.ru_utime + usage_after.ru_stime
                            - usage_before.ru_utime - usage_before.ru_stime),
            "exit_code": process.returncode,
            "console_log": str(prefix) + "-console.log",
            "host_log": str(prefix) + "-host.log"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--core", choices=["legacy", "boot", "cli"], required=True)
    parser.add_argument("--image", type=Path, required=True, help="local extracted IOS ELF")
    parser.add_argument("--rom", type=Path, default=ROOT / "stable/mips64_microcode")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--label", default="cli")
    parser.add_argument("--output", type=Path, default=ROOT / "build/cli-bench")
    parser.add_argument("--cpu", type=int, help="CPU affinity inherited by native workers")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.runs < 1 or not 0 < args.timeout <= 600 or Path(args.label).name != args.label:
        parser.error("use positive runs, timeout <=600 seconds, and a simple label")
    for name in ("binary", "image", "rom", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    args.output.mkdir(parents=True, exist_ok=True)
    with args.image.open("rb") as image:
        digest = hashlib.file_digest(image, "sha256").hexdigest()
    report = {"core": args.core, "binary": str(args.binary), "image_sha256": digest,
              "cpu": args.cpu, "model": "C7200 NPE-400", "ram_megabytes": 256,
              "samples": []}
    for number in range(1, args.runs + 1):
        sample = boot(args, number)
        report["samples"].append(sample)
        print(json.dumps(sample), flush=True)
        (args.output / f"{args.label}.json").write_text(json.dumps(report, indent=2) + "\n")
    for field in ("banner_seconds", "return_seconds", "prompt_seconds", "banner_to_prompt_seconds",
                  "cpu_seconds"):
        values = [sample[field] for sample in report["samples"] if field in sample]
        if values:
            report["median_" + field] = statistics.median(values)
    (args.output / f"{args.label}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Median launch to Router>: {report['median_prompt_seconds']:.3f} s")
    print(f"Median banner to Router>: {report['median_banner_to_prompt_seconds']:.3f} s")


if __name__ == "__main__":
    main()
