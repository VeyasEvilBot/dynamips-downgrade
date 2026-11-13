#!/usr/bin/env python3
"""Measure process launch to the IOS banner using the same extracted ELF."""

import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import selectors
import signal
import socket
import statistics
import subprocess
import tempfile
import time

BANNER = b"Cisco IOS Software, 7200 Software"
ROOT = Path(__file__).resolve().parents[1]


def boot(args, number):
    env = os.environ.copy()
    prefix = args.output / f"{args.label}-{number}"
    if args.profiler:
        env["LD_PRELOAD"] = str(args.profiler)
        env["CPUPROFILE"] = str(prefix) + ".prof"
        env["CPUPROFILE_FREQUENCY"] = "250"
    command = [str(args.binary)]
    if args.core == "legacy":
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        command += ["-P", "7200", "-t", "npe-400", "-r", "256", "-X",
                    "--noctrl", "-T", str(port), str(args.image)]
    else:
        command += ["--rom", str(ROOT / "stable/mips64_microcode"),
                    "--ios", str(args.image), "--budget", "2000000000"]
    output = bytearray()
    banner = None
    console = None
    with tempfile.TemporaryDirectory(prefix="dynamips-bench-") as work:
        started = time.monotonic()
        process = subprocess.Popen(
            ["stdbuf", "-oL", "-eL", *command], cwd=work, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
            start_new_session=True)
        try:
            with selectors.DefaultSelector() as events:
                events.register(process.stdout, selectors.EVENT_READ)
                if args.core == "legacy":
                    while time.monotonic() - started < 5:
                        try:
                            console = socket.create_connection(("127.0.0.1", port), timeout=0.1)
                            events.register(console, selectors.EVENT_READ)
                            break
                        except OSError:
                            if process.poll() is not None:
                                break
                            time.sleep(0.02)
                    if console is None:
                        raise RuntimeError("legacy TCP console did not start")
                while time.monotonic() - started < args.timeout:
                    for key, _ in events.select(0.2):
                        data = os.read(key.fileobj.fileno(), 65536)
                        if not data:
                            events.unregister(key.fileobj)
                            continue
                        output.extend(data)
                        if banner is None and BANNER in output:
                            banner = time.monotonic() - started
                    if banner is not None and args.core == "legacy":
                        break
                    if not events.get_map():
                        break
                if banner is None:
                    raise RuntimeError("IOS banner not reached before exit/timeout")
                if args.core == "new" and process.poll() is None:
                    # Allow normal exit to flush the optional userspace profile.
                    process.wait(timeout=2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            output.extend(process.stdout.read())
            process.stdout.close()
            process.stdin.close()
            if console is not None:
                console.close()
            Path(str(prefix) + ".log").write_bytes(output)
    return {"banner_seconds": banner, "exit_code": process.returncode,
            "log": str(Path(str(prefix) + ".log"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--image", type=Path, required=True, help="extracted IOS ELF, not ZIP")
    parser.add_argument("--core", choices=["new", "legacy"], default="new")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--label", default="boot")
    parser.add_argument("--output", type=Path, default=ROOT / "build/boot-bench")
    parser.add_argument("--cpu", type=int, help="pin emulator and worker threads to this CPU")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--profiler", type=Path, help="libprofiler.so for a separate sampled run")
    args = parser.parse_args()
    if args.runs < 1 or args.timeout <= 0 or Path(args.label).name != args.label:
        parser.error("use positive runs/timeout and a simple output label")
    if args.profiler and args.core != "new":
        parser.error("profile only the new core, which exits normally after its boot checkpoint")
    for name in ("binary", "image", "output", "profiler"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    with args.image.open("rb") as image:
        digest = hashlib.file_digest(image, "sha256").hexdigest()
    report = {"core": args.core, "binary": str(args.binary), "image": str(args.image),
              "image_sha256": digest,
              "cpu": args.cpu, "profiled": bool(args.profiler), "samples": []}
    for number in range(1, args.runs + 1):
        sample = boot(args, number)
        report["samples"].append(sample)
        print(json.dumps(sample), flush=True)
    report["median_banner_seconds"] = statistics.median(s["banner_seconds"] for s in report["samples"])
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    (args.output / f"{args.label}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Median to banner: {report['median_banner_seconds']:.3f} s")


if __name__ == "__main__":
    main()
