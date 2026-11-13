#!/usr/bin/env python3
"""Interleave new and legacy process starts against the same IOS banner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

from boot import ROOT, boot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new", type=Path)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "build/boot-compare")
    parser.add_argument("--runs", type=int, default=7, help="starts per core")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.runs < 1 or args.timeout <= 0:
        parser.error("runs and timeout must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.image = args.image.resolve()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    with args.image.open("rb") as image:
        digest = hashlib.file_digest(image, "sha256").hexdigest()
    report = {"image": str(args.image), "image_sha256": digest, "cpu": args.cpu,
              "profiled": False, "binaries": {core: str(getattr(args, core).resolve()) for core in ("new", "legacy")}, "samples": {"new": [], "legacy": []}}
    for number in range(1, args.runs + 1):
        # Reverse each pair to reduce systematic ordering and warm-up bias.
        order = ("new", "legacy") if number % 2 else ("legacy", "new")
        for core in order:
            run = SimpleNamespace(binary=getattr(args, core).resolve(), core=core,
                                  image=args.image, output=args.output, label=core,
                                  profiler=None, timeout=args.timeout)
            sample = boot(run, number)
            report["samples"][core].append(sample)
            print(json.dumps({"core": core, "run": number, **sample}), flush=True)
    report["median_seconds"] = {
        core: statistics.median(sample["banner_seconds"] for sample in samples)
        for core, samples in report["samples"].items()}
    report["legacy_over_new"] = report["median_seconds"]["legacy"] / report["median_seconds"]["new"]
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"median_seconds": report["median_seconds"],
                      "legacy_over_new": report["legacy_over_new"]}), flush=True)


if __name__ == "__main__":
    main()
