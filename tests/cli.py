#!/usr/bin/env python3
"""Check production CLI help and validation before guest execution starts."""

import argparse
from pathlib import Path
import subprocess
import tempfile


def invoke(binary, *arguments):
    return subprocess.run([str(binary), *arguments], capture_output=True, timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for arguments in [("--help",), ("run", "--help")]:
        result = invoke(binary, *arguments)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert b"run" in result.stdout or b"IMAGE" in result.stdout
        assert not result.stderr, "help should use stdout"
    with tempfile.TemporaryDirectory(prefix="dynamips-cli-", dir="/tmp") as temporary:
        image = Path(temporary) / "image.elf"
        # Validation failures must be reported before attempting to load this file.
        image.write_bytes(b"not an ELF image")
        invalid = [(), ("unknown",), ("run",),
                   ("run", str(image) + ".missing"),
                   ("run", str(image), "--unknown"),
                   ("run", str(image), "--console", "/tmp/example.sock", "--stdio"),
                   ("run", str(image), "--console", ""),
                   ("run", str(image), "--rom", str(image) + ".missing")]
        invalid.extend(("run", str(image), "--budget", value)
                       for value in ["0", "-1", "1.5", str(1 << 64), "invalid"])
        for arguments in invalid:
            result = invoke(binary, *arguments)
            assert result.returncode != 0, f"accepted invalid arguments: {arguments}"
            assert result.stderr, f"missing diagnostic: {arguments}"
            assert b"firmware" not in result.stderr and b"boot:" not in result.stderr, (
                f"invalid arguments reached execution: {arguments}: {result.stderr!r}"
            )
    print("CLI: help, required arguments, console modes and budget bounds passed")


if __name__ == "__main__":
    main()
