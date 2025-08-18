#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile
from pathlib import Path

proj_dir = Path(__file__).resolve().parents[1]
formatter = 'clang-format'
src_dirs = [f"{proj_dir}/src", f"{proj_dir}/inc"]
exclude = []

with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
    files = [
        f for d in src_dirs for f in Path(d).rglob('*')
        if f.is_file()
            and f.suffix in (".cpp", ".h", ".cu", ".cuh")
            and all(xcl not in f._str for xcl in exclude)
    ]
    for f in files:
        temp.write(f"{str(f)}\n")

r = subprocess.run(
    [formatter, "--Werror", "--dry-run", f"--style=file", *files] +
    sys.argv[1:])

os.remove(temp.name)
sys.exit(r.returncode)
