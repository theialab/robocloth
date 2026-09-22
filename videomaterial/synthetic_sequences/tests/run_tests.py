#!/usr/bin/env python3
"""Tiny test runner (the render env has no pytest): python tests/run_tests.py [pattern]"""
import importlib.util, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(pattern=""):
    fails = 0
    for f in sorted(HERE.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(vars(mod)):
            if not name.startswith("test_") or pattern not in name:
                continue
            t0 = time.time()
            try:
                getattr(mod, name)()
                print(f"PASS {f.stem}.{name} ({time.time()-t0:.2f}s)")
            except Exception:
                fails += 1
                print(f"FAIL {f.stem}.{name}\n{traceback.format_exc()}")
    print("all tests passed" if not fails else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
