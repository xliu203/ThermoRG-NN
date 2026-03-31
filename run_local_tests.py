#!/usr/bin/env python3
"""
run_local_tests.py - Quick local test suite (no GPU, no CIFAR-10 download)
Run before pushing to GitHub to catch trivial errors and avoid wasting GPU quota.
"""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

TESTS = [
    ("Architecture forward (all 15 models)", "python3 tests/test_architectures.py"),
    ("TAS metrics computation",              "python3 tests/test_tas_metrics.py"),
    ("Feature extractor hooks",              "python3 tests/test_feature_extractor.py"),
]

def run_tests():
    failed = []
    for desc, cmd in TESTS:
        print(f"\n{'='*60}")
        print(f"Running: {desc}")
        print(f"Command: {cmd}")
        print('='*60)
        r = subprocess.run(cmd, shell=True)
        if r.returncode != 0:
            print(f"FAILED: {desc}")
            failed.append(desc)
        else:
            print(f"PASSED: {desc}")
    print(f"\n{'='*60}")
    if failed:
        print(f"\n❌ {len(failed)} test(s) failed:")
        for f in failed: print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
