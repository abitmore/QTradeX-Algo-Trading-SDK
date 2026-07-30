"""
Integration test suite for QTradeX SDK.
Runs extinction_event.py through every optimizer and tune type via the CLI dispatch path.
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BOT_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "demos", "extinction_event.py")
TIMEOUT = 15


def run(cmd, desc, expect_code=0):
    env = os.environ.copy()
    env["QTRADEX_NONINTERACTIVE"] = "1"
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, BOT_SCRIPT] + cmd,
            capture_output=True, text=True, timeout=60, env=env,
        )
    except subprocess.TimeoutExpired:
        return (desc, "TIMEOUT", time.time() - start)
    elapsed = time.time() - start

    if proc.returncode == expect_code:
        return (desc, "PASS", elapsed)

    stderr_tail = proc.stderr.strip().split("\n")[-5:] if proc.stderr else []
    stdout_tail = proc.stdout.strip().split("\n")[-3:] if proc.stdout else []
    tail = (stderr_tail + stdout_tail)[-5:]
    return (desc, "FAIL", elapsed, proc.returncode, tail)


def banner(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print()


def print_results(results):
    passed = 0
    failed = 0
    print()
    print("=" * 72)
    print("  RESULTS")
    print("=" * 72)
    for r in results:
        desc, status, elapsed = r[0], r[1], r[2]
        if status == "PASS":
            passed += 1
            print(f"  PASS  [{elapsed:5.1f}s]  {desc}")
        else:
            failed += 1
            rc, tail = r[3], r[4]
            print(f"  FAIL  [{elapsed:5.1f}s]  {desc}  (exit {rc})")
            if tail:
                for line in tail:
                    print(f"        {line}")
    print()
    print(f"  {passed} passed, {failed} failed")
    print()
    return failed


# ── Sequential preamble ──────────────────────────────────────────────────────

results = []
banner("Smoke & seed")
results.append(run(["--help"], "help flag"))
results.append(run(["--tune=bot", "--optimize=GridSearch", "--timeout=5"],
    "seed: GridSearch 5s (saves tune)"))

# ── Parallel tests ───────────────────────────────────────────────────────────

banner("All tests (parallel)")

test_cases = []
for tt in ["bot", "drop", "best", "latest"]:
    test_cases.append(([f"--tune={tt}", "--backtest"], f"backtest tune={tt}"))
for opt in ["QPSO", "LSGA", "IPSE", "AION", "GridSearch", "RL"]:
    test_cases.append(
        (["--tune=bot", f"--optimize={opt}", f"--timeout={TIMEOUT}"], f"optimizer {opt}")
    )

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(run, cmd, desc) for cmd, desc in test_cases]
    for f in as_completed(futures):
        results.append(f.result())

sys.exit(print_results(results))
