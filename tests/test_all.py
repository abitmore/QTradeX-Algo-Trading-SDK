"""
Integration test suite for QTradeX SDK.
Runs extinction_event.py through every optimizer and tune type via the CLI dispatch path.
"""

import os
import subprocess
import sys
import time

BOT_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "demos", "extinction_event.py")
TIMEOUT = 15
results = []


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
        results.append((desc, "TIMEOUT", time.time() - start))
        return False
    elapsed = time.time() - start

    if proc.returncode == expect_code:
        results.append((desc, "PASS", elapsed))
        return True

    stderr_tail = proc.stderr.strip().split("\n")[-5:] if proc.stderr else []
    stdout_tail = proc.stdout.strip().split("\n")[-3:] if proc.stdout else []
    tail = (stderr_tail + stdout_tail)[-5:]
    results.append((desc, "FAIL", elapsed, proc.returncode, tail))
    return False


def banner(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print()


def print_results():
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
            rc = r[3]
            tail = r[4]
            print(f"  FAIL  [{elapsed:5.1f}s]  {desc}  (exit {rc})")
            if tail:
                for line in tail:
                    print(f"        {line}")
    print()
    print(f"  {passed} passed, {failed} failed")
    print()
    return failed


# ── Smoke test ───────────────────────────────────────────────────────────────

banner("Smoke test")
run(["--help"], "help flag")

# ── Seed a saved tune so best/latest can load ────────────────────────────────

banner("Seed saved tune")
run(["--tune=bot", "--backtest"], "seed: backtest bot (creates no tune)")
run(["--tune=bot", "--optimize=GridSearch", "--timeout=5"],
    "seed: GridSearch 5s (saves tune)")

# ── Backtest with each tune type ─────────────────────────────────────────────

banner("Backtest — all tune types")
tune_types = ["bot", "drop", "best", "latest"]
for tt in tune_types:
    run([f"--tune={tt}", "--backtest"], f"backtest tune={tt}")

# ── Optimizer tests ──────────────────────────────────────────────────────────

banner("Optimizers (timeout=15s)")

optimizers = ["QPSO", "LSGA", "IPSE", "AION", "GridSearch", "RL"]
for opt in optimizers:
    run(["--tune=bot", f"--optimize={opt}", f"--timeout={TIMEOUT}"],
        f"optimizer {opt}")

# ── Summary ───────────────────────────────────────────────────────────────────

sys.exit(print_results())
