#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse LIBERO evaluation log files and calculate average success rate across tasks.

Log format expected:
  - Task names appear after "Task: ..." (may span multiple lines)
  - Per-task success rate: "Current task success rate: X.XX" (value on same or next line)
  - Final line: "Total success rate: X.XX"

Usage:
  python utils/calculate_libero_success_rate.py --log_dir logs/libero_batch_20260411_112838
"""

import re
import argparse
from pathlib import Path


def parse_log_file(file_path: str) -> list[tuple[str, float]]:
    """
    Parse a single LIBERO eval log file and extract per-task success rates.

    Returns:
        List of (task_name, success_rate) tuples.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # --- Extract unique task names in order ---
    # Task name appears as:
    #   "                          Task: <partial name>"
    #   "                          <continued name>"      (indented continuation)
    # until a line with a timestamp like "04/11 [HH:MM:SS]" or progress bar.
    task_names = []
    i = 0
    while i < len(lines):
        m = re.search(r"Task:\s*(.+)", lines[i])
        if m:
            parts = [m.group(1).strip()]
            # Collect continuation lines (indented, no timestamp, no INFO)
            j = i + 1
            while j < len(lines):
                line = lines[j]
                # Stop at timestamp lines, INFO lines, progress bars, or empty lines
                if re.match(r"\d{2}/\d{2}\s+\[", line) or "INFO" in line or line.strip() == "":
                    break
                parts.append(line.strip())
                j += 1
            name = " ".join(parts)
            task_names.append(name)
            i = j
        else:
            i += 1

    # De-duplicate task names, keep unique in order
    seen = set()
    unique_tasks = []
    for name in task_names:
        if name not in seen:
            seen.add(name)
            unique_tasks.append(name)

    # --- Extract per-task success rates ---
    # "Current task success rate: 0.92" — value on same line or next line
    content = "".join(lines)
    task_rates = []
    for m in re.finditer(
        r"Current task success rate:\s*([\d.]+)?(?:\s*eval_libero\.py:\d+\s*\n\s*([\d.]+))?",
        content,
    ):
        val = m.group(1) or m.group(2)
        if val:
            task_rates.append(float(val))

    return list(zip(unique_tasks, task_rates))


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average success rate from LIBERO eval log files."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing eval_libero_*.log files",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"Error: {log_dir} is not a valid directory")
        return

    eval_files = sorted(log_dir.glob("eval_libero_*.log"))
    if not eval_files:
        print(f"Error: no eval_libero_*.log files found in {log_dir}")
        return

    print(f"Found {len(eval_files)} eval log file(s)\n")

    all_results = []  # (suite_name, task_name, rate)

    for fpath in eval_files:
        suite_name = fpath.stem  # e.g. eval_libero_10_gpu3
        task_results = parse_log_file(fpath)

        if not task_results:
            print(f"[WARN] No task results parsed from {fpath.name}")
            continue

        print(f"--- {suite_name} ({len(task_results)} tasks) ---")
        for task_name, rate in task_results:
            print(f"  {task_name}: {rate:.2f}")
            all_results.append((suite_name, task_name, rate))

        suite_rates = [r for _, r in task_results]
        suite_avg = sum(suite_rates) / len(suite_rates)
        print(f"  Suite avg: {suite_avg:.4f} ({suite_avg * 100:.2f}%)\n")

    if not all_results:
        print("Error: no task results found in any file")
        return

    all_rates = [r for _, _, r in all_results]
    avg = sum(all_rates) / len(all_rates)

    print("=" * 60)
    print(f"Total tasks: {len(all_rates)}")
    print(f"Average success rate: {avg:.4f} ({avg * 100:.2f}%)")
    print(f"Max: {max(all_rates):.4f}  Min: {min(all_rates):.4f}")


if __name__ == "__main__":
    main()
