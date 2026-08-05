#!/usr/bin/env python3
"""Parse unitrace hardware-counter CSV logs.

Handles commas inside kernel names by parsing right-to-left.

Usage:
    python parse_hw_metrics.py <log_file> [options]

Options:
    --kernel <substring>   Filter to kernels matching this substring
    --skip-warmup <N>      Skip first N instances of each kernel (default: 1)
    --format <json|csv>    Output format (default: json)
    --summary              Print median summary for the dominant kernel
    --output <path>        Write output to file (default: stdout)

Examples:
    python parse_hw_metrics.py compute_basic_raw.log --kernel "AvgPool2d" --summary
    python parse_hw_metrics.py ve_profile_raw.log --format json --output parsed.json
"""

import argparse
import json
import re
import statistics
import sys


def parse_unitrace_metrics(log_text):
    """Parse unitrace metric log, handling commas in kernel names."""
    lines = log_text.splitlines()
    header_line = None
    data_start = None

    for i, line in enumerate(lines):
        if re.match(r"^=== Device #\d+ Metrics ===$", line.strip()):
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    header_line = lines[j].strip()
                    data_start = j + 1
                    break
            break

    if header_line is None:
        raise ValueError("No '=== Device #N Metrics ===' section found in log")

    header_fields = [h.strip() for h in header_line.split(",")]
    num_columns = len(header_fields)

    rows = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("=== Device #"):
            continue
        fields = line.split(",")
        if len(fields) < num_columns:
            continue
        # Right-to-left: all columns except first are numeric
        metric_values = fields[-(num_columns - 1):]
        kernel_name = ",".join(fields[: len(fields) - (num_columns - 1)])
        kernel_name = kernel_name.strip().strip('"')
        row = {"Kernel": kernel_name}
        for col_name, value in zip(header_fields[1:], metric_values):
            row[col_name.strip()] = value.strip()
        rows.append(row)

    return header_fields, rows


def filter_kernel(rows, kernel_substring):
    """Filter rows to those whose Kernel name contains the substring."""
    if not kernel_substring:
        return rows
    return [r for r in rows if kernel_substring in r["Kernel"]]


def skip_warmup(rows, n=1):
    """Skip first N instances of each kernel (by order of appearance)."""
    seen_counts = {}
    result = []
    for row in rows:
        k = row["Kernel"]
        seen_counts[k] = seen_counts.get(k, 0) + 1
        if seen_counts[k] > n:
            result.append(row)
    return result


def compute_median_summary(rows, header_fields):
    """Compute median of each numeric column across rows."""
    if not rows:
        return {}
    summary = {"Kernel": rows[0]["Kernel"], "count": len(rows)}
    for col in header_fields[1:]:
        col = col.strip()
        values = []
        for r in rows:
            try:
                values.append(float(r.get(col, "")))
            except (ValueError, TypeError):
                pass
        if values:
            summary[col] = statistics.median(values)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Parse unitrace HW metric logs")
    parser.add_argument("log_file", help="Path to raw unitrace log file")
    parser.add_argument("--kernel", default=None, help="Filter to kernel substring")
    parser.add_argument("--skip-warmup", type=int, default=1, help="Skip first N instances")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--summary", action="store_true", help="Print median summary")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    with open(args.log_file) as f:
        log_text = f.read()

    header_fields, rows = parse_unitrace_metrics(log_text)

    if args.kernel:
        rows = filter_kernel(rows, args.kernel)

    rows = skip_warmup(rows, args.skip_warmup)

    if not rows:
        print("WARNING: no rows after filtering/warmup-skip", file=sys.stderr)

    if args.summary:
        result = compute_median_summary(rows, header_fields)
    else:
        result = {"header": header_fields, "rows": rows}

    output_text = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
