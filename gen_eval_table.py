#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path


BASE_COLUMNS = [
    "epoch",
    "valid_lattice_xe_loss",
    "valid_lattice_beam_acc",
    "valid_lattice_perfect",
    "valid_lattice_correct",
    "valid_lattice_bitwise_acc",
    "valid_lattice_percs_diff",
    "valid_lattice_beam_acc_1",
    "secret_eval_count",
    "secret_perfect_count",
    "secret_best_match",
    "secret_best_total",
    "secret_best_ratio",
]


def _summarize_secret(entries):
    eval_count = len(entries)
    perfect_count = 0
    best_match = None
    best_total = None
    best_ratio = None

    for entry in entries:
        matched = entry.get("matched")
        total = entry.get("total")
        is_perfect = entry.get("perfect") is True
        if matched is not None and total:
            if matched == total:
                is_perfect = True
            ratio = matched / total
            if (
                best_ratio is None
                or ratio > best_ratio
                or (ratio == best_ratio and matched > best_match)
            ):
                best_ratio = ratio
                best_match = matched
                best_total = total
        if is_perfect:
            perfect_count += 1

    return {
        "secret_eval_count": eval_count,
        "secret_perfect_count": perfect_count,
        "secret_best_match": best_match,
        "secret_best_total": best_total,
        "secret_best_ratio": best_ratio,
    }


def parse_log(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    rows = []
    secret_entries = []
    n_value = None

    re_n = re.compile(r"^\s*N:\s*(\d+)\s*$")
    re_secret = re.compile(
        r"Secret matching(?: \(s'\))?:\s*(\d+)/(\d+)\s*(?:bits|entries) matched.*K=([0-9]+)"
    )
    re_all = re.compile(r"All (?:bits in secret|entries in s') .*K=([0-9]+)")

    for line in log_path.read_text().splitlines():
        if n_value is None:
            match_n = re_n.match(line)
            if match_n:
                n_value = int(match_n.group(1))

        match_secret = re_secret.search(line)
        if match_secret:
            matched = int(match_secret.group(1))
            total = int(match_secret.group(2))
            k_val = int(match_secret.group(3))
            secret_entries.append(
                {"matched": matched, "total": total, "k": k_val, "perfect": matched == total}
            )
            continue

        match_all = re_all.search(line)
        if match_all:
            k_val = int(match_all.group(1))
            matched = total = n_value if n_value is not None else None
            secret_entries.append(
                {"matched": matched, "total": total, "k": k_val, "perfect": True}
            )
            continue

        if "__log__:" not in line:
            continue

        payload = line.split("__log__:", 1)[1].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        row = dict(data)
        row.update(_summarize_secret(secret_entries))
        rows.append(row)
        secret_entries = []

    return rows


def build_header(rows):
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    extra = [k for k in sorted(all_keys) if k not in BASE_COLUMNS]
    return [k for k in BASE_COLUMNS if k in all_keys] + extra


def main():
    parser = argparse.ArgumentParser(
        description="Generate a per-epoch metrics table from train.log"
    )
    parser.add_argument("--log", required=True, help="Path to train.log")
    parser.add_argument("--out", default="", help="Optional CSV output path")
    args = parser.parse_args()

    rows = parse_log(args.log)
    if not rows:
        raise SystemExit("No __log__ entries found.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            header = build_header(rows)
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        header = build_header(rows)
        writer = csv.DictWriter(sys.stdout, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
