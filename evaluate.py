#!/usr/bin/env python3
"""
Run the analyzer over the labelled sample logs and report how well the local model does.

Examples:
  python evaluate.py                      # every sample log, 3 runs each
  python evaluate.py --runs 1             # quick check, one run each
  python evaluate.py --category usb       # only the usb log
  python evaluate.py --model qwen3:8b --out results.json

Needs Ollama running with the model installed. Each analysis takes about a minute,
so the default (9 logs x 3 runs) takes roughly half an hour.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from config import CATEGORIES, OLLAMA_MODEL
from modules.evaluation import format_report, load_dataset, run_evaluation, summarise


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the analyzer on the labelled sample logs.")
    parser.add_argument("--runs", type=int, default=3, help="runs per log (default 3)")
    parser.add_argument("--category", choices=CATEGORIES, help="only evaluate this category")
    parser.add_argument("--model", default=None, help=f"Ollama model (default {OLLAMA_MODEL})")
    parser.add_argument("--out", default="evaluation_results.json", help="where to save the raw results (JSON)")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    # the pipeline logs a warning per flagged answer; the report already counts them
    logging.getLogger("kernel_log_analyzer").setLevel(logging.ERROR)
    entries = load_dataset(category=args.category)
    if not entries:
        print("No logs found for that category.", file=sys.stderr)
        return 1

    model = args.model or OLLAMA_MODEL
    print(f"Evaluating {model} on {len(entries)} log(s), {args.runs} run(s) each...\n")

    def show(record):
        if record["ok"]:
            print(f"  {record['category']:<11} run {record['run']}: {record['subsystem']:<10} {record['severity']:<8} "
                  f"{record['total_s']:>5}s  evidence {record['evidence_total'] - record['evidence_unverified']}/{record['evidence_total']}", flush=True)
        else:
            print(f"  {record['category']:<11} run {record['run']}: FAILED  {record['error']}", flush=True)

    records = run_evaluation(entries, runs=args.runs, model=args.model, progress=show)
    summary = summarise(records)

    out = Path(args.out)
    out.write_text(json.dumps({"model": model, "summary": summary, "runs": records}, indent=2), encoding="utf-8")
    print("\n" + format_report(summary, records, model))
    print(f"\nRaw results saved to {out.resolve()}")
    return 0 if summary["answered"] else 1


if __name__ == "__main__":
    sys.exit(main())
