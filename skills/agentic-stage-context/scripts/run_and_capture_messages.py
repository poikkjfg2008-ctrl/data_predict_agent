#!/usr/bin/env python3
"""Run AgenticPredictionPipeline and save stage messages to JSON."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline and capture stage messages.")
    parser.add_argument("--data-file", required=True, help="Path to CSV/XLSX training data.")
    parser.add_argument("--target-col", default=None, help="Optional target column.")
    parser.add_argument("--model-preference", default=None, help="Optional model preference: speed/accuracy/interpretability.")
    parser.add_argument("--output-dir", default="./output", help="Pipeline artifact output directory.")
    parser.add_argument("--output-json", required=True, help="Where to write full pipeline result JSON.")
    return parser.parse_args()


def main():
    args = parse_args()

    from agentic_prediction_pipeline import AgenticPredictionPipeline

    pipeline = AgenticPredictionPipeline(output_dir=args.output_dir)
    results = pipeline.run_full_pipeline(
        file_path=args.data_file,
        target_col=args.target_col,
        model_preference=args.model_preference,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary = results.get("summary", {})
    print(
        json.dumps(
            {
                "saved_to": str(output_path),
                "status": summary.get("status", "unknown"),
                "failed_stage": summary.get("failed_stage"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
