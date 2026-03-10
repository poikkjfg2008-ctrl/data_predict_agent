#!/usr/bin/env python3
"""Assemble stage-level agent messages into deployable context payloads."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

STAGE_ORDER = [
    "exploration",
    "preprocessing",
    "feature_engineering",
    "model_selection",
    "training",
    "evaluation",
    "prediction",
]

STATUS_PRIORITY = {
    "error": 4,
    "needs_decision": 3,
    "warning": 2,
    "success": 1,
}


def _load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object")
    if "stages" not in payload or not isinstance(payload["stages"], dict):
        raise ValueError("input JSON missing object field: stages")
    return payload


def _sort_stage_items(stages: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    def key_fn(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
        stage = item[0]
        index = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
        return (index, stage)

    normalized = []
    for stage, content in stages.items():
        normalized.append((stage, content if isinstance(content, dict) else {"raw": content}))
    return sorted(normalized, key=key_fn)


def _detect_global_status(stage_items: List[Tuple[str, Dict[str, Any]]]) -> str:
    strongest = "success"
    strongest_score = 0
    for _, payload in stage_items:
        status = str(payload.get("status", "success")).lower()
        score = STATUS_PRIORITY.get(status, 0)
        if score > strongest_score:
            strongest = status
            strongest_score = score
    return strongest


def _build_context_blocks(stage_items: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for stage_name, payload in stage_items:
        blocks.append(
            {
                "stage": stage_name,
                "status": payload.get("status", "unknown"),
                "message": payload.get("message", ""),
                "critical_data": payload.get("data", {}),
                "next_actions": payload.get("next_actions", []),
                "suggestions": payload.get("suggestions", []),
                "agent_hints": payload.get("agent_hints", {}),
            }
        )
    return blocks


def _build_prompt_context(bundle: Dict[str, Any], max_data_chars: int) -> str:
    lines = [
        "# Pipeline Stage Context",
        f"run_id: {bundle['run_id']}",
        f"overall_status: {bundle['overall_status']}",
        "",
        "## Stage Timeline",
    ]

    for block in bundle["context_blocks"]:
        data_text = json.dumps(block["critical_data"], ensure_ascii=False)
        if len(data_text) > max_data_chars:
            data_text = data_text[:max_data_chars] + " ...<truncated>"

        lines.extend(
            [
                f"### {block['stage']}",
                f"- status: {block['status']}",
                f"- message: {block['message']}",
                f"- suggestions: {json.dumps(block['suggestions'], ensure_ascii=False)}",
                f"- next_actions: {json.dumps(block['next_actions'], ensure_ascii=False)}",
                f"- agent_hints: {json.dumps(block['agent_hints'], ensure_ascii=False)}",
                f"- critical_data: {data_text}",
                "",
            ]
        )

    lines.extend(
        [
            "## Execution Policy",
            "1. 优先处理 status=error 的阶段并执行 next_actions。",
            "2. 对 needs_decision 阶段，先读取 suggestions，再向上游发起补充问题。",
            "3. 成功阶段仅保留关键数据，避免把完整原始数据塞进上下文。",
        ]
    )
    return "\n".join(lines)


def assemble_bundle(source: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    stage_items = _sort_stage_items(source["stages"])
    summary = source.get("summary", {})
    bundle = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "overall_status": _detect_global_status(stage_items),
        "summary": summary if isinstance(summary, dict) else {"raw_summary": summary},
        "context_blocks": _build_context_blocks(stage_items),
    }
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble stage messages to context bundle")
    parser.add_argument("--input", required=True, help="path to pipeline_results.json")
    parser.add_argument("--output-json", required=True, help="path to output context bundle JSON")
    parser.add_argument("--output-md", required=True, help="path to output prompt context markdown")
    parser.add_argument("--run-id", default="run-local", help="logical run id for traceability")
    parser.add_argument(
        "--max-data-chars",
        type=int,
        default=800,
        help="truncate stage data in markdown summary after N chars",
    )
    args = parser.parse_args()

    source = _load_results(Path(args.input))
    bundle = assemble_bundle(source, args.run_id)
    prompt_text = _build_prompt_context(bundle, max_data_chars=max(100, args.max_data_chars))

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(prompt_text, encoding="utf-8")

    print(f"[ok] wrote context bundle: {output_json}")
    print(f"[ok] wrote prompt context: {output_md}")
    print(f"[info] stage_count={len(bundle['context_blocks'])} overall_status={bundle['overall_status']}")


if __name__ == "__main__":
    main()
