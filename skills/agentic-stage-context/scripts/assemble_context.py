#!/usr/bin/env python3
"""Assemble stage-wise pipeline messages into agent-ready context JSON."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_STAGE_ORDER = [
    "exploration",
    "preprocessing",
    "feature_engineering",
    "model_selection",
    "training",
    "evaluation",
    "prediction",
]


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _normalize_actions(actions: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in _safe_list(actions)[:max_items]:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "action": item.get("action"),
                "description": item.get("description"),
                "required": bool(item.get("required", False)),
                "condition": item.get("condition"),
                "parameters": item.get("parameters", {}),
            }
        )
    return normalized


def _stage_brief(stage_name: str, payload: Dict[str, Any], max_suggestions: int, max_actions: int) -> Dict[str, Any]:
    suggestions = [str(x) for x in _safe_list(payload.get("suggestions"))[:max_suggestions]]
    actions = _normalize_actions(payload.get("next_actions", []), max_actions)
    required_actions = [a for a in actions if a.get("required")]

    return {
        "stage": stage_name,
        "status": payload.get("status", "unknown"),
        "message": payload.get("message", ""),
        "suggestions": suggestions,
        "next_actions": actions,
        "required_actions": required_actions,
        "agent_hints": payload.get("agent_hints", {}),
        "data_highlights": _extract_data_highlights(stage_name, payload.get("data", {})),
    }


def _extract_data_highlights(stage_name: str, data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    highlights: Dict[str, Any] = {}
    if stage_name == "exploration":
        highlights["shape"] = data.get("shape")
        highlights["target_candidates"] = data.get("target_candidates", [])[:3]
        highlights["feature_candidates_count"] = len(data.get("feature_candidates", []))
    elif stage_name == "preprocessing":
        highlights["feature_columns"] = data.get("feature_columns", [])
        highlights["target_columns"] = data.get("target_columns", [])
        highlights["steps"] = data.get("preprocessing_log", [])[:5]
    elif stage_name == "feature_engineering":
        highlights["selected_features"] = data.get("selected_features", [])
        highlights["selected_count"] = len(data.get("selected_features", []))
    elif stage_name == "model_selection":
        highlights["recommended_model"] = (data.get("recommended_model") or {}).get("name")
        highlights["learning_strategy"] = (data.get("learning_strategy") or {}).get("strategy")
    elif stage_name == "training":
        highlights["model_name"] = data.get("model_name")
        highlights["val_r2"] = ((data.get("performance") or {}).get("val_r2"))
        highlights["cv_mean"] = ((data.get("cross_validation") or {}).get("mean"))
    elif stage_name == "evaluation":
        metrics = data.get("metrics", {})
        highlights["metrics"] = {
            "r2": metrics.get("r2"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
        }
    elif stage_name == "prediction":
        preds = data.get("predictions", [])
        highlights["prediction_count"] = len(preds)
        highlights["prediction_preview"] = preds[:5]

    # Keep unknown but potentially useful keys in extras for forward compatibility.
    known_keys = {
        "shape",
        "target_candidates",
        "feature_candidates",
        "feature_columns",
        "target_columns",
        "preprocessing_log",
        "selected_features",
        "recommended_model",
        "learning_strategy",
        "model_name",
        "performance",
        "cross_validation",
        "metrics",
        "predictions",
    }
    extras = {k: v for k, v in data.items() if k not in known_keys}
    if extras:
        highlights["extras"] = extras

    return highlights


def assemble_context(results: Dict[str, Any], max_suggestions: int, max_actions: int, include_raw: bool) -> Dict[str, Any]:
    stages = results.get("stages", {}) if isinstance(results, dict) else {}
    summary = results.get("summary", {}) if isinstance(results, dict) else {}
    agent_loop = results.get("agent_loop", {}) if isinstance(results, dict) else {}

    stage_names = [s for s in DEFAULT_STAGE_ORDER if s in stages]
    stage_names.extend([s for s in stages.keys() if s not in stage_names])

    stage_contexts = []
    all_required_actions: List[Dict[str, Any]] = []
    all_suggestions: List[str] = []
    failed_stage = summary.get("failed_stage")

    for stage_name in stage_names:
        payload = stages.get(stage_name, {})
        if not isinstance(payload, dict):
            continue
        brief = _stage_brief(stage_name, payload, max_suggestions=max_suggestions, max_actions=max_actions)
        if include_raw:
            brief["raw_stage_payload"] = payload
        stage_contexts.append(brief)
        all_required_actions.extend(brief.get("required_actions", []))
        all_suggestions.extend(brief.get("suggestions", []))
        if payload.get("status") == "error" and failed_stage is None:
            failed_stage = stage_name

    pipeline_status = "failed" if (summary.get("status") == "failed" or failed_stage) else "success"

    handoff = {
        "high_priority_actions": all_required_actions,
        "recommended_actions": all_suggestions[:8],
        "recovery_actions": [
            "检查 failed_stage 的输入数据与列名一致性",
            "按 required_actions 顺序执行恢复动作",
            "重新运行失败阶段并比较前后 stage_contexts",
        ]
        if pipeline_status == "failed"
        else [],
        "routing_hint": "recovery_mode" if pipeline_status == "failed" else "normal_mode",
    }

    return {
        "meta": {
            "context_schema": "agentic-stage-context.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "agentic_prediction_pipeline",
        },
        "pipeline_status": pipeline_status,
        "failed_stage": failed_stage,
        "summary": summary,
        "agent_loop": agent_loop,
        "stage_contexts": stage_contexts,
        "agent_handoff": handoff,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble pipeline results into agent-ready context JSON")
    parser.add_argument("--input", required=True, help="Path to pipeline_results.json")
    parser.add_argument("--output", required=True, help="Path to output agent_context.json")
    parser.add_argument("--max-suggestions", type=int, default=3, help="Max suggestions per stage")
    parser.add_argument("--max-next-actions", type=int, default=3, help="Max next actions per stage")
    parser.add_argument("--include-raw", action="store_true", help="Include raw stage payload in each stage context")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    context_payload = assemble_context(
        results,
        max_suggestions=max(1, args.max_suggestions),
        max_actions=max(1, args.max_next_actions),
        include_raw=args.include_raw,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(context_payload, f, ensure_ascii=False, indent=2)

    print(f"Context assembled: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
