#!/usr/bin/env python3
"""Assemble stage-wise AgentMessage outputs into deployable agent context bundles."""

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
]


def load_pipeline_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compact_data(data: Any, max_items: int) -> Any:
    if not isinstance(data, dict):
        return data

    compact: Dict[str, Any] = {}
    for idx, (key, value) in enumerate(data.items()):
        if idx >= max_items:
            compact["_truncated"] = f"Only first {max_items} keys are kept."
            break
        compact[key] = value
    return compact


def build_stage_contexts(results: Dict[str, Any], max_data_items: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, Any]]]:
    stages = results.get("stages", {})
    stage_contexts: List[Dict[str, Any]] = []
    risk_flags: List[Dict[str, str]] = []
    next_actions: List[Dict[str, Any]] = []

    ordered_stages = [s for s in STAGE_ORDER if s in stages]
    ordered_stages.extend([s for s in stages if s not in ordered_stages])

    for stage in ordered_stages:
        payload = stages.get(stage, {})
        status = payload.get("status", "unknown")
        context = {
            "stage": stage,
            "status": status,
            "message": payload.get("message", ""),
            "important_data": _compact_data(payload.get("data", {}), max_data_items),
            "suggestions": payload.get("suggestions", []),
            "next_actions": payload.get("next_actions", []),
            "agent_hints": payload.get("agent_hints", {}),
        }
        stage_contexts.append(context)

        if status in {"error", "warning"}:
            risk_flags.append({"stage": stage, "status": status, "message": context["message"]})

        for action in context["next_actions"]:
            tagged = dict(action)
            tagged["stage"] = stage
            next_actions.append(tagged)

    next_actions.sort(key=lambda x: (not bool(x.get("required", False)), x.get("stage", "")))
    return stage_contexts, risk_flags, next_actions


def build_bundle(results: Dict[str, Any], source_path: Path, max_data_items: int) -> Dict[str, Any]:
    summary = results.get("summary", {})
    stage_contexts, risk_flags, next_actions = build_stage_contexts(results, max_data_items)

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(source_path),
            "stage_count": len(stage_contexts),
        },
        "summary": {
            "status": summary.get("status", "succeeded"),
            "failed_stage": summary.get("failed_stage"),
            "reason": summary.get("reason"),
            "final_model": summary.get("final_model"),
            "test_r2": summary.get("test_r2"),
            "test_rmse": summary.get("test_rmse"),
            "selected_features": summary.get("selected_features", []),
        },
        "stage_contexts": stage_contexts,
        "decision_context": {
            "recommended_next_actions": next_actions,
            "risk_flags": risk_flags,
        },
    }


def render_markdown(bundle: Dict[str, Any]) -> str:
    summary = bundle["summary"]
    lines = [
        "# Agent Context Bundle",
        "",
        "## Global Summary",
        f"- Status: `{summary.get('status')}`",
        f"- Failed Stage: `{summary.get('failed_stage')}`",
        f"- Final Model: `{summary.get('final_model')}`",
        f"- Test R2: `{summary.get('test_r2')}`",
        f"- Test RMSE: `{summary.get('test_rmse')}`",
        "",
        "## Stage Snapshots",
    ]

    for stage in bundle["stage_contexts"]:
        lines.extend(
            [
                "",
                f"### {stage['stage']}",
                f"- Status: `{stage['status']}`",
                f"- Message: {stage['message']}",
                "- Suggestions:",
            ]
        )

        suggestions = stage.get("suggestions", []) or ["(none)"]
        lines.extend([f"  - {item}" for item in suggestions])

        lines.append("- Next Actions:")
        actions = stage.get("next_actions", []) or [{"description": "(none)"}]
        for action in actions:
            action_name = action.get("action", "unknown_action")
            desc = action.get("description", "")
            required = action.get("required", False)
            lines.append(f"  - `{action_name}`: {desc} (required={required})")

    lines.extend(["", "## Recommended Next Actions (cross-stage)"])
    for action in bundle["decision_context"]["recommended_next_actions"][:10] or [{"description": "(none)"}]:
        lines.append(
            f"- [{action.get('stage', 'unknown')}] `{action.get('action', 'unknown_action')}`: "
            f"{action.get('description', '')} (required={action.get('required', False)})"
        )

    lines.extend(["", "## Risk Flags"])
    risk_flags = bundle["decision_context"]["risk_flags"]
    if not risk_flags:
        lines.append("- No error/warning flags detected.")
    else:
        for risk in risk_flags:
            lines.append(f"- [{risk['stage']}] {risk['status']}: {risk['message']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble pipeline stage messages into agent context files.")
    parser.add_argument("--input", required=True, type=Path, help="Path to pipeline_results.json")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for output files")
    parser.add_argument("--max-data-items", type=int, default=12, help="Max keys kept from each stage.data")
    args = parser.parse_args()

    results = load_pipeline_results(args.input)
    bundle = build_bundle(results, args.input, args.max_data_items)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "agent_context_bundle.json"
    md_path = args.output_dir / "agent_context_bundle.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(render_markdown(bundle))

    print(f"Generated: {json_path}")
    print(f"Generated: {md_path}")


if __name__ == "__main__":
    main()
