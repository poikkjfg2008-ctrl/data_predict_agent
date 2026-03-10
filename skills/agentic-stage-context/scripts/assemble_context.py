#!/usr/bin/env python3
"""Assemble pipeline stage messages into an agent-ready context package."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

STAGE_ORDER = [
    "exploration",
    "preprocessing",
    "feature_engineering",
    "model_selection",
    "model_training",
    "evaluation",
    "prediction",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert pipeline messages to context package.")
    parser.add_argument("--input", required=True, help="Pipeline results JSON path.")
    parser.add_argument("--context-json", required=True, help="Output path for context JSON.")
    parser.add_argument("--context-md", required=True, help="Output path for context Markdown.")
    parser.add_argument("--max-suggestions", type=int, default=2, help="Max suggestions per stage.")
    parser.add_argument("--max-actions", type=int, default=3, help="Max next_actions per stage.")
    return parser.parse_args()


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_stages(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "stages" in payload and isinstance(payload["stages"], dict):
        return payload["stages"]
    # fallback: raw stage map
    return {k: v for k, v in payload.items() if isinstance(v, dict) and "status" in v and "message" in v}


def build_context(payload: Dict[str, Any], max_suggestions: int, max_actions: int) -> Dict[str, Any]:
    stages = _extract_stages(payload)
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}

    stage_summaries: List[Dict[str, Any]] = []
    critical_findings: List[Dict[str, str]] = []
    action_pool: List[Dict[str, Any]] = []

    ordered_keys = [k for k in STAGE_ORDER if k in stages] + [k for k in stages if k not in STAGE_ORDER]

    for stage in ordered_keys:
        data = stages[stage]
        status = data.get("status", "unknown")
        message = data.get("message", "")
        suggestions = (data.get("suggestions") or [])[:max_suggestions]
        next_actions = (data.get("next_actions") or [])[:max_actions]
        hints = data.get("agent_hints") or {}

        stage_summaries.append(
            {
                "stage": stage,
                "status": status,
                "message": message,
                "suggestions": suggestions,
                "next_actions": next_actions,
                "agent_hints": hints,
            }
        )

        if status in {"warning", "error", "needs_decision"}:
            critical_findings.append({"stage": stage, "status": status, "message": message})

        action_pool.extend(next_actions)

    unique_actions = []
    seen = set()
    for action in action_pool:
        key = (action.get("action"), action.get("description"))
        if key not in seen:
            seen.add(key)
            unique_actions.append(action)

    pipeline_status = summary.get("status", "completed")
    if any(item["status"] == "error" for item in critical_findings):
        pipeline_status = "failed"

    recovery_plan = [
        {
            "step": "定位失败阶段",
            "detail": summary.get("failed_stage") or "根据 critical_findings 中 status=error 的 stage 定位",
        },
        {
            "step": "执行修复动作",
            "detail": "优先执行失败阶段给出的 next_actions（如 retry/check_path/补齐特征列）",
        },
        {
            "step": "局部重跑",
            "detail": "从失败阶段起重跑，不要盲目继续后续阶段",
        },
    ] if pipeline_status == "failed" else []

    brief_lines = [
        f"pipeline_status: {pipeline_status}",
        "stage_summary:",
    ]
    for item in stage_summaries:
        brief_lines.append(f"- {item['stage']} [{item['status']}]: {item['message']}")

    return {
        "pipeline_status": pipeline_status,
        "stage_order": ordered_keys,
        "stage_summaries": stage_summaries,
        "critical_findings": critical_findings,
        "recommended_actions": unique_actions[:8],
        "recovery_plan": recovery_plan,
        "agent_loop_brief": "\n".join(brief_lines),
    }


def to_markdown(context: Dict[str, Any]) -> str:
    lines = [
        "# Agent Context Package",
        "",
        f"- **pipeline_status**: `{context['pipeline_status']}`",
        "",
        "## Stage Summaries",
    ]

    for item in context["stage_summaries"]:
        lines.extend(
            [
                f"### {item['stage']} ({item['status']})",
                item["message"] or "(no message)",
                "",
            ]
        )
        if item["suggestions"]:
            lines.append("Suggestions:")
            lines.extend([f"- {s}" for s in item["suggestions"]])
            lines.append("")
        if item["next_actions"]:
            lines.append("Next actions:")
            for action in item["next_actions"]:
                lines.append(f"- {action.get('action')}: {action.get('description', '')}")
            lines.append("")

    lines.append("## Agent Loop Brief")
    lines.append("```")
    lines.append(context["agent_loop_brief"])
    lines.append("```")
    lines.append("")

    if context["recovery_plan"]:
        lines.append("## Recovery Plan")
        for step in context["recovery_plan"]:
            lines.append(f"- **{step['step']}**: {step['detail']}")

    return "\n".join(lines)


def main():
    args = parse_args()
    payload = _load(args.input)
    context = build_context(payload, max_suggestions=args.max_suggestions, max_actions=args.max_actions)

    json_path = Path(args.context_json)
    md_path = Path(args.context_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(context), encoding="utf-8")

    print(json.dumps({"context_json": str(json_path), "context_md": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
