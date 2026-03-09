"""Agent loop for orchestrating AgenticPredictionPipeline stages.

This module provides an opinionated agent loop that can:
1. run stage-by-stage execution
2. inspect each AgentMessage and make decisions
3. stop safely on errors or warnings requiring human confirmation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from agentic_prediction_pipeline import AgenticPredictionPipeline, StageStatus, AgentMessage


@dataclass
class LoopDecision:
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class DefaultDecisionPolicy:
    """Simple decision policy for autonomous execution."""

    def decide(self, stage: str, msg: AgentMessage, context: Dict[str, Any]) -> LoopDecision:
        if msg.status == StageStatus.ERROR.value:
            return LoopDecision(action="stop", reason=f"{stage} failed: {msg.message}")

        if stage == "exploration":
            target_candidates = msg.data.get("target_candidates") or []
            if not context.get("target_col"):
                if target_candidates:
                    context["target_col"] = target_candidates[0]
                else:
                    return LoopDecision(action="stop", reason="No target candidates found")
            if isinstance(context["target_col"], dict):
                context["target_col"] = context["target_col"]["column"]

            if not context.get("feature_cols"):
                context["feature_cols"] = msg.data.get("feature_candidates", [])
            if not context["feature_cols"]:
                return LoopDecision(action="stop", reason="No feature columns found")
            return LoopDecision(action="continue", reason="Target/features confirmed")

        if stage == "model_selection":
            recommended = msg.data.get("recommended_model") or {}
            context["model_name"] = context.get("model_name") or recommended.get("name", "ridge")
            return LoopDecision(action="continue", reason="Model selected")

        return LoopDecision(action="continue", reason="Proceed to next stage")


class AgentLoop:
    """Stateful loop runner for agentic pipeline."""

    def __init__(self, pipeline: AgenticPredictionPipeline, policy: Optional[DefaultDecisionPolicy] = None):
        self.pipeline = pipeline
        self.policy = policy or DefaultDecisionPolicy()

    def run(self, file_path: str, target_col: Optional[str] = None, feature_cols: Optional[List[str]] = None,
            model_preference: Optional[str] = "accuracy") -> Dict[str, Any]:
        ctx: Dict[str, Any] = {
            "file_path": file_path,
            "target_col": target_col,
            "feature_cols": feature_cols,
            "model_preference": model_preference,
            "messages": {}
        }

        msg1 = self.pipeline.explorer.explore(file_path, target_hint=target_col)
        ctx["messages"]["exploration"] = msg1.to_dict()
        d1 = self.policy.decide("exploration", msg1, ctx)
        if d1.action == "stop":
            return {"status": "stopped", "reason": d1.reason, "messages": ctx["messages"]}

        df = self.pipeline.explorer.raw_data
        msg2 = self.pipeline.preprocessor.preprocess(df, ctx["feature_cols"], [ctx["target_col"]])
        ctx["messages"]["preprocessing"] = msg2.to_dict()
        d2 = self.policy.decide("preprocessing", msg2, ctx)
        if d2.action == "stop":
            return {"status": "stopped", "reason": d2.reason, "messages": ctx["messages"]}

        self.pipeline.state["processed_data"] = self.pipeline.preprocessor.processed_data
        msg3 = self.pipeline.feature_engineer.engineer_features(
            self.pipeline.state["processed_data"], ctx["feature_cols"], ctx["target_col"]
        )
        ctx["messages"]["feature_engineering"] = msg3.to_dict()
        d3 = self.policy.decide("feature_engineering", msg3, ctx)
        if d3.action == "stop":
            return {"status": "stopped", "reason": d3.reason, "messages": ctx["messages"]}

        selected = self.pipeline.feature_engineer.selected_features
        split_msg = self.pipeline._split_data(self.pipeline.state["processed_data"], selected, ctx["target_col"])
        if split_msg is not None:
            ctx["messages"]["data_split"] = split_msg.to_dict()
            return {"status": "stopped", "reason": split_msg.message, "messages": ctx["messages"]}

        pairs = self.pipeline.state["data_pairs"]
        msg4 = self.pipeline.model_selector.recommend(
            len(pairs["X_train"]), pairs["X_train"].shape[1], user_preference=ctx["model_preference"]
        )
        ctx["messages"]["model_selection"] = msg4.to_dict()
        d4 = self.policy.decide("model_selection", msg4, ctx)
        if d4.action == "stop":
            return {"status": "stopped", "reason": d4.reason, "messages": ctx["messages"]}

        msg5 = self.pipeline.trainer.train(
            pairs["X_train"], pairs["y_train"],
            pairs["X_val"], pairs["y_val"],
            model_name=ctx.get("model_name", "ridge"),
            strategy=msg4.data["learning_strategy"]["strategy"],
            cv_folds=msg4.data["learning_strategy"]["cv_folds"],
            use_grid_search=True
        )
        ctx["messages"]["training"] = msg5.to_dict()
        d5 = self.policy.decide("training", msg5, ctx)
        if d5.action == "stop":
            return {"status": "stopped", "reason": d5.reason, "messages": ctx["messages"]}

        self.pipeline.evaluator.model = self.pipeline.trainer.model
        msg6 = self.pipeline.evaluator.evaluate(pairs["X_test"], pairs["y_test"], cv_folds=5)
        ctx["messages"]["evaluation"] = msg6.to_dict()

        return {
            "status": "completed",
            "summary": {
                "final_model": ctx.get("model_name", "ridge"),
                "r2": msg6.data["metrics"]["r2"],
                "rmse": msg6.data["metrics"]["rmse"],
                "selected_features": selected
            },
            "messages": ctx["messages"]
        }
