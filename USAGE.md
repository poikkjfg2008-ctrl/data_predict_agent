# Agentic Prediction Pipeline 使用文档

## 快速开始

```python
from agentic_prediction_pipeline import AgenticPredictionPipeline, AgentLoop

pipeline = AgenticPredictionPipeline(output_dir="./output")
results = pipeline.run_full_pipeline(
    file_path="./data/train.csv",
    target_col="target",          # 可选
    model_preference="accuracy"   # speed/accuracy/interpretability
)

# agent loop（推荐）
loop = AgentLoop(pipeline)
loop_results = loop.run("./data/train.csv", target_col="target")
```

## 结果结构

- `results["stages"]`: 每个阶段的 `AgentMessage`
- `results["summary"]`: 最终模型、R²、RMSE、选中特征
- `results["agent_loop"]`（仅 loop）: 决策日志与完成状态

## 新数据预测

```python
pred_msg = pipeline.predict_new("./data/new.csv")
print(pred_msg.status)
print(pred_msg.data.get("predictions", []))
```

## 边界情况行为

- 空文件/空 DataFrame：在 exploration/preprocessing/feature_engineering/training/evaluation 阶段返回 `error`
- 缺失列：返回 `missing_columns` 错误提示
- 未知类别：推理阶段自动回退到训练集中首个已知类别，避免 `LabelEncoder` 崩溃
- 交叉验证折数：自动使用安全折数，避免极小样本触发异常
- 目标列候选为空：全流程提前失败并在 `summary` 给出失败阶段

## 产物

- `output/pipeline_results.json`
- `output/model_<model_name>.pkl`
