# Agentic Prediction Pipeline 使用文档

## 快速开始

```python
from agentic_prediction_pipeline import AgenticPredictionPipeline, AgentLoop

pipeline = AgenticPredictionPipeline(output_dir="./output")
loop = AgentLoop(pipeline)

result = loop.run(
    file_path="./train.csv",
    target_col=None,              # 可选，自动识别
    model_preference="accuracy", # speed / accuracy / interpretability
    auto_retry=True               # 过拟合时自动尝试备选模型
)

print(result["status"])
print(result["summary"])
```

## 两种运行方式

- `run_full_pipeline(...)`：一键全流程，适合脚本批处理。
- `AgentLoop.run(...)`：面向 agent 的决策循环，适合需要中间状态决策、失败恢复、重试策略的场景。

## 常见边界情况处理

- 空文件 / 空数据：`data_exploration` 返回错误并给出修复建议。
- 目标列缺失：自动兜底；仍无法确认时返回 `target_not_found`。
- 特征列全常量：预处理阶段自动剔除；若无可用特征则中止并提示。
- 新数据出现未知类别：`SafeLabelEncoder` 将未知值映射到 `__UNKNOWN__`，避免预测阶段崩溃。
- 训练样本过小：自动关闭不安全的 CV 设置，避免 `n_splits > n_samples`。

## 预测新数据

```python
pred_msg = pipeline.predict_new("./new_data.csv")
print(pred_msg.status)
print(pred_msg.data.get("predictions", [])[:5])
```

要求新数据包含训练时所需的特征列；若缺失会返回结构化错误消息。
