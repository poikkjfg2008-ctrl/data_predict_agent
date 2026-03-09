# Agent Loop 使用文档

## 目标
`AgentLoop` 为 agent 场景提供**可中断、可覆写决策、可观测阶段输出**的回归预测执行流。

## 快速开始
```python
from agentic_prediction_pipeline import AgentLoop

loop = AgentLoop()
result = loop.run(
    file_path="./data/train.csv",
    target_col="target",   # 可选
    model_preference="accuracy",  # speed/accuracy/interpretability
    decision_overrides={
        "preprocessing_strategy": "robust",
        "feature_method": "mutual_info",
        "use_grid_search": True
    }
)

print(result["status"])
print(result.get("summary", {}))
```

## 输出结构
- `status`: `success/failed/running`
- `stages`: 每个阶段的 `AgentMessage`
- `failed_stage`: 失败阶段（若失败）
- `reason`: 失败原因（若失败）
- `summary`: 成功摘要

## 可覆盖决策（decision_overrides）
- `preprocessing_strategy`: `auto | minmax | robust`
- `feature_method`: `correlation | mutual_info`
- `selected_features`: 手动指定特征
- `model_name`: 强制指定模型（如 `ridge`）
- `use_grid_search`: 是否调参

## 常见失败与处理
- `exploration` 失败：检查文件路径/格式（仅支持 CSV/Excel）
- `preprocessing` 失败：列名不存在或目标列缺失
- `feature_engineering` 失败：没有有效特征列
- `split` 失败：样本太少（<8）
- `training` 失败：训练样本不足或模型配置问题
