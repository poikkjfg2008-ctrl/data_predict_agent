# Agentic Prediction Pipeline 开发文档

## 模块结构

- `DataExplorer`：读取数据、质量分析、目标/特征候选建议。
- `DataPreprocessor`：缺失值处理、类别编码、特征缩放、常量特征剔除。
- `FeatureEngineer`：特征重要性计算与选择。
- `ModelSelector`：根据样本规模和偏好推荐模型。
- `ModelTrainer`：训练、调参、过拟合检测。
- `Evaluator`：测试集评估、交叉验证、预测。
- `AgentLoop`：将上述阶段串成可观测、可恢复、可重试的 agent 回路。

## AgentLoop 设计原则

1. **每步可观测**：所有阶段结果进入 `trace`。
2. **失败可中断**：任何阶段报错，立即返回错误上下文。
3. **决策可插拔**：策略（特征方法、模型偏好、重试）可通过参数控制。
4. **状态可延续**：回路完成后同步 `pipeline.state` 供 `predict_new` 使用。

## 新增稳健性改进

- 安全编码器 `SafeLabelEncoder` 处理预测阶段未知类别。
- 缺失值填充增加全空列兜底（数值列填 0、类别列填 `__MISSING__`）。
- 自动移除常量特征，避免无效训练。
- 显式校验特征/目标列是否存在。
- 交叉验证折数添加最小样本保护。
- `run_full_pipeline` 增加阶段错误短路返回，避免链式异常。

## 建议测试

```bash
python -m py_compile agentic_prediction_pipeline.py
python agentic_prediction_pipeline.py
```

如要覆盖极端场景，可新增以下数据集：

- 仅 1 行样本
- 全空目标列
- 全常量特征列
- 预测集含训练未见类别
