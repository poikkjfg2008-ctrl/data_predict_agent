# Agentic Prediction Pipeline 开发文档

## 模块结构

- `DataExplorer`: 数据加载、质量分析、目标/特征候选
- `DataPreprocessor`: 缺失值、编码、缩放
- `FeatureEngineer`: 特征重要性计算与选择
- `ModelSelector`: 学习策略与模型推荐
- `ModelTrainer`: 训练、网格搜索、训练/验证评估
- `Evaluator`: 测试集评估与预测
- `AgenticPredictionPipeline`: 端到端编排
- `AgentLoop`: 对全流程结果做 agent 决策日志聚合

## 本次优化重点

1. **统一错误返回**：新增 `_error_message`，保持每个阶段错误输出结构一致。
2. **输入校验强化**：针对空数据、空特征、缺失列、shape 不一致做前置校验。
3. **特征选择健壮性**：修复 `k` 显式传入时 `selected_features` 可能为空的问题。
4. **CV 折数安全策略**：新增 `get_safe_cv_folds`，极小样本时避免 CV 抛错。
5. **推理鲁棒性**：处理推理数据缺失列与未知类别，避免线上推理中断。
6. **流程短路**：`run_full_pipeline` 在阶段错误时立刻失败返回，避免级联异常。

## 开发建议

- 新阶段必须返回 `AgentMessage`
- 新增模型时同步更新 `ModelSelector.AVAILABLE_MODELS`
- 任何异常分支优先通过 `_error_message` 返回，不抛裸异常给上层
- 对生产推理优先保持“可降级不崩溃”

## 验证命令

```bash
python -m py_compile agentic_prediction_pipeline.py
python agentic_prediction_pipeline.py
```
