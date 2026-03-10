# 记忆档案

## 2026-03-09

### Agentic数值预测系统（新版推荐）

为用户开发了agent友好的数值预测管道，每个阶段输出结构化消息，支持agent根据中间结果做决策。

**程序路径**: `/root/.openclaw/workspace/agentic_prediction_pipeline.py`

**架构特点**:
- **AgentMessage**: 统一的消息格式，包含status/message/data/suggestions/next_actions/agent_hints
- **分阶段调用**: 每个阶段独立执行，返回结构化结果供agent决策
- **智能策略**: 自动识别零样本/小样本/常规场景，推荐合适的学习策略

**6个阶段**:
1. **数据探索** (DataExplorer): 输出数据结构、质量报告、目标列建议
2. **数据预处理** (DataPreprocessor): 缺失值处理、编码、标准化
3. **特征工程** (FeatureEngineer): 特征重要性分析、自动特征选择
4. **模型选择** (ModelSelector): 根据数据规模智能推荐模型和学习策略
5. **模型训练** (ModelTrainer): 支持零样本/小样本/常规训练，自动调参
6. **评估预测** (Evaluator): 交叉验证、性能评估、新数据预测

**支持的模型**:
- 轻量级: Ridge, Lasso, ElasticNet, KNN
- 集成模型: Random Forest, Gradient Boosting, Extra Trees
- 其他: SVR
- 深度学习: Neural Network (PyTorch，可选)

**学习策略**:
| 样本数 | 策略 | 推荐模型 |
|--------|------|----------|
| <50 | zero_shot | Ridge, Lasso (强正则化) |
| 50-200 | few_shot | Ridge, Lasso, ElasticNet |
| 200-1000 | small_scale | Ridge, Random Forest, GB |
| >1000 | full_training | Random Forest, GB, Extra Trees |

**自测结果**: ✓ 通过（500样本测试数据，R² ~0.27）

**使用方法**:
```python
from agentic_prediction_pipeline import AgenticPredictionPipeline

# 一键运行完整流程
pipeline = AgenticPredictionPipeline(output_dir="./output")
results = pipeline.run_full_pipeline(
    file_path="your_data.csv",
    target_col="target",  # 可选，自动检测
    model_preference="accuracy"  # speed/accuracy/interpretability
)

# 查看每个阶段的结果
for stage, result in results["stages"].items():
    print(f"{stage}: {result['message']}")

# 对新数据预测
pred_msg = pipeline.predict_new("new_data.csv")
print(pred_msg.data["predictions"])

# 分阶段调用（agent控制流）
explorer = DataExplorer()
msg1 = explorer.explore("data.csv")
# agent根据msg1决定下一步...
```

为用户开发了一个完整的数值预测任务管道，支持从数据处理到模型预测的全流程。

**程序路径**: `/root/.openclaw/workspace/numerical_prediction_pipeline.py`

**功能特点**:
- 阶段1: 数据加载与预处理（自动检测特征/目标、缺失值处理、标准化）
- 阶段2: 特征选择（基于相关性）和数据对构建（训练/验证/测试集划分）
- 阶段3: 模型自动选择（根据数据特点选择统计方法或深度学习）
- 阶段4: 模型训练（支持Ridge/Lasso/随机森林/梯度提升/神经网络）
- 阶段5: 预测与评估（含交叉验证）

**支持的模型**:
- 统计模型: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
- 深度学习: Neural Network (PyTorch，可选)

**自测结果**: ✓ 通过（1000样本测试数据，R² ~0.19-0.20）

**使用方法**:
```python
from numerical_prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline(log_dir="./logs")
results = pipeline.run(
    file_path="your_data.csv",
    output_dir="./output"
)

# 对新数据预测
predictions = pipeline.predict_new("new_data.csv")
```


### 本轮会话更新（2026-03-09）
- 用户新增任务：优化 `agentic_prediction_pipeline.py` 的边界处理。
- 用户新增任务：构建适配该项目的 agent loop。
- 用户新增任务：补齐使用文档和开发文档。
- 已执行：增强输入校验、统一错误消息、CV 折数安全处理、推理未知类别/缺失列保护、流程失败短路，新增 `AgentLoop`、`USAGE.md`、`DEVELOPMENT.md`。

## 2026-03-10

### Skill化集成需求（新）
- 用户希望把 `data_predict_agent` 项目中的阶段消息（AgentMessage）以 skill 形式组装，接入其已部署智能体的 context。
- 用户明确要求参考 skill-creator，产出完整 `SKILL.md` 与 `scripts/xxx.py`，覆盖全流程接入。
- 已新增 skill 草案目录：`skills/agentic-stage-context/`，用于运行管道并生成 context package。
