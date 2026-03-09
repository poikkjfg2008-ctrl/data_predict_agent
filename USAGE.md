# Agentic Prediction Pipeline 使用文档

## 快速开始

### 1) 一键全流程

```python
from agentic_prediction_pipeline import AgenticPredictionPipeline

pipeline = AgenticPredictionPipeline(output_dir='./output')
results = pipeline.run_full_pipeline(
    file_path='train.csv',
    target_col='target',
    model_preference='accuracy',
)
print(results['summary'])
```

### 2) 使用 AgentLoop（推荐给 Agent 场景）

```python
from agentic_prediction_pipeline import AgenticPredictionPipeline
from agent_loop import AgentLoop

pipeline = AgenticPredictionPipeline(output_dir='./output')
loop = AgentLoop(pipeline)
result = loop.run(file_path='train.csv', target_col='target')

if result['status'] == 'completed':
    print('done:', result['summary'])
else:
    print('stopped:', result['reason'])
```

## 常见边界情况处理

- 空数据文件：探索阶段直接返回 `error`，并提示检查输入。
- 未找到目标列：探索阶段返回 `warning` 并要求手动指定目标列。
- 特征或目标列不存在：预处理阶段会中止并返回缺失列清单。
- 训练样本过少：数据切分与训练阶段都有显式保护。
- 预测数据类别出现未知值：自动回退到已知类别，避免编码器崩溃。
- 预测数据缺失训练所需特征：预测阶段返回错误，避免静默失败。

## 输出说明

每个阶段输出 `AgentMessage`：
- `status`: success / warning / error / needs_decision
- `message`: 人类可读状态
- `data`: 结构化结果
- `suggestions`: 推荐动作
- `next_actions`: 下一步候选动作
- `agent_hints`: 给 Agent 的决策信号
