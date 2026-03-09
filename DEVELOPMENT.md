# Agentic Prediction Pipeline 开发文档

## 架构概览

核心模块：
1. `DataExplorer`：文件加载、质量分析、目标列候选
2. `DataPreprocessor`：缺失值处理、编码、缩放
3. `FeatureEngineer`：特征重要性计算与筛选
4. `ModelSelector`：根据数据规模推荐模型策略
5. `ModelTrainer`：训练、调参、过拟合检查
6. `Evaluator`：测试评估与预测
7. `AgentLoop`：基于阶段消息自动决策并推进流程

## 关键设计

### 1. 统一消息接口
所有阶段都返回 `AgentMessage`，便于 Agent 进行规则决策与工具编排。

### 2. 防御式边界处理
新增处理点：
- 空 DataFrame 快速失败
- 缺失列检测
- 空特征集检测
- 小样本保护（切分和训练）
- 未知类别预测容错
- 新数据缺失特征显式报错

### 3. AgentLoop 的可扩展性
`DefaultDecisionPolicy` 是默认策略，可通过自定义 policy 注入：

```python
class MyPolicy(DefaultDecisionPolicy):
    def decide(self, stage, msg, context):
        ...
```

你可以按业务覆盖：
- 模型偏好选择
- warning 时中断/继续策略
- 人工确认闸门

## 建议后续优化

- 将预处理器与模型一起保存为完整 artifact（pipeline 化）。
- 增加时间序列专用 split 与指标。
- 为 `AgentLoop` 增加重试与回滚机制。
- 补充 `pytest` 自动化测试，覆盖边界 case。
