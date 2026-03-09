# 开发文档：agentic_prediction_pipeline 优化说明

## 本次优化范围
1. 边界条件与异常路径增强
2. 增加 Agent Loop 编排层
3. 输出面向 agent 的失败原因和下一步建议

## 关键改动

### 1) 公共安全函数
- `get_safe_cv_folds`: 保证交叉验证 folds 在合法范围，防止小样本报错。
- `sanitize_feature_columns`: 过滤不存在的特征列和目标列。

### 2) DataPreprocessor 边界处理
- 增加特征列/目标列存在性校验。
- 类别缺失值填充从“直接 mode[0]”改为“mode 或 unknown”，避免空众数报错。

### 3) FeatureEngineer 边界处理
- 目标列不存在、特征列无效时返回错误型 `AgentMessage`。
- 统一基于 `valid_features` 计算重要性，避免 KeyError/形状错配。

### 4) ModelSelector 边界处理
- 样本数或特征数不足时直接返回错误。
- 无推荐模型候选时返回错误而非继续访问空对象。

### 5) ModelTrainer 边界处理
- 修正 `y_val` 默认值（从 `np.ndarray` 类型对象改为 `None`）。
- 小样本训练提前失败并返回结构化错误。
- GridSearchCV 使用 `get_safe_cv_folds`。

### 6) Pipeline 主流程健壮性
- 每阶段失败即停止并输出 `summary.status=failed`。
- 自动目标列/特征列为空时安全失败。
- 小样本时采用更保守 split，样本不足（<8）直接失败提示。

### 7) 新增 AgentLoop
- 独立的 agent 控制层，支持每阶段覆写决策。
- 统一输出 `status/failed_stage/reason/summary`，便于上层agent做回退策略。

## 建议的后续演进
- 将预处理封装为 sklearn `Pipeline`，保证训练/预测变换一致。
- 类别特征替换为 `OneHotEncoder(handle_unknown='ignore')`。
- 引入 `pytest` 和最小数据集回归测试，覆盖 tiny/small/dirty data。
