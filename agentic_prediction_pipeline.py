"""
Agentic数值预测系统
===================
每个步骤输出结构化消息，方便agent根据当前结果做下一步决策

使用方式:
1. 阶段化调用: 每个阶段独立执行，返回agent-friendly消息
2. 完整管道: 一键运行全流程
3. 交互式: agent根据中间结果动态调整参数
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import mutual_info_regression, SelectKBest, RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# 可选: 深度学习
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# 数据结构和消息格式定义
# ============================================================================

class StageStatus(Enum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    NEEDS_DECISION = "needs_decision"


@dataclass
class AgentMessage:
    """Agent友好的消息格式"""
    stage: str
    status: str
    message: str
    data: Dict[str, Any]
    suggestions: List[str]
    next_actions: List[Dict[str, Any]]
    agent_hints: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "stage": self.stage,
            "status": self.status,
            "message": self.message,
            "data": self.data,
            "suggestions": self.suggestions,
            "next_actions": self.next_actions,
            "agent_hints": self.agent_hints
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ============================================================================
# 工具函数
# ============================================================================

def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """分析数据质量"""
    quality_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "completeness": {},
        "uniqueness": {},
        "outliers": {},
        "data_types": {}
    }
    
    if len(df) == 0:
        return quality_report

    for col in df.columns:
        # 完整性
        missing_pct = df[col].isnull().mean() * 100
        quality_report["completeness"][col] = {
            "missing_count": int(df[col].isnull().sum()),
            "missing_pct": float(missing_pct),
            "status": "good" if missing_pct < 5 else "warning" if missing_pct < 20 else "critical"
        }
        
        # 唯一性
        unique_ratio = df[col].nunique() / max(len(df), 1)
        quality_report["uniqueness"][col] = {
            "unique_count": int(df[col].nunique()),
            "unique_ratio": float(unique_ratio),
            "is_id_like": unique_ratio > 0.9
        }
        
        # 异常值 (数值列)
        if pd.api.types.is_numeric_dtype(df[col]):
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_count = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
            quality_report["outliers"][col] = {
                "outlier_count": int(outlier_count),
                "outlier_pct": float(outlier_count / len(df) * 100)
            }
        
        # 数据类型
        quality_report["data_types"][col] = str(df[col].dtype)
    
    return quality_report


def get_recommended_scaler(data_stats: Dict) -> str:
    """根据数据特征推荐缩放器"""
    has_outliers = any(
        stats.get("outlier_pct", 0) > 5 
        for stats in data_stats.get("outliers", {}).values()
    )
    return "robust" if has_outliers else "standard"


def get_learning_strategy(n_samples: int, n_features: int) -> Dict:
    """根据数据规模推荐学习策略"""
    if n_samples < 50:
        return {
            "strategy": "zero_shot",
            "description": "样本极少，建议使用预训练模型或领域知识",
            "recommended_models": ["ridge", "lasso"],
            "cv_folds": 3,
            "needs_augmentation": True
        }
    elif n_samples < 200:
        return {
            "strategy": "few_shot",
            "description": "小样本场景，使用正则化模型+数据增强",
            "recommended_models": ["ridge", "lasso", "elasticnet"],
            "cv_folds": 3,
            "needs_augmentation": True
        }
    elif n_samples < 1000:
        return {
            "strategy": "small_scale",
            "description": "中小规模数据，可用集成模型",
            "recommended_models": ["ridge", "random_forest", "gradient_boosting"],
            "cv_folds": 5,
            "needs_augmentation": False
        }
    else:
        return {
            "strategy": "full_training",
            "description": "充足样本，可使用复杂模型",
            "recommended_models": ["random_forest", "gradient_boosting", "extra_trees"],
            "cv_folds": 5,
            "needs_augmentation": False
        }


def get_safe_cv_folds(n_samples: int, requested_folds: int = 5) -> int:
    """安全的交叉验证折数，避免样本过少导致报错"""
    if n_samples <= 3:
        return 2
    return max(2, min(requested_folds, n_samples))


def sanitize_feature_columns(df: pd.DataFrame, feature_cols: List[str], target_col: Optional[str] = None) -> List[str]:
    """过滤掉不存在的列和目标列，避免下游崩溃"""
    valid_cols = [c for c in feature_cols if c in df.columns]
    if target_col:
        valid_cols = [c for c in valid_cols if c != target_col]
    return valid_cols


# ============================================================================
# 阶段1: 数据探索与理解
# ============================================================================

class DataExplorer:
    """数据探索阶段 - 输出agent友好的数据洞察"""
    
    def __init__(self):
        self.raw_data = None
        self.exploration_report = {}
    
    def explore(self, file_path: str, target_hint: Optional[str] = None) -> AgentMessage:
        """
        探索数据文件，返回结构化报告
        
        Args:
            file_path: 数据文件路径
            target_hint: 目标列名提示(可选)
        """
        # 加载数据
        try:
            if file_path.endswith('.csv'):
                self.raw_data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.raw_data = pd.read_excel(file_path)
            else:
                return AgentMessage(
                    stage="data_exploration",
                    status=StageStatus.ERROR.value,
                    message="不支持的文件格式，请使用 .csv 或 .xlsx",
                    data={},
                    suggestions=["转换文件格式为 CSV", "检查文件路径"],
                    next_actions=[
                        {"action": "convert_format", "description": "转换数据格式"},
                        {"action": "check_path", "description": "检查文件路径"}
                    ],
                    agent_hints={"error_type": "unsupported_format"}
                )
        except Exception as e:
            return AgentMessage(
                stage="data_exploration",
                status=StageStatus.ERROR.value,
                message=f"数据加载失败: {str(e)}",
                data={},
                suggestions=["检查文件是否存在", "检查文件编码格式"],
                next_actions=[{"action": "retry", "description": "重试加载"}],
                agent_hints={"error": str(e)}
            )
        
        # 基础统计
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.raw_data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.raw_data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # 质量分析
        quality_report = analyze_data_quality(self.raw_data)
        
        # 目标列建议
        target_candidates = []
        if target_hint and target_hint in self.raw_data.columns:
            target_candidates = [target_hint]
        else:
            # 如果存在名为'target'的列，优先作为目标
            if 'target' in numeric_cols:
                target_candidates = ['target']
            else:
                # 选择数值型列中变异最大的作为候选目标
                for col in numeric_cols:
                    cv = self.raw_data[col].std() / (abs(self.raw_data[col].mean()) + 1e-8)
                    target_candidates.append({"column": col, "cv": float(cv)})
                target_candidates.sort(key=lambda x: x["cv"], reverse=True)
                target_candidates = [x["column"] for x in target_candidates[:3]]
        
        # 特征列建议
        feature_candidates = [c for c in numeric_cols if c not in target_candidates]
        
        self.exploration_report = {
            "file_path": file_path,
            "shape": self.raw_data.shape,
            "columns": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "datetime": datetime_cols
            },
            "quality": quality_report,
            "target_candidates": target_candidates,
            "feature_candidates": feature_candidates,
            "preview": self.raw_data.head(5).to_dict(orient='records')
        }
        
        # 生成建议
        suggestions = []
        if quality_report["completeness"]:
            high_missing = [c for c, v in quality_report["completeness"].items() if v["status"] == "critical"]
            if high_missing:
                suggestions.append(f"列 {high_missing} 缺失严重，建议处理或删除")
        
        if len(numeric_cols) < 2:
            suggestions.append("数值列不足，可能需要特征工程或数据转换")
        
        # 下一步动作
        next_actions = [
            {
                "action": "confirm_target",
                "description": "确认目标列",
                "parameters": {"candidates": target_candidates[:3]},
                "required": True
            },
            {
                "action": "select_features",
                "description": "选择特征列",
                "parameters": {"candidates": feature_candidates, "auto_select": True},
                "required": False
            },
            {
                "action": "preprocess",
                "description": "进入数据预处理阶段",
                "parameters": {},
                "required": False
            }
        ]
        
        return AgentMessage(
            stage="data_exploration",
            status=StageStatus.SUCCESS.value,
            message=f"成功加载数据: {self.raw_data.shape[0]} 行 × {self.raw_data.shape[1]} 列",
            data=self.exploration_report,
            suggestions=suggestions,
            next_actions=next_actions,
            agent_hints={
                "data_size": "small" if len(self.raw_data) < 1000 else "medium" if len(self.raw_data) < 10000 else "large",
                "quality_score": self._calculate_quality_score(quality_report),
                "needs_target_confirmation": target_hint is None
            }
        )
    
    def _calculate_quality_score(self, quality_report: Dict) -> float:
        """计算数据质量评分"""
        scores = []
        for col_stats in quality_report.get("completeness", {}).values():
            scores.append(100 - col_stats["missing_pct"])
        return np.mean(scores) if scores else 0


# ============================================================================
# 阶段2: 数据预处理
# ============================================================================

class DataPreprocessor:
    """数据预处理阶段"""
    
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.encoders = {}
        self.processed_data = None
        self.feature_columns = []
        self.target_columns = []
    
    def preprocess(self, 
                   df: pd.DataFrame,
                   feature_cols: List[str],
                   target_cols: List[str],
                   strategy: str = "auto") -> AgentMessage:
        """
        执行数据预处理
        
        Args:
            df: 原始数据
            feature_cols: 特征列
            target_cols: 目标列
            strategy: 处理策略 (auto/minimal/robust)
        """
        self.feature_columns = feature_cols
        self.target_columns = target_cols
        
        processed_df = df.copy()
        preprocessing_log = []

        # 列校验
        missing_feature_cols = [c for c in feature_cols if c not in processed_df.columns]
        missing_target_cols = [c for c in target_cols if c not in processed_df.columns]
        feature_cols = [c for c in feature_cols if c in processed_df.columns]
        target_cols = [c for c in target_cols if c in processed_df.columns]

        if not feature_cols or not target_cols:
            return AgentMessage(
                stage="data_preprocessing",
                status=StageStatus.ERROR.value,
                message="预处理失败: 特征列或目标列为空/不存在",
                data={
                    "missing_feature_cols": missing_feature_cols,
                    "missing_target_cols": missing_target_cols,
                    "available_columns": processed_df.columns.tolist()
                },
                suggestions=["检查列名拼写", "重新确认目标列和特征列"],
                next_actions=[{"action": "confirm_columns", "description": "重新确认列配置", "required": True}],
                agent_hints={"error_type": "invalid_columns"}
            )
        
        # 1. 处理缺失值
        missing_stats = {}
        for col in feature_cols + target_cols:
            if col in processed_df.columns:
                missing_count = processed_df[col].isnull().sum()
                if missing_count > 0:
                    missing_pct = missing_count / len(processed_df) * 100
                    missing_stats[col] = {"count": int(missing_count), "pct": float(missing_pct)}
                    
                    if pd.api.types.is_numeric_dtype(processed_df[col]):
                        fill_value = processed_df[col].median() if strategy == "robust" else processed_df[col].mean()
                        processed_df[col] = processed_df[col].fillna(fill_value)
                        method = "中位数" if strategy == "robust" else "均值"
                        preprocessing_log.append(f"列 '{col}': 使用{method}填充 {missing_count} 个缺失值")
                    else:
                        mode_series = processed_df[col].mode(dropna=True)
                        fill_value = mode_series.iloc[0] if not mode_series.empty else "unknown"
                        processed_df[col] = processed_df[col].fillna(fill_value)
                        preprocessing_log.append(f"列 '{col}': 使用众数/默认值填充 {missing_count} 个缺失值")
        
        # 2. 编码分类变量
        categorical_features = [c for c in feature_cols if c in processed_df.columns 
                                and not pd.api.types.is_numeric_dtype(processed_df[c])]
        for col in categorical_features:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            self.encoders[col] = le
            preprocessing_log.append(f"列 '{col}': 使用LabelEncoder编码")
        
        # 3. 标准化/归一化
        numeric_features = [c for c in feature_cols if c in processed_df.columns 
                           and pd.api.types.is_numeric_dtype(processed_df[c])]
        
        if numeric_features:
            if strategy == "robust":
                self.scaler = RobustScaler()
                preprocessing_log.append("使用RobustScaler(抗异常值)")
            elif strategy == "minmax":
                self.scaler = MinMaxScaler()
                preprocessing_log.append("使用MinMaxScaler(归一化到0-1)")
            else:
                self.scaler = StandardScaler()
                preprocessing_log.append("使用StandardScaler(标准化)")
            
            processed_df[numeric_features] = self.scaler.fit_transform(processed_df[numeric_features])
        
        self.processed_data = processed_df
        
        # 计算处理后统计
        final_stats = {
            "shape": processed_df[feature_cols + target_cols].shape,
            "dropped_missing_columns": {
                "features": missing_feature_cols,
                "targets": missing_target_cols
            },
            "features": numeric_features,
            "encoded_categorical": list(self.encoders.keys()),
            "missing_handled": missing_stats
        }
        
        # 建议
        suggestions = []
        if missing_stats:
            suggestions.append(f"已处理 {len(missing_stats)} 个列的缺失值")
        if self.encoders:
            suggestions.append(f"已编码 {len(self.encoders)} 个分类变量")
        
        next_actions = [
            {
                "action": "feature_engineering",
                "description": "进入特征工程阶段",
                "parameters": {"processed_data_ready": True},
                "required": False
            },
            {
                "action": "split_data",
                "description": "直接划分训练/测试集",
                "parameters": {"test_size": 0.2},
                "required": False
            }
        ]
        
        return AgentMessage(
            stage="data_preprocessing",
            status=StageStatus.SUCCESS.value,
            message=f"预处理完成: {len(preprocessing_log)} 个处理步骤",
            data={
                "preprocessing_log": preprocessing_log,
                "final_stats": final_stats,
                "feature_columns": feature_cols,
                "target_columns": target_cols
            },
            suggestions=suggestions,
            next_actions=next_actions,
            agent_hints={
                "scaler_type": type(self.scaler).__name__ if self.scaler else None,
                "categorical_encoded": len(self.encoders) > 0,
                "missing_handled": len(missing_stats) > 0,
                "ready_for_training": True
            }
        )


# ============================================================================
# 阶段3: 特征工程
# ============================================================================

class FeatureEngineer:
    """特征工程阶段"""
    
    def __init__(self):
        self.selected_features = []
        self.feature_importance = {}
        self.transformers = []
    
    def engineer_features(self,
                         df: pd.DataFrame,
                         feature_cols: List[str],
                         target_col: str,
                         method: str = "correlation",
                         k: int = None) -> AgentMessage:
        """
        特征工程与选择
        
        Args:
            method: 特征选择方法 (correlation/mutual_info/rfe)
            k: 选择前k个特征，None表示自动决定
        """
        valid_features = sanitize_feature_columns(df, feature_cols, target_col=target_col)
        if target_col not in df.columns:
            return AgentMessage(
                stage="feature_engineering",
                status=StageStatus.ERROR.value,
                message=f"目标列不存在: {target_col}",
                data={"available_columns": df.columns.tolist()},
                suggestions=["确认目标列名"],
                next_actions=[{"action": "confirm_target", "description": "重新确认目标列", "required": True}],
                agent_hints={"error_type": "missing_target"}
            )
        if not valid_features:
            return AgentMessage(
                stage="feature_engineering",
                status=StageStatus.ERROR.value,
                message="没有可用的特征列",
                data={"requested_features": feature_cols, "available_columns": df.columns.tolist()},
                suggestions=["至少提供一个存在的特征列"],
                next_actions=[{"action": "select_features", "description": "重新选择特征", "required": True}],
                agent_hints={"error_type": "no_valid_features"}
            )

        X = df[valid_features].values
        y = df[target_col].values
        
        # 计算特征重要性
        importance_scores = {}
        
        if method == "correlation":
            for i, col in enumerate(valid_features):
                try:
                    corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                    if not np.isnan(corr):
                        importance_scores[col] = float(corr)
                except:
                    importance_scores[col] = 0.0
        
        elif method == "mutual_info":
            try:
                mi_scores = mutual_info_regression(X, y, random_state=42)
                for i, col in enumerate(valid_features):
                    importance_scores[col] = float(mi_scores[i])
            except:
                importance_scores = {col: 0.0 for col in feature_cols}
        
        # 排序并选择特征
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 自动决定k值
        if k is None:
            # 如果相关性太低，保留至少一些特征
            positive_features = [(f, s) for f, s in sorted_features if s > 0]
            if not positive_features:
                # 如果没有正相关特征，保留原始特征的一半
                self.selected_features = valid_features[:max(1, len(valid_features) // 2)]
            else:
                # 保留重要性 > 0.05 或至少保留50%的特征
                k = max(1, len([s for s in importance_scores.values() if s > 0.05]))
                k = max(k, len(valid_features) // 2)
                k = min(k, len(valid_features))  # 不能超过总数
                self.selected_features = [f for f, _ in sorted_features[:k]]
        self.feature_importance = importance_scores
        
        # 特征工程建议
        suggestions = []
        if len(self.selected_features) < len(valid_features):
            suggestions.append(f"从 {len(valid_features)} 个特征中选择了 {len(self.selected_features)} 个重要特征")
        
        # 计算特征重要性分布
        importance_values = list(importance_scores.values())
        if importance_values:
            importance_stats = {
                "mean": float(np.mean(importance_values)),
                "std": float(np.std(importance_values)) if len(importance_values) > 1 else 0.0,
                "max": float(np.max(importance_values)),
                "min": float(np.min(importance_values))
            }
        else:
            importance_stats = {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
        
        # 判断是否需要特征工程
        high_importance_count = sum(1 for v in importance_values if v > 0.3)
        needs_engineering = high_importance_count < 2 and len(valid_features) > 5
        
        next_actions = [
            {
                "action": "confirm_features",
                "description": "确认使用选中的特征",
                "parameters": {"selected_features": self.selected_features},
                "required": True
            },
            {
                "action": "create_interactions",
                "description": "创建交互特征",
                "parameters": {},
                "required": False,
                "condition": "if needs_engineering"
            },
            {
                "action": "proceed_to_model_selection",
                "description": "进入模型选择阶段",
                "parameters": {"feature_count": len(self.selected_features)},
                "required": False
            }
        ]
        
        return AgentMessage(
            stage="feature_engineering",
            status=StageStatus.SUCCESS.value,
            message=f"特征工程完成: 选择了 {len(self.selected_features)} 个特征",
            data={
                "feature_importance": self.feature_importance,
                "selected_features": self.selected_features,
                "importance_stats": importance_stats,
                "dropped_features": [f for f in valid_features if f not in self.selected_features],
                "invalid_feature_columns": [f for f in feature_cols if f not in valid_features]
            },
            suggestions=suggestions,
            next_actions=next_actions,
            agent_hints={
                "feature_reduction_ratio": 1 - len(self.selected_features) / max(len(valid_features), 1),
                "high_importance_features": high_importance_count,
                "needs_feature_engineering": needs_engineering,
                "recommended_method": "mutual_info" if method == "correlation" else "correlation"
            }
        )


# ============================================================================
# 阶段4: 模型选择与策略
# ============================================================================

class ModelSelector:
    """模型选择阶段 - 根据数据特点智能推荐模型"""
    
    AVAILABLE_MODELS = {
        "ridge": {
            "class": Ridge,
            "params": {"alpha": [0.01, 0.1, 1.0, 10.0]},
            "pros": ["正则化防止过拟合", "适合小样本", "可解释性强"],
            "cons": ["只能拟合线性关系"],
            "complexity": "low"
        },
        "lasso": {
            "class": Lasso,
            "params": {"alpha": [0.001, 0.01, 0.1, 1.0]},
            "pros": ["特征选择", "产生稀疏模型", "适合高维数据"],
            "cons": ["可能过于激进地删除特征"],
            "complexity": "low"
        },
        "elasticnet": {
            "class": ElasticNet,
            "params": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.3, 0.5, 0.7]},
            "pros": ["结合Ridge和Lasso优点", "更灵活的 regularization"],
            "cons": ["需要调两个参数"],
            "complexity": "low"
        },
        "random_forest": {
            "class": RandomForestRegressor,
            "params": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
            "pros": ["处理非线性关系", "不易过拟合", "给出特征重要性"],
            "cons": ["训练较慢", "预测较慢", "占用内存大"],
            "complexity": "medium"
        },
        "gradient_boosting": {
            "class": GradientBoostingRegressor,
            "params": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
            "pros": ["高精度", "处理复杂模式", "适合表格数据"],
            "cons": ["容易过拟合", "训练慢", "对异常值敏感"],
            "complexity": "high"
        },
        "extra_trees": {
            "class": ExtraTreesRegressor,
            "params": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
            "pros": ["比随机森林更快", "更不容易过拟合"],
            "cons": ["方差可能更高"],
            "complexity": "medium"
        },
        "svr": {
            "class": SVR,
            "params": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
            "pros": ["适合非线性", "泛化能力强"],
            "cons": ["大规模数据训练慢", "需要特征缩放"],
            "complexity": "medium"
        },
        "knn": {
            "class": KNeighborsRegressor,
            "params": {"n_neighbors": [3, 5, 10]},
            "pros": ["简单直观", "无需训练", "适合局部模式"],
            "cons": ["预测慢", "对特征缩放敏感", "维度灾难"],
            "complexity": "low"
        }
    }
    
    def recommend(self,
                  n_samples: int,
                  n_features: int,
                  linearity_hint: Optional[float] = None,
                  user_preference: Optional[str] = None) -> AgentMessage:
        """
        智能推荐模型和学习策略
        
        Args:
            n_samples: 样本数量
            n_features: 特征数量
            linearity_hint: 线性拟合度(0-1)，None表示未知
            user_preference: 用户偏好 (speed/accuracy/interpretability)
        """
        if n_samples <= 1 or n_features <= 0:
            return AgentMessage(
                stage="model_selection",
                status=StageStatus.ERROR.value,
                message="样本数或特征数不足，无法推荐模型",
                data={"n_samples": n_samples, "n_features": n_features},
                suggestions=["补充数据或检查特征处理阶段"],
                next_actions=[{"action": "review_data", "description": "检查数据准备", "required": True}],
                agent_hints={"error_type": "insufficient_data"}
            )

        # 获取学习策略
        learning_strategy = get_learning_strategy(n_samples, n_features)
        
        # 筛选候选模型
        candidates = []
        for model_name in learning_strategy["recommended_models"]:
            if model_name in self.AVAILABLE_MODELS:
                model_info = self.AVAILABLE_MODELS[model_name]
                candidates.append({
                    "name": model_name,
                    "complexity": model_info["complexity"],
                    "pros": model_info["pros"],
                    "cons": model_info["cons"],
                    "score": self._score_model(model_name, n_samples, n_features, user_preference)
                })
        
        # 按分数排序
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # 推荐最佳模型
        recommended = candidates[0] if candidates else None
        
        # 构建解释
        if not recommended:
            return AgentMessage(
                stage="model_selection",
                status=StageStatus.ERROR.value,
                message="没有可用模型候选",
                data={"learning_strategy": learning_strategy},
                suggestions=["检查模型配置"],
                next_actions=[{"action": "custom_model", "description": "手动选择模型", "required": True}],
                agent_hints={"error_type": "no_candidate_models"}
            )

        explanation = f"基于{learning_strategy['strategy']}策略，推荐 {recommended['name']} 模型"
        if learning_strategy["needs_augmentation"]:
            explanation += "。建议考虑数据增强以提升性能"
        
        data = {
            "learning_strategy": learning_strategy,
            "candidates": candidates,
            "recommended_model": recommended,
            "data_profile": {
                "n_samples": n_samples,
                "n_features": n_features,
                "samples_per_feature": n_samples / max(n_features, 1),
                "size_category": "tiny" if n_samples < 50 else "small" if n_samples < 200 else "medium" if n_samples < 1000 else "large"
            }
        }
        
        suggestions = [
            f"学习策略: {learning_strategy['description']}",
            f"推荐模型: {recommended['name']} (复杂度: {recommended['complexity']})",
            f"交叉验证折数: {learning_strategy['cv_folds']}"
        ]
        
        if learning_strategy["needs_augmentation"]:
            suggestions.append("⚠️ 样本较少，建议使用正则化或数据增强")
        
        next_actions = [
            {
                "action": "confirm_model",
                "description": f"使用推荐模型: {recommended['name']}",
                "parameters": {"model_name": recommended['name']},
                "required": True
            },
            {
                "action": "try_multiple",
                "description": "训练多个候选模型并比较",
                "parameters": {"candidates": [c['name'] for c in candidates[:3]]},
                "required": False
            },
            {
                "action": "custom_model",
                "description": "选择其他模型",
                "parameters": {"available": list(self.AVAILABLE_MODELS.keys())},
                "required": False
            }
        ]
        
        return AgentMessage(
            stage="model_selection",
            status=StageStatus.SUCCESS.value,
            message=explanation,
            data=data,
            suggestions=suggestions,
            next_actions=next_actions,
            agent_hints={
                "is_few_shot": learning_strategy["strategy"] in ["zero_shot", "few_shot"],
                "recommended_model": recommended['name'],
                "alternative_models": [c['name'] for c in candidates[1:3]],
                "risk_level": "high" if n_samples < 100 else "medium" if n_samples < 1000 else "low"
            }
        )
    
    def _score_model(self, model_name: str, n_samples: int, n_features: int, preference: Optional[str]) -> float:
        """为模型打分"""
        model_info = self.AVAILABLE_MODELS[model_name]
        score = 50  # 基础分
        
        # 样本量适配
        if model_info["complexity"] == "low" and n_samples < 200:
            score += 20
        elif model_info["complexity"] == "high" and n_samples > 1000:
            score += 20
        
        # 用户偏好
        if preference == "speed" and model_info["complexity"] == "low":
            score += 15
        elif preference == "accuracy" and model_info["complexity"] == "high":
            score += 15
        elif preference == "interpretability":
            if model_name in ["ridge", "lasso", "elasticnet"]:
                score += 20
        
        return score


# ============================================================================
# 阶段5: 模型训练
# ============================================================================

class ModelTrainer:
    """模型训练阶段 - 支持零样本/小样本/常规训练"""
    
    def __init__(self):
        self.model = None
        self.best_params = {}
        self.training_history = {}
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              model_name: str = "ridge",
              strategy: str = "auto",
              cv_folds: int = 5,
              use_grid_search: bool = False) -> AgentMessage:
        """
        训练模型
        
        Args:
            strategy: 训练策略 (auto/zero_shot/few_shot/full)
            use_grid_search: 是否使用网格搜索调参
        """
        from sklearn.base import clone
        
        if len(X_train) < 2:
            return AgentMessage(
                stage="model_training",
                status=StageStatus.ERROR.value,
                message="训练样本过少，至少需要2条样本",
                data={"n_train_samples": int(len(X_train))},
                suggestions=["增加训练数据", "调整数据划分比例"],
                next_actions=[{"action": "resplit_data", "description": "重新划分数据", "required": True}],
                agent_hints={"error_type": "insufficient_train_samples"}
            )

        selector = ModelSelector()
        model_config = selector.AVAILABLE_MODELS.get(model_name, selector.AVAILABLE_MODELS["ridge"])
        
        # 根据策略调整参数
        if strategy in ["zero_shot", "few_shot"]:
            # 小样本: 更强的正则化
            if model_name in ["ridge", "lasso", "elasticnet"]:
                model_config["params"] = {"alpha": [0.1, 1.0, 10.0, 100.0]}
            use_grid_search = True  # 强制搜索最佳参数
        
        # 创建模型
        if use_grid_search and len(model_config["params"]) > 0:
            base_model = model_config["class"]()
            grid_search = GridSearchCV(
                base_model, 
                model_config["params"],
                cv=get_safe_cv_folds(len(X_train), cv_folds),
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            search_msg = f"网格搜索完成，最佳参数: {self.best_params}"
        else:
            self.model = model_config["class"]()
            self.model.fit(X_train, y_train)
            search_msg = "使用默认参数训练"
        
        # 训练集评估
        train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred)
        
        # 验证集评估
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)
        
        self.training_history = {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_params": self.best_params
        }
        
        # 过拟合检测
        overfitting_warning = None
        if val_metrics and train_metrics["r2"] - val_metrics["r2"] > 0.1:
            overfitting_warning = f"可能存在过拟合 (训练R²={train_metrics['r2']:.3f}, 验证R²={val_metrics['r2']:.3f})"
        
        suggestions = [search_msg]
        if overfitting_warning:
            suggestions.append(f"⚠️ {overfitting_warning}")
            suggestions.append("建议: 增加正则化强度或减少模型复杂度")
        
        if train_metrics["r2"] < 0.3:
            suggestions.append("模型表现一般，建议尝试更复杂的模型或特征工程")
        elif train_metrics["r2"] > 0.8:
            suggestions.append("模型表现优秀!")
        
        next_actions = [
            {
                "action": "evaluate",
                "description": "在测试集上评估模型",
                "parameters": {},
                "required": True
            },
            {
                "action": "retrain",
                "description": "使用不同参数重新训练",
                "parameters": {"model_name": model_name},
                "required": False
            },
            {
                "action": "try_other_model",
                "description": "尝试其他模型",
                "parameters": {},
                "required": False
            }
        ]
        
        status = StageStatus.WARNING.value if overfitting_warning else StageStatus.SUCCESS.value
        
        return AgentMessage(
            stage="model_training",
            status=status,
            message=f"训练完成: {model_name} 模型",
            data={
                "model_name": model_name,
                "best_params": self.best_params,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "overfitting_detected": overfitting_warning is not None
            },
            suggestions=suggestions,
            next_actions=next_actions,
            agent_hints={
                "model_ready": True,
                "needs_retrain": overfitting_warning is not None,
                "performance_tier": "high" if train_metrics["r2"] > 0.7 else "medium" if train_metrics["r2"] > 0.4 else "low",
                "training_samples": len(X_train)
            }
        )
    
    def _calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred)) if np.all(y_true != 0) else None
        }


# ============================================================================
# 阶段6: 评估与预测
# ============================================================================

class Evaluator:
    """评估与预测阶段"""
    
    def __init__(self, model=None):
        self.model = model
    
    def evaluate(self,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 model=None,
                 cv_folds: int = 5) -> AgentMessage:
        """评估模型性能"""
        
        model = model or self.model
        if model is None:
            return AgentMessage(
                stage="evaluation",
                status=StageStatus.ERROR.value,
                message="没有可评估的模型",
                data={},
                suggestions=["先训练模型"],
                next_actions=[{"action": "train", "description": "返回训练阶段", "required": True}],
                agent_hints={"error": "no_model"}
            )
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        metrics = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_test, y_pred)) if np.all(y_test != 0) else None
        }
        
        # 交叉验证
        from sklearn.model_selection import cross_val_score
        try:
            cv_scores = cross_val_score(model, X_test, y_test, cv=get_safe_cv_folds(len(X_test), cv_folds), scoring='r2')
            cv_results = {
                "scores": cv_scores.tolist(),
                "mean": float(cv_scores.mean()),
                "std": float(cv_scores.std())
            }
        except:
            cv_results = None
        
        # 结果解读
        interpretation = []
        if metrics["r2"] > 0.8:
            interpretation.append("模型解释能力强 (R² > 0.8)")
        elif metrics["r2"] > 0.5:
            interpretation.append("模型有一定解释能力 (R² > 0.5)")
        else:
            interpretation.append("模型解释能力较弱 (R² < 0.5)，建议改进")
        
        if metrics["mape"] and metrics["mape"] < 10:
            interpretation.append("预测误差小，模型可靠")
        elif metrics["mape"] and metrics["mape"] < 20:
            interpretation.append("预测误差中等")
        elif metrics["mape"]:
            interpretation.append("预测误差较大，需谨慎使用")
        
        data = {
            "metrics": metrics,
            "cv_results": cv_results,
            "predictions": {
                "y_true_sample": y_test[:10].tolist() if len(y_test) > 10 else y_test.tolist(),
                "y_pred_sample": y_pred[:10].tolist() if len(y_pred) > 10 else y_pred.tolist()
            }
        }
        
        suggestions = interpretation
        if metrics["r2"] < 0.5:
            suggestions.append("建议: 尝试更复杂的模型、增加特征或获取更多数据")
        
        next_actions = [
            {
                "action": "predict_new",
                "description": "对新数据进行预测",
                "parameters": {},
                "required": False
            },
            {
                "action": "save_model",
                "description": "保存训练好的模型",
                "parameters": {},
                "required": False
            },
            {
                "action": "restart",
                "description": "重新开始新任务",
                "parameters": {},
                "required": False
            }
        ]
        
        return AgentMessage(
            stage="evaluation",
            status=StageStatus.SUCCESS.value,
            message=f"评估完成: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}",
            data=data,
            suggestions=suggestions,
            next_actions=next_actions,
            agent_hints={
                "model_performance": "good" if metrics["r2"] > 0.7 else "fair" if metrics["r2"] > 0.4 else "poor",
                "reliable_for_production": metrics["r2"] > 0.7 and (metrics["mape"] is None or metrics["mape"] < 15),
                "can_predict": True
            }
        )
    
    def predict(self, X: np.ndarray, model=None) -> AgentMessage:
        """对新数据进行预测"""
        model = model or self.model
        if model is None:
            return AgentMessage(
                stage="prediction",
                status=StageStatus.ERROR.value,
                message="没有可用的预测模型",
                data={},
                suggestions=["先训练模型"],
                next_actions=[],
                agent_hints={}
            )
        
        predictions = model.predict(X)
        
        return AgentMessage(
            stage="prediction",
            status=StageStatus.SUCCESS.value,
            message=f"完成 {len(predictions)} 个样本的预测",
            data={
                "predictions": predictions.tolist(),
                "prediction_stats": {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions))
                }
            },
            suggestions=["预测完成，可保存结果或进行进一步分析"],
            next_actions=[
                {"action": "save_predictions", "description": "保存预测结果", "required": False},
                {"action": "visualize", "description": "可视化预测结果", "required": False}
            ],
            agent_hints={"predictions_ready": True}
        )


# ============================================================================
# 主控管道 - AgenticPredictionPipeline
# ============================================================================

class AgenticPredictionPipeline:
    """
    Agentic数值预测管道
    
    支持两种方式:
    1. 分阶段调用: 每个步骤返回AgentMessage，agent可以根据结果做决策
    2. 一键运行: run() 方法执行完整流程
    """
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 各阶段模块
        self.explorer = DataExplorer()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_selector = ModelSelector()
        self.trainer = ModelTrainer()
        self.evaluator = Evaluator()
        
        # 状态保存
        self.state = {
            "raw_data": None,
            "processed_data": None,
            "feature_cols": [],
            "target_col": None,
            "selected_features": [],
            "data_pairs": None,
            "model": None,
            "training_result": None
        }
    
    def run_full_pipeline(self,
                          file_path: str,
                          target_col: Optional[str] = None,
                          feature_cols: Optional[List[str]] = None,
                          model_preference: Optional[str] = None) -> Dict:
        """
        运行完整管道
        
        Returns:
            Dict containing all stage results
        """
        results = {"stages": {}}
        
        # 阶段1: 数据探索
        print("=" * 60)
        print("阶段1: 数据探索")
        print("=" * 60)
        msg1 = self.explorer.explore(file_path, target_hint=target_col)
        print(msg1.to_json())
        results["stages"]["exploration"] = msg1.to_dict()
        self.state["raw_data"] = self.explorer.raw_data
        
        # 自动确定目标列和特征列
        if msg1.status != StageStatus.SUCCESS.value:
            results["summary"] = {"status": "failed", "failed_stage": "exploration", "reason": msg1.message}
            return results

        if target_col is None:
            target_candidates = msg1.data.get("target_candidates", [])
            if not target_candidates:
                results["summary"] = {"status": "failed", "failed_stage": "exploration", "reason": "未检测到可用目标列"}
                return results
            target_col = target_candidates[0]
            if isinstance(target_col, dict):
                target_col = target_col["column"]
        
        if feature_cols is None:
            feature_cols = msg1.data.get("feature_candidates", [])

        feature_cols = sanitize_feature_columns(self.state["raw_data"], feature_cols, target_col=target_col)
        if not feature_cols:
            results["summary"] = {"status": "failed", "failed_stage": "exploration", "reason": "无有效特征列"}
            return results
        
        self.state["target_col"] = target_col
        self.state["feature_cols"] = feature_cols
        
        # 阶段2: 预处理
        print("\n" + "=" * 60)
        print("阶段2: 数据预处理")
        print("=" * 60)
        msg2 = self.preprocessor.preprocess(
            self.state["raw_data"],
            feature_cols,
            [target_col]
        )
        print(msg2.to_json())
        results["stages"]["preprocessing"] = msg2.to_dict()
        if msg2.status == StageStatus.ERROR.value:
            results["summary"] = {"status": "failed", "failed_stage": "preprocessing", "reason": msg2.message}
            return results
        self.state["processed_data"] = self.preprocessor.processed_data
        
        # 阶段3: 特征工程
        print("\n" + "=" * 60)
        print("阶段3: 特征工程")
        print("=" * 60)
        msg3 = self.feature_engineer.engineer_features(
            self.state["processed_data"],
            feature_cols,
            target_col
        )
        if msg3.status == StageStatus.ERROR.value:
            results["stages"]["feature_engineering"] = msg3.to_dict()
            results["summary"] = {"status": "failed", "failed_stage": "feature_engineering", "reason": msg3.message}
            return results
        print(msg3.to_json())
        results["stages"]["feature_engineering"] = msg3.to_dict()
        self.state["selected_features"] = self.feature_engineer.selected_features
        
        # 数据划分
        from sklearn.model_selection import train_test_split
        df = self.state["processed_data"]
        selected = self.state["selected_features"]
        
        X = df[selected].values
        y = df[target_col].values
        
        n_samples = len(df)
        if n_samples < 8:
            results["summary"] = {"status": "failed", "failed_stage": "split", "reason": "样本量过少，无法稳定划分训练/验证/测试集"}
            return results

        test_size = 0.2 if n_samples >= 20 else 0.25
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        val_size = 0.125 if len(X_train) >= 16 else 0.2
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
        
        self.state["data_pairs"] = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }
        
        # 阶段4: 模型选择
        print("\n" + "=" * 60)
        print("阶段4: 模型选择")
        print("=" * 60)
        msg4 = self.model_selector.recommend(
            len(X_train),
            X_train.shape[1],
            user_preference=model_preference
        )
        if msg4.status == StageStatus.ERROR.value:
            results["stages"]["model_selection"] = msg4.to_dict()
            results["summary"] = {"status": "failed", "failed_stage": "model_selection", "reason": msg4.message}
            return results
        print(msg4.to_json())
        results["stages"]["model_selection"] = msg4.to_dict()
        recommended_model = msg4.data["recommended_model"]["name"]
        
        # 阶段5: 训练
        print("\n" + "=" * 60)
        print("阶段5: 模型训练")
        print("=" * 60)
        msg5 = self.trainer.train(
            X_train, y_train,
            X_val, y_val,
            model_name=recommended_model,
            strategy=msg4.data["learning_strategy"]["strategy"],
            cv_folds=msg4.data["learning_strategy"]["cv_folds"],
            use_grid_search=True
        )
        print(msg5.to_json())
        results["stages"]["training"] = msg5.to_dict()
        if msg5.status == StageStatus.ERROR.value:
            results["summary"] = {"status": "failed", "failed_stage": "training", "reason": msg5.message}
            return results
        self.state["model"] = self.trainer.model
        self.evaluator.model = self.trainer.model
        
        # 阶段6: 评估
        print("\n" + "=" * 60)
        print("阶段6: 模型评估")
        print("=" * 60)
        msg6 = self.evaluator.evaluate(X_test, y_test, cv_folds=5)
        print(msg6.to_json())
        results["stages"]["evaluation"] = msg6.to_dict()
        
        # 保存结果
        results["summary"] = {
            "final_model": recommended_model,
            "test_r2": msg6.data["metrics"]["r2"],
            "test_rmse": msg6.data["metrics"]["rmse"],
            "selected_features": selected
        }
        
        # 保存到文件
        result_path = os.path.join(self.output_dir, "pipeline_results.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存模型
        model_path = os.path.join(self.output_dir, f"model_{recommended_model}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.trainer.model, f)
        
        print("\n" + "=" * 60)
        print("管道执行完成!")
        print(f"结果保存至: {result_path}")
        print(f"模型保存至: {model_path}")
        print("=" * 60)
        
        return results
    
    def predict_new(self, file_path: str) -> AgentMessage:
        """对新数据进行预测"""
        if self.state["model"] is None:
            return AgentMessage(
                stage="prediction",
                status=StageStatus.ERROR.value,
                message="模型尚未训练",
                data={},
                suggestions=["先运行完整管道"],
                next_actions=[],
                agent_hints={}
            )
        
        # 加载新数据
        try:
            if file_path.endswith('.csv'):
                new_df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                new_df = pd.read_excel(file_path)
            else:
                return AgentMessage(
                    stage="prediction",
                    status=StageStatus.ERROR.value,
                    message="不支持的预测文件格式",
                    data={"file_path": file_path},
                    suggestions=["请使用CSV或Excel文件"],
                    next_actions=[],
                    agent_hints={"error_type": "unsupported_format"}
                )
        except Exception as e:
            return AgentMessage(
                stage="prediction",
                status=StageStatus.ERROR.value,
                message=f"预测数据加载失败: {str(e)}",
                data={"file_path": file_path},
                suggestions=["检查文件路径和格式"],
                next_actions=[],
                agent_hints={"error": str(e)}
            )
        
        # 应用相同的预处理
        for col, encoder in self.preprocessor.encoders.items():
            if col in new_df.columns:
                known_values = set(encoder.classes_.tolist())
                new_values = new_df[col].astype(str)
                if any(v not in known_values for v in new_values):
                    return AgentMessage(
                        stage="prediction",
                        status=StageStatus.ERROR.value,
                        message=f"预测数据包含训练时未见过的类别: {col}",
                        data={"column": col},
                        suggestions=["统一训练/预测数据的类别取值", "考虑改用OneHotEncoder并处理未知类别"],
                        next_actions=[],
                        agent_hints={"error_type": "unknown_category"}
                    )
                new_df[col] = encoder.transform(new_values)
        
        # 选择特征并缩放
        selected = self.state["selected_features"]
        original_features = self.state["feature_cols"]
        
        # 使用scaler（需要全部原始特征）
        missing_features = [c for c in original_features if c not in new_df.columns]
        if missing_features:
            return AgentMessage(
                stage="prediction",
                status=StageStatus.ERROR.value,
                message="预测失败: 新数据缺少训练特征列",
                data={"missing_features": missing_features},
                suggestions=["补齐缺失特征列", "检查训练和预测数据schema一致性"],
                next_actions=[],
                agent_hints={"error_type": "missing_features"}
            )

        # 填充新数据缺失值（与训练阶段一致）
        for col in original_features:
            if new_df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(new_df[col]):
                    new_df[col] = new_df[col].fillna(new_df[col].median())
                else:
                    mode_series = new_df[col].mode(dropna=True)
                    new_df[col] = new_df[col].fillna(mode_series.iloc[0] if not mode_series.empty else "unknown")

        X_new_all = new_df[original_features].values
        X_new_all = self.preprocessor.scaler.transform(X_new_all)
        X_new_df = pd.DataFrame(X_new_all, columns=original_features)
        X_new = X_new_df[selected].values
        
        return self.evaluator.predict(X_new)




# ============================================================================
# Agent Loop: 面向自治Agent的控制流封装
# ============================================================================

class AgentLoop:
    """将pipeline拆成可中断、可决策、可恢复的agent loop。"""

    def __init__(self, pipeline: Optional[AgenticPredictionPipeline] = None):
        self.pipeline = pipeline or AgenticPredictionPipeline()

    def run(self,
            file_path: str,
            target_col: Optional[str] = None,
            feature_cols: Optional[List[str]] = None,
            model_preference: Optional[str] = None,
            decision_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        逐阶段执行并允许agent覆写决策。

        decision_overrides支持:
        - preprocessing_strategy: auto/minmax/robust
        - feature_method: correlation/mutual_info
        - selected_features: List[str]
        - model_name: 指定模型名
        - use_grid_search: bool
        """
        decision_overrides = decision_overrides or {}
        pipe = self.pipeline
        outputs = {"stages": {}, "status": "running"}

        msg1 = pipe.explorer.explore(file_path, target_hint=target_col)
        outputs["stages"]["exploration"] = msg1.to_dict()
        if msg1.status != StageStatus.SUCCESS.value:
            outputs.update({"status": "failed", "failed_stage": "exploration", "reason": msg1.message})
            return outputs

        inferred_target = target_col or (msg1.data.get("target_candidates") or [None])[0]
        if isinstance(inferred_target, dict):
            inferred_target = inferred_target.get("column")
        if not inferred_target:
            outputs.update({"status": "failed", "failed_stage": "exploration", "reason": "目标列未确定"})
            return outputs

        inferred_features = feature_cols or msg1.data.get("feature_candidates", [])
        inferred_features = sanitize_feature_columns(pipe.explorer.raw_data, inferred_features, target_col=inferred_target)
        if not inferred_features:
            outputs.update({"status": "failed", "failed_stage": "exploration", "reason": "无可用特征列"})
            return outputs

        preprocess_strategy = decision_overrides.get("preprocessing_strategy", "auto")
        msg2 = pipe.preprocessor.preprocess(pipe.explorer.raw_data, inferred_features, [inferred_target], strategy=preprocess_strategy)
        outputs["stages"]["preprocessing"] = msg2.to_dict()
        if msg2.status != StageStatus.SUCCESS.value:
            outputs.update({"status": "failed", "failed_stage": "preprocessing", "reason": msg2.message})
            return outputs

        feature_method = decision_overrides.get("feature_method", "correlation")
        msg3 = pipe.feature_engineer.engineer_features(pipe.preprocessor.processed_data, inferred_features, inferred_target, method=feature_method)
        outputs["stages"]["feature_engineering"] = msg3.to_dict()
        if msg3.status != StageStatus.SUCCESS.value:
            outputs.update({"status": "failed", "failed_stage": "feature_engineering", "reason": msg3.message})
            return outputs

        selected_features = decision_overrides.get("selected_features", msg3.data.get("selected_features", []))
        selected_features = [f for f in selected_features if f in pipe.preprocessor.processed_data.columns]
        if not selected_features:
            outputs.update({"status": "failed", "failed_stage": "feature_engineering", "reason": "选中特征为空"})
            return outputs

        df = pipe.preprocessor.processed_data
        X = df[selected_features].values
        y = df[inferred_target].values

        if len(df) < 8:
            outputs.update({"status": "failed", "failed_stage": "split", "reason": "样本不足(<8)，无法稳定划分"})
            return outputs

        test_size = 0.2 if len(df) >= 20 else 0.25
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        val_size = 0.125 if len(X_train) >= 16 else 0.2
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

        msg4 = pipe.model_selector.recommend(len(X_train), X_train.shape[1], user_preference=model_preference)
        outputs["stages"]["model_selection"] = msg4.to_dict()
        if msg4.status != StageStatus.SUCCESS.value:
            outputs.update({"status": "failed", "failed_stage": "model_selection", "reason": msg4.message})
            return outputs

        model_name = decision_overrides.get("model_name") or msg4.data["recommended_model"]["name"]
        use_grid_search = decision_overrides.get("use_grid_search", True)
        msg5 = pipe.trainer.train(
            X_train, y_train, X_val, y_val,
            model_name=model_name,
            strategy=msg4.data["learning_strategy"]["strategy"],
            cv_folds=msg4.data["learning_strategy"]["cv_folds"],
            use_grid_search=use_grid_search
        )
        outputs["stages"]["training"] = msg5.to_dict()
        if msg5.status == StageStatus.ERROR.value:
            outputs.update({"status": "failed", "failed_stage": "training", "reason": msg5.message})
            return outputs

        pipe.evaluator.model = pipe.trainer.model
        msg6 = pipe.evaluator.evaluate(X_test, y_test, cv_folds=5)
        outputs["stages"]["evaluation"] = msg6.to_dict()

        outputs["status"] = "success"
        outputs["summary"] = {
            "target_col": inferred_target,
            "selected_features": selected_features,
            "final_model": model_name,
            "r2": msg6.data.get("metrics", {}).get("r2"),
            "rmse": msg6.data.get("metrics", {}).get("rmse")
        }
        return outputs

# ============================================================================
# 自测
# ============================================================================

def generate_test_data(n_samples: int = 500, n_features: int = 8, output_dir: str = "./test_data"):
    """生成测试数据"""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 生成目标 (非线性关系)
    y = (
        2 * X[:, 0] +
        1.5 * X[:, 1]**2 +
        0.5 * np.sin(X[:, 2] * 3) +
        X[:, 3] * X[:, 4] +
        0.3 * np.random.randn(n_samples)
    )
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # 添加一些缺失值
    mask = np.random.random(df.shape) < 0.02
    df = df.mask(mask)
    
    train_path = os.path.join(output_dir, "train_data.csv")
    df.to_csv(train_path, index=False)
    
    # 测试数据
    X_test = np.random.randn(100, n_features)
    y_test = (
        2 * X_test[:, 0] +
        1.5 * X_test[:, 1]**2 +
        0.5 * np.sin(X_test[:, 2] * 3) +
        X_test[:, 3] * X_test[:, 4]
    )
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['target'] = y_test
    
    test_path = os.path.join(output_dir, "test_data.csv")
    df_test.to_csv(test_path, index=False)
    
    return train_path, test_path


def run_self_test():
    """运行自测"""
    print("\n" + "=" * 70)
    print("AGENTIC数值预测系统 - 自测")
    print("=" * 70)
    
    # 生成测试数据
    train_path, test_path = generate_test_data(n_samples=500, n_features=8)
    
    # 创建管道并运行
    pipeline = AgenticPredictionPipeline(output_dir="./test_output")
    results = pipeline.run_full_pipeline(train_path)
    
    # 测试新数据预测
    print("\n" + "=" * 70)
    print("测试: 新数据预测")
    print("=" * 70)
    pred_msg = pipeline.predict_new(test_path)
    print(pred_msg.to_json())
    
    # 总结
    print("\n" + "=" * 70)
    print("自测总结")
    print("=" * 70)
    summary = results["summary"]
    print(f"✓ 最终模型: {summary['final_model']}")
    print(f"✓ 测试集 R²: {summary['test_r2']:.4f}")
    print(f"✓ 测试集 RMSE: {summary['test_rmse']:.4f}")
    print(f"✓ 选中特征: {summary['selected_features']}")
    print(f"✓ 预测测试: {pred_msg.status}")
    print("\n自测通过 ✓")
    
    return results


if __name__ == "__main__":
    run_self_test()
