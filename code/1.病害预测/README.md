# 1. 病害预测模块

## 目标
基于 XGBoost 构建鱼/菜发病概率预测模型，结合 SHAP 提供可解释性分析。

## 待实现
- [ ] 构建分类数据集（从 `featured_*.csv` 提取特征 + 病害标签）
- [ ] 训练 XGBoost 分类模型
- [ ] SHAP 特征重要性分析与单样本归因图
- [ ] 红光训练 → 喀左测试（交叉验证）
- [ ] 输出 Baseline F1 Score

## 输入数据
- `data/featured_红光.csv`
- `data/featured_喀左.csv`
- `data/augmented_红光.csv`
- `data/augmented_喀左.csv`
