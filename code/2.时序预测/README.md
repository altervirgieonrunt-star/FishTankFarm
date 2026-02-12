# 2. 时序预测模块

## 目标
利用 Chronos 时序基座模型进行 Zero-Shot 环境参数预测（水温、溶氧、氨氮）。

## 待实现
- [ ] 安装 Chronos（`pip install chronos-forecasting`）
- [ ] Zero-Shot 预测（chronos-t5-tiny，72h 预测窗口）
- [ ] 评估指标：MAE / RMSE / 置信区间覆盖率
- [ ] 绘制真实值 vs 预测值对比曲线
- [ ] （可选）Fine-tune 提升精度

## 输入数据
- `data/cleaned_红光.csv`
- `data/cleaned_喀左.csv`
