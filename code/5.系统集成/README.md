# 5. 系统集成与可视化

## 目标
构建 Streamlit Dashboard，集成所有模型，实现"随机数据 → 预测 → 可视化"完整演示。

## 待实现
- [ ] Streamlit/Gradio 可视化界面开发
- [ ] 封装鲁棒预处理管道（robust_preprocess）
- [ ] "随机数据 → 模型预测 → 可视化"演示 Pipeline
- [ ] ONNX 模型导出与推理速度验证
- [ ] 打包 requirements.txt + 一键运行脚本
- [ ] 压力测试：模拟异常数据输入验证系统不崩溃

## 依赖模块
- `1.病害预测/` — XGBoost 模型
- `2.时序预测/` — Chronos 模型
- `3.PINN/` — PINN 模型
- `4.因果发现/` — 因果图
