#!/usr/bin/env python3
"""
鱼菜共生数据预处理 — 一键执行全流程
运行方式:
    source code/.venv/bin/activate
    python code/0.数据清洗/run_all.py
"""

import sys
import os
import time
import importlib
import importlib.util

# 确保 code/ 目录在 import 路径中
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)


def import_step(filename):
    """导入以数字开头的模块（Python 不允许直接 import 01_xxx）"""
    module_name = filename.replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(CODE_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()
    print('=' * 70)
    print('🐟🥬 鱼菜共生数据预处理流水线')
    print('=' * 70)

    # ── Step 1: 加载与修复 ──
    step1 = import_step('01_load_and_fix.py')
    data = step1.load_all()

    # ── Step 2: 逐小时字符串解析 ──
    step2 = import_step('02_parse_hourly.py')
    data = step2.parse_all_hourly(data)

    # ── Step 3: 发病文本解析与聚合 ──
    step3 = import_step('03_parse_disease.py')
    data = step3.parse_all_disease(data)

    # ── Step 4: 时间对齐与多表合并 ──
    step4 = import_step('04_time_align_merge.py')
    data = step4.time_align_merge(data)

    # ── Step 5: 缺失值插补 ──
    step5 = import_step('05_imputation.py')
    data = step5.impute_all(data)

    # ── Step 6: 验证与导出 ──
    step6 = import_step('06_export_report.py')
    step6.validate_and_export(data)

    elapsed = time.time() - t0
    print(f'\n{"=" * 70}')
    print(f'✅ 全流程完成! 耗时 {elapsed:.1f} 秒')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
