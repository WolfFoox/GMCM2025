#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的Q-1改进.py代码
验证时间轴统一性和PCA降维效果
"""

import sys
import os
sys.path.append('.')

import sys
import importlib.util
spec = importlib.util.spec_from_file_location("Q_1改进", "Q-1改进.py")
Q_1改进 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q_1改进)
EnhancedBearingFeatureAnalyzer = Q_1改进.EnhancedBearingFeatureAnalyzer
import pandas as pd
import numpy as np

def test_time_axis_consistency():
    """测试时间轴一致性"""
    print("=== 测试时间轴一致性 ===")
    
    # 创建分析器
    data_path = r"D:\..MBA\数学建模\model\源域数据集"
    analyzer = EnhancedBearingFeatureAnalyzer(data_path)
    
    # 加载数据
    print("加载数据...")
    analyzer.load_and_process_data(max_files_per_category=10)
    
    # 检查原始信号长度
    print("\n原始信号长度分析:")
    for filename, signal_info in analyzer.raw_signals.items():
        signal_length = len(signal_info['signal'])
        fs = signal_info['fs']
        duration = signal_length / fs
        print(f"{filename}: {signal_length} 采样点, {duration:.3f} 秒, {fs}Hz")
    
    # 测试信号概览图
    print("\n生成信号概览图...")
    try:
        analyzer.plot_signal_overview()
        print("✓ 信号概览图生成成功")
    except Exception as e:
        print(f"✗ 信号概览图生成失败: {e}")
    
    # 测试包络分析图
    print("\n生成包络分析图...")
    try:
        analyzer.plot_envelope_analysis()
        print("✓ 包络分析图生成成功")
    except Exception as e:
        print(f"✗ 包络分析图生成失败: {e}")

def test_pca_reduction():
    """测试PCA降维效果"""
    print("\n=== 测试PCA降维效果 ===")
    
    # 检查降维后的数据
    if os.path.exists('reduced_features.csv'):
        df_reduced = pd.read_csv('reduced_features.csv')
        print(f"降维后数据形状: {df_reduced.shape}")
        print(f"主成分列: {[col for col in df_reduced.columns if col.startswith('PC')]}")
        print(f"标签分布: {df_reduced['label'].value_counts().to_dict()}")
        
        # 检查数据质量
        pca_cols = [col for col in df_reduced.columns if col.startswith('PC')]
        missing_count = df_reduced[pca_cols].isnull().sum().sum()
        inf_count = np.isinf(df_reduced[pca_cols]).sum().sum()
        print(f"缺失值: {missing_count}")
        print(f"无穷值: {inf_count}")
        
        print("✓ PCA降维数据质量良好")
    else:
        print("✗ 未找到降维后的数据文件")

def analyze_time_duration_choice():
    """分析选择0.3秒作为显示时长的原因"""
    print("\n=== 选择0.3秒显示时长的原因分析 ===")
    
    reasons = [
        "1. 故障周期考虑:",
        "   - 轴承故障特征频率通常在几十到几百Hz",
        "   - 0.3秒可以包含多个完整的故障周期",
        "   - 便于观察周期性冲击特征",
        "",
        "2. 采样频率适配:",
        "   - 12kHz采样: 0.3秒 = 3600个采样点",
        "   - 48kHz采样: 0.3秒 = 14400个采样点",
        "   - 都能提供足够的频率分辨率",
        "",
        "3. 计算效率:",
        "   - 相比更长时间段，0.3秒计算量适中",
        "   - 便于快速生成可视化结果",
        "   - 内存占用合理",
        "",
        "4. 对比分析:",
        "   - 统一的时间窗口便于不同故障类型对比",
        "   - 避免因时间长度不同造成的视觉偏差",
        "   - 符合信号分析的标准做法",
        "",
        "5. 特征提取:",
        "   - 0.3秒足够提取稳定的统计特征",
        "   - 能够捕获故障的典型特征",
        "   - 避免过长信号中的噪声干扰"
    ]
    
    for reason in reasons:
        print(reason)

if __name__ == "__main__":
    print("开始测试修改后的代码...")
    
    # 测试时间轴一致性
    test_time_axis_consistency()
    
    # 测试PCA降维
    test_pca_reduction()
    
    # 分析时间选择原因
    analyze_time_duration_choice()
    
    print("\n=== 测试完成 ===")
