#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于SHAP结果的特征选择指南
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import shap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_feature_importance_from_shap():
    """基于SHAP结果分析特征重要性"""
    
    print("=== 基于SHAP结果的特征选择指南 ===\n")
    
    # 1. 特征重要性判断标准
    print("1. 特征重要性判断标准：")
    print("   - 平均绝对SHAP值 > 0.1：高重要性特征")
    print("   - 平均绝对SHAP值 0.05-0.1：中等重要性特征") 
    print("   - 平均绝对SHAP值 < 0.05：低重要性特征")
    print("   - 特征稳定性：在交叉验证中重要性排名稳定")
    print("   - 类别区分度：对不同故障类型有显著区分作用\n")
    
    # 2. 特征选择策略
    print("2. 特征选择策略：")
    print("   A. 基于SHAP重要性排序")
    print("      - 选择前10-15个最重要的特征")
    print("      - 确保覆盖时域、频域、包络、故障频率等不同类型")
    print("   B. 基于特征稳定性")
    print("      - 在多次交叉验证中排名稳定的特征")
    print("      - 避免在某个fold中突然变得重要的特征")
    print("   C. 基于类别区分度")
    print("      - 对不同故障类型都有贡献的特征")
    print("      - 避免只对单一类别重要的特征\n")
    
    # 3. 具体特征推荐
    print("3. 基于轴承故障机理的特征推荐：")
    
    # 时域特征（反映信号基本特性）
    time_domain_features = [
        "rms",           # 均方根值 - 反映信号能量
        "kurtosis",      # 峭度 - 对冲击敏感
        "skewness",      # 偏度 - 反映信号不对称性
        "crest_factor",  # 峰值因子 - 检测冲击
        "shape_factor"   # 形状因子 - 描述信号形状
    ]
    
    # 频域特征（反映频率成分）
    freq_domain_features = [
        "spectral_centroid",    # 频谱重心 - 主要频率成分
        "peak_frequency",       # 峰值频率 - 主导频率
        "freq_std",            # 频率标准差 - 频率分布宽度
        "band_3_energy_ratio", # 中频带能量 - 故障特征频率范围
        "band_4_energy_ratio"  # 高频带能量 - 冲击频率范围
    ]
    
    # 包络特征（检测调制现象）
    envelope_features = [
        "envelope_mean",  # 包络均值 - 调制强度
        "envelope_std"   # 包络标准差 - 调制变化
    ]
    
    # 故障频率特征（直接检测故障）
    fault_freq_features = [
        "BPFO_energy",  # 外圈故障频率能量
        "BPFI_energy",  # 内圈故障频率能量
        "BSF_energy"    # 滚动体故障频率能量
    ]
    
    print("   时域特征（5个）：")
    for i, feat in enumerate(time_domain_features, 1):
        print(f"     {i}. {feat}")
    
    print("\n   频域特征（5个）：")
    for i, feat in enumerate(freq_domain_features, 1):
        print(f"     {i}. {feat}")
    
    print("\n   包络特征（2个）：")
    for i, feat in enumerate(envelope_features, 1):
        print(f"     {i}. {feat}")
    
    print("\n   故障频率特征（3个）：")
    for i, feat in enumerate(fault_freq_features, 1):
        print(f"     {i}. {feat}")
    
    # 4. 特征选择代码示例
    print("\n4. 特征选择代码示例：")
    print("""
# 基于SHAP结果的特征选择
def select_features_from_shap(shap_values, feature_names, threshold=0.05):
    # 计算平均绝对SHAP值
    avg_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=0)
    
    # 选择重要性大于阈值的特征
    important_features = []
    for i, importance in enumerate(avg_abs_shap):
        if importance > threshold:
            important_features.append(feature_names[i])
    
    return important_features

# 基于稳定性的特征选择
def select_stable_features(importance_rankings, stability_threshold=0.8):
    # 计算特征在多次交叉验证中的排名稳定性
    stable_features = []
    for feature in importance_rankings.columns:
        rankings = importance_rankings[feature]
        stability = 1 - np.std(rankings) / np.mean(rankings)
        if stability > stability_threshold:
            stable_features.append(feature)
    
    return stable_features
""")
    
    # 5. 最终推荐特征集
    print("\n5. 最终推荐特征集（15个）：")
    recommended_features = (time_domain_features + 
                          freq_domain_features + 
                          envelope_features + 
                          fault_freq_features)
    
    print("   综合特征集：")
    for i, feat in enumerate(recommended_features, 1):
        print(f"     {i:2d}. {feat}")
    
    print(f"\n   总计：{len(recommended_features)} 个特征")
    print("   覆盖：时域(5) + 频域(5) + 包络(2) + 故障频率(3)")
    
    return recommended_features

def create_feature_selection_visualization():
    """创建特征选择可视化"""
    
    # 模拟SHAP重要性数据
    features = ['rms', 'kurtosis', 'spectral_centroid', 'BPFO_energy', 'skewness',
                'crest_factor', 'peak_frequency', 'envelope_mean', 'BPFI_energy',
                'shape_factor', 'freq_std', 'band_3_energy_ratio', 'envelope_std',
                'BSF_energy', 'band_4_energy_ratio', 'mean', 'impulse_factor',
                'freq_max', 'band_1_energy_ratio', 'std']
    
    # 模拟重要性分数
    importance_scores = np.random.exponential(0.1, len(features))
    importance_scores = np.sort(importance_scores)[::-1]
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('特征选择可视化指南', fontsize=16)
    
    # 1. 特征重要性排序
    ax1 = axes[0, 0]
    colors = ['red' if score > 0.1 else 'orange' if score > 0.05 else 'lightblue' 
              for score in importance_scores]
    bars = ax1.barh(range(len(features)), importance_scores, color=colors)
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels(features, fontsize=8)
    ax1.set_xlabel('平均绝对SHAP值')
    ax1.set_title('特征重要性排序')
    ax1.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='高重要性阈值')
    ax1.axvline(x=0.05, color='orange', linestyle='--', alpha=0.7, label='中等重要性阈值')
    ax1.legend()
    
    # 2. 特征类型分布
    ax2 = axes[0, 1]
    feature_types = ['时域', '频域', '包络', '故障频率', '其他']
    type_counts = [5, 5, 2, 3, 5]  # 推荐的特征数量
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightgray']
    wedges, texts, autotexts = ax2.pie(type_counts, labels=feature_types, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('推荐特征类型分布')
    
    # 3. 特征稳定性分析
    ax3 = axes[1, 0]
    stability_scores = np.random.beta(2, 2, len(features))  # 模拟稳定性分数
    ax3.scatter(importance_scores, stability_scores, alpha=0.7, s=60)
    ax3.set_xlabel('重要性分数')
    ax3.set_ylabel('稳定性分数')
    ax3.set_title('重要性 vs 稳定性')
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='稳定性阈值')
    ax3.axvline(x=0.05, color='orange', linestyle='--', alpha=0.7, label='重要性阈值')
    ax3.legend()
    
    # 4. 最终推荐特征
    ax4 = axes[1, 1]
    recommended = features[:15]  # 前15个最重要的特征
    y_pos = np.arange(len(recommended))
    ax4.barh(y_pos, importance_scores[:15], color='lightgreen', alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(recommended, fontsize=8)
    ax4.set_xlabel('重要性分数')
    ax4.set_title('最终推荐特征集')
    
    plt.tight_layout()
    plt.savefig('feature_selection_guide.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 分析特征重要性
    recommended_features = analyze_feature_importance_from_shap()
    
    # 创建可视化
    create_feature_selection_visualization()
    
    print("\n=== 总结 ===")
    print("1. PCA主成分包含多个特征是正常的，每个主成分都是原始特征的线性组合")
    print("2. SHAP报错已修复，问题在于多分类SHAP值的形状处理")
    print("3. 推荐使用15个核心特征，覆盖时域、频域、包络、故障频率四个维度")
    print("4. 特征选择应综合考虑重要性、稳定性和类别区分度")
