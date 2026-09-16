#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP特征解释与交叉验证分析
使用TreeExplainer计算多分类情形下的Shapley值，进行交互依赖分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 确保results文件夹存在
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"创建结果文件夹: {results_dir}")

print("=== SHAP 特征解释与交叉验证分析 ===")

# 1. 加载数据
try:
    df = pd.read_csv('bearing_features.csv')
    print(f"成功加载 bearing_features.csv，共 {len(df)} 个样本，{len(df.columns)-4} 个原始特征。")
except FileNotFoundError:
    print("错误：未找到 bearing_features.csv 文件。")
    exit()

# 检查标签分布
print("\n原始标签分布:")
print(df['label'].value_counts())

# 2. 数据预处理
# 填充NaN和Inf值
df = df.fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)

# 编码标签
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
class_names = le.classes_
print(f"\n编码后的类别名称: {class_names}")

# 3. 特征选择 - 基于PCA分析结果选择关键特征
# 选择对前8个主成分贡献最大的特征，避免冗余
selected_features = [
    # 时域核心特征
    'rms', 'std', 'mean', 'skewness', 'kurtosis', 
    'shape_factor', 'crest_factor', 'impulse_factor',
    
    # 频域核心特征
    'freq_std', 'spectral_centroid', 'peak_frequency', 'freq_max',
    'band_1_energy_ratio', 'band_3_energy_ratio', 'band_4_energy_ratio',
    
    # 包络特征
    'envelope_mean', 'envelope_std',
    
    # 故障频率特征
    'BPFO_energy', 'BPFI_energy', 'BSF_energy'
]

# 确保所有选定特征都在DataFrame中
missing_features = [f for f in selected_features if f not in df.columns]
if missing_features:
    print(f"警告：以下选定特征在数据中缺失，将跳过: {missing_features}")
    selected_features = [f for f in selected_features if f in df.columns]

if not selected_features:
    print("错误：没有可用的特征进行分析。")
    exit()

X = df[selected_features]
y = df['label_encoded']

print(f"\n已选择 {len(selected_features)} 个特征进行SHAP分析:")
for i, feat in enumerate(selected_features, 1):
    print(f"{i:2d}. {feat}")

# 4. 交叉验证设置
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

accuracy_scores = []
f1_scores = []
models = []
X_test_folds = []
y_test_folds = []

print(f"\n=== 开始 {n_splits} 折交叉验证 ===")
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"\n--- 折叠 {fold+1}/{n_splits} ---")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练XGBoost分类器
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(class_names),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    model.fit(X_train_scaled, y_train)
    models.append((model, scaler))
    X_test_folds.append(X_test_scaled)
    y_test_folds.append(y_test)

    # 评估模型
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy_scores.append(acc)
    f1_scores.append(f1)

    print(f"  准确率: {acc:.4f}")
    print(f"  F1分数 (加权): {f1:.4f}")

print(f"\n=== 交叉验证结果 ===")
print(f"平均准确率: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"平均F1分数: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# 5. 选择最佳模型进行SHAP分析
best_model_idx = np.argmax(accuracy_scores)
best_model, best_scaler = models[best_model_idx]
best_X_test = X_test_folds[best_model_idx]
best_y_test = y_test_folds[best_model_idx]

print(f"\n使用第 {best_model_idx+1} 折的模型进行SHAP分析 (准确率: {accuracy_scores[best_model_idx]:.4f})")

# 6. SHAP值计算与可视化
print("\n=== SHAP 特征解释 ===")

# 创建TreeExplainer
explainer = shap.TreeExplainer(best_model)

# 计算SHAP值
print("计算SHAP值...")
shap_values = explainer.shap_values(best_X_test)

# 6.1 全局特征重要性 (Summary Plot)
print("生成SHAP全局特征重要性图...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, best_X_test, 
                  feature_names=selected_features, 
                  class_names=class_names, 
                  show=False)
plt.title('SHAP 全局特征重要性 (Summary Plot)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.2 特征重要性条形图
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, best_X_test, 
                  plot_type="bar", 
                  feature_names=selected_features, 
                  class_names=class_names, 
                  show=False)
plt.title('SHAP 平均绝对值特征重要性 (Bar Plot)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'shap_bar_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.3 特征交互依赖分析
print("生成SHAP特征交互依赖图...")

# 计算每个特征的平均绝对SHAP值
avg_abs_shap_values = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
top_feature_indices = np.argsort(avg_abs_shap_values)[::-1][:4]  # 前4个最重要特征
top_features_for_dependence = [selected_features[i] for i in top_feature_indices]

print(f"选择前4个最重要特征进行依赖分析: {top_features_for_dependence}")

for i, feature_name in enumerate(top_features_for_dependence):
    print(f"  生成特征 '{feature_name}' 的依赖图...")
    
    # 为每个类别生成依赖图
    for class_idx, class_name in enumerate(class_names):
        plt.figure(figsize=(10, 6))
        # 修复：使用正确的SHAP值形状
        shap.dependence_plot(
            feature_name, 
            shap_values[class_idx],  # 这是 (n_samples, n_features) 形状
            best_X_test, 
            feature_names=selected_features,
            interaction_index="auto", 
            show=False
        )
        plt.title(f'SHAP 依赖图: {feature_name} (类别: {class_name})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'shap_dependence_{feature_name}_{class_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

# 6.4 特征交互热力图
print("生成特征交互热力图...")
plt.figure(figsize=(12, 10))

# 计算特征间的SHAP值相关性
shap_corr = np.corrcoef(np.abs(shap_values).mean(axis=0).T)
mask = np.triu(np.ones_like(shap_corr, dtype=bool))

sns.heatmap(shap_corr, 
            mask=mask,
            xticklabels=selected_features,
            yticklabels=selected_features,
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8})
plt.title('SHAP值特征交互热力图', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'shap_interaction_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.5 混淆矩阵
print("生成混淆矩阵...")
y_pred_best = best_model.predict(best_X_test)
cm = confusion_matrix(best_y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('混淆矩阵 (最佳模型)', fontsize=16)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.6 特征重要性排序
print("\n=== 特征重要性排序 ===")
feature_importance = avg_abs_shap_values
importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("特征重要性排名:")
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    print(f"{i:2d}. {row['feature']:20s}: {row['importance']:.4f}")

# 保存特征重要性结果
importance_df.to_csv(os.path.join(results_dir, 'feature_importance_ranking.csv'), 
                    index=False, encoding='utf-8-sig')

print(f"\n=== 分析完成 ===")
print(f"所有结果已保存到 '{results_dir}' 文件夹")
print("生成的文件包括:")
print("- shap_summary_plot.png: 全局特征重要性")
print("- shap_bar_plot.png: 特征重要性条形图")
print("- shap_dependence_*.png: 特征依赖图")
print("- shap_interaction_heatmap.png: 特征交互热力图")
print("- confusion_matrix.png: 混淆矩阵")
print("- feature_importance_ranking.csv: 特征重要性排名")
