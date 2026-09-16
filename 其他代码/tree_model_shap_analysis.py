#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树模型多分类预测与SHAP可解释性分析
使用RF、XGBoost、LightGBM进行轴承故障诊断，并通过SHAP分析模型预测机制
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("警告: XGBoost未安装，将跳过XGBoost模型")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("警告: LightGBM未安装，将跳过LightGBM模型")
    LIGHTGBM_AVAILABLE = False
import shap
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

class TreeModelsSHAPAnalysis:
    def __init__(self, data_path='bearing_features.csv', n_splits=5, random_state=42):
        self.data_path = data_path
        self.n_splits = n_splits
        self.random_state = random_state
        self.results_dir = "tree_models_results"
        
        # 创建结果文件夹
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"创建结果文件夹: {self.results_dir}")
        
        # 加载数据
        self.load_data()
        
        # 定义特征组
        self.feature_groups = {
            'time_domain': ['rms', 'std', 'mean', 'skewness', 'kurtosis', 
                           'shape_factor', 'crest_factor', 'impulse_factor'],
            'freq_domain': ['freq_std', 'spectral_centroid', 'peak_frequency', 'freq_max',
                           'band_1_energy_ratio', 'band_3_energy_ratio', 'band_4_energy_ratio'],
            'envelope': ['envelope_mean', 'envelope_std', 'envelope_max'],
            'fault_freq': ['BPFO_energy', 'BPFI_energy', 'BSF_energy', 'FR_energy']
        }
        
    def load_data(self):
        """加载和预处理数据"""
        print("=== 加载数据 ===")
        df = pd.read_csv(self.data_path)
        
        # 数据清洗
        df = df.fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 编码标签
        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['label'])
        self.class_names = le.classes_
        
        # 获取所有可用特征
        self.all_features = [col for col in df.columns 
                           if col not in ['label', 'filename', 'rpm', 'sampling_rate', 'label_encoded']]
        
        # 过滤存在的特征
        self.available_features = [f for f in self.all_features if f in df.columns]
        
        self.X = df[self.available_features]
        self.y = df['label_encoded']
        
        print(f"数据形状: {self.X.shape}")
        print(f"类别: {self.class_names}")
        print(f"可用特征数: {len(self.available_features)}")
        
    def get_optimal_features(self):
        """基于消融实验结果获取最优特征组合"""
        # 基于之前的消融实验结果，选择最优的11个特征
        optimal_features = [
            'freq_std', 'spectral_centroid', 'peak_frequency', 'freq_max',
            'band_1_energy_ratio', 'band_3_energy_ratio', 'band_4_energy_ratio',
            'BPFO_energy', 'BPFI_energy', 'BSF_energy', 'FR_energy'
        ]
        
        # 确保所有特征都存在
        available_optimal = [f for f in optimal_features if f in self.available_features]
        print(f"使用最优特征组合: {len(available_optimal)} 个特征")
        return available_optimal
    
    def train_tree_models(self):
        """训练三种树模型"""
        print("\n=== 训练树模型 ===")
        
        # 获取最优特征
        optimal_features = self.get_optimal_features()
        X_optimal = self.X[optimal_features]
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_optimal)
        
        # 定义模型（根据可用性动态添加）
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced'
            )
        }
        
        # 添加XGBoost模型（如果可用）
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=len(self.class_names),
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                verbosity=0  # 减少输出
            )
        
        # 添加LightGBM模型（如果可用）
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=len(self.class_names),
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                class_weight='balanced',
                verbosity=-1  # 减少输出
            )
        
        # 交叉验证评估
        cv_results = {}
        trained_models = {}
        
        # 使用分层K折交叉验证确保每折都有各类别样本
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            print(f"\n训练 {name}...")
            
            # 详细的交叉验证评估
            accuracy_scores = []
            f1_scores = []
            precision_scores = []
            recall_scores = []
            
            # 手动进行交叉验证以获取更详细的结果
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, self.y)):
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val_fold = self.y[train_idx], self.y[val_idx]
                
                # 训练模型
                model.fit(X_train_fold, y_train_fold)
                
                # 预测
                y_pred_fold = model.predict(X_val_fold)
                
                # 计算指标
                accuracy_scores.append(accuracy_score(y_val_fold, y_pred_fold))
                f1_scores.append(f1_score(y_val_fold, y_pred_fold, average='weighted'))
                precision_scores.append(f1_score(y_val_fold, y_pred_fold, average='weighted'))
                recall_scores.append(f1_score(y_val_fold, y_pred_fold, average='weighted'))
            
            # 训练完整模型用于SHAP分析
            model.fit(X_scaled, self.y)
            trained_models[name] = model
            
            cv_results[name] = {
                'accuracy': {
                    'mean': np.mean(accuracy_scores),
                    'std': np.std(accuracy_scores),
                    'scores': accuracy_scores
                },
                'f1': {
                    'mean': np.mean(f1_scores),
                    'std': np.std(f1_scores),
                    'scores': f1_scores
                },
                'precision': {
                    'mean': np.mean(precision_scores),
                    'std': np.std(precision_scores),
                    'scores': precision_scores
                },
                'recall': {
                    'mean': np.mean(recall_scores),
                    'std': np.std(recall_scores),
                    'scores': recall_scores
                }
            }
            
            print(f"  准确率: {cv_results[name]['accuracy']['mean']:.4f} ± {cv_results[name]['accuracy']['std']:.4f}")
            print(f"  F1分数: {cv_results[name]['f1']['mean']:.4f} ± {cv_results[name]['f1']['std']:.4f}")
            print(f"  精确率: {cv_results[name]['precision']['mean']:.4f} ± {cv_results[name]['precision']['std']:.4f}")
            print(f"  召回率: {cv_results[name]['recall']['mean']:.4f} ± {cv_results[name]['recall']['std']:.4f}")
        
        self.cv_results = cv_results
        self.trained_models = trained_models
        self.scaler = scaler
        self.optimal_features = optimal_features
        self.X_scaled = X_scaled
        
        return cv_results, trained_models
    
    def plot_model_comparison(self):
        """绘制模型性能对比图"""
        print("\n=== 绘制模型性能对比 ===")
        
        # 准备数据
        model_names = list(self.cv_results.keys())
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('树模型性能对比分析', fontsize=16)
        
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(model_names)]
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            means = [self.cv_results[name][metric]['mean'] for name in model_names]
            stds = [self.cv_results[name][metric]['std'] for name in model_names]
            
            bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'模型{metric.capitalize()}对比')
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建交叉验证稳定性分析图
        self.plot_cv_stability()
        
        return fig
    
    def plot_cv_stability(self):
        """绘制交叉验证稳定性分析"""
        print("生成交叉验证稳定性分析图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 各折性能变化
        ax1 = axes[0]
        for i, (name, results) in enumerate(self.cv_results.items()):
            ax1.plot(range(1, self.n_splits+1), results['accuracy']['scores'], 
                    marker='o', label=name, linewidth=2, markersize=6)
        
        ax1.set_xlabel('交叉验证折数')
        ax1.set_ylabel('准确率')
        ax1.set_title('各折性能变化趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 性能稳定性箱线图
        ax2 = axes[1]
        accuracy_data = [results['accuracy']['scores'] for results in self.cv_results.values()]
        model_names = list(self.cv_results.keys())
        
        box_plot = ax2.boxplot(accuracy_data, labels=model_names, patch_artist=True)
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(model_names)]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('准确率')
        ax2.set_title('模型性能稳定性分析')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'cv_stability_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def shap_analysis(self, model_name='RandomForest'):
        """SHAP可解释性分析"""
        print(f"\n=== SHAP分析 ({model_name}) ===")
        
        model = self.trained_models[model_name]
        
        # 创建SHAP解释器
        try:
            if model_name == 'RandomForest':
                explainer = shap.TreeExplainer(model)
            else:  # XGBoost 和 LightGBM
                explainer = shap.TreeExplainer(model)
        except Exception as e:
            print(f"创建SHAP解释器失败: {e}")
            return None
        
        # 计算SHAP值
        print("计算SHAP值...")
        try:
            shap_values = explainer.shap_values(self.X_scaled)
            print(f"SHAP值形状: {np.array(shap_values).shape}")
        except Exception as e:
            print(f"计算SHAP值失败: {e}")
            return None
        
        # 1. 全局特征重要性
        print("生成全局特征重要性图...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, self.X_scaled, 
                         feature_names=self.optimal_features, 
                         class_names=self.class_names, 
                         show=False)
        plt.title(f'SHAP全局特征重要性 - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'shap_summary_{model_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 特征重要性条形图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, self.X_scaled, 
                         plot_type="bar", 
                         feature_names=self.optimal_features, 
                         class_names=self.class_names, 
                         show=False)
        plt.title(f'SHAP平均绝对值特征重要性 - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'shap_bar_{model_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 特征交互依赖分析
        print("生成特征交互依赖图...")
        self.plot_feature_interactions(shap_values, model_name)
        
        # 4. 特征重要性排序
        importance_df = self.analyze_feature_importance(shap_values, model_name)
        
        # 5. 特征交互作用分析
        interactions = self.analyze_feature_interactions(shap_values, model_name)
        
        # 6. 模型改进建议
        self.generate_improvement_suggestions(importance_df, interactions, model_name)
        
        return shap_values
    
    def plot_feature_interactions(self, shap_values, model_name):
        """绘制特征交互依赖图"""
        # 计算特征重要性
        if len(shap_values.shape) == 3:  # (n_classes, n_samples, n_features)
            avg_abs_shap = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
        else:  # (n_samples, n_features)
            avg_abs_shap = np.abs(np.array(shap_values)).mean(axis=0)
        
        # 确保长度匹配
        if len(avg_abs_shap) != len(self.optimal_features):
            min_len = min(len(avg_abs_shap), len(self.optimal_features))
            avg_abs_shap = avg_abs_shap[:min_len]
            features_to_use = self.optimal_features[:min_len]
        else:
            features_to_use = self.optimal_features
        
        top_feature_indices = np.argsort(avg_abs_shap)[::-1][:3]  # 只选择前3个特征
        top_features = [features_to_use[i] for i in top_feature_indices]
        
        print(f"分析前3个重要特征的交互作用: {top_features}")
        
        # 简化依赖图生成 - 只生成总体依赖图，不按类别分别生成
        for i, feature_name in enumerate(top_features):
            print(f"  生成特征 '{feature_name}' 的依赖图...")
            
            try:
                plt.figure(figsize=(10, 6))
                
                # 使用平均SHAP值
                if len(shap_values.shape) == 3:  # (n_classes, n_samples, n_features)
                    shap_values_avg = np.mean(shap_values, axis=0)  # 平均所有类别的SHAP值
                else:  # (n_samples, n_features)
                    shap_values_avg = shap_values
                
                # 找到特征索引
                feature_idx = features_to_use.index(feature_name)
                
                shap.dependence_plot(
                    feature_idx,
                    shap_values_avg, 
                    self.X_scaled, 
                    feature_names=features_to_use,
                    interaction_index="auto", 
                    show=False
                )
                plt.title(f'SHAP依赖图: {feature_name} - {model_name}', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 
                                       f'shap_dependence_{feature_name}_{model_name.lower()}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    成功生成 {feature_name} 的依赖图")
                
            except Exception as e:
                print(f"    生成 {feature_name} 的依赖图失败: {e}")
                continue
    
    def analyze_feature_importance(self, shap_values, model_name):
        """分析特征重要性"""
        # 计算平均绝对SHAP值
        if len(shap_values.shape) == 3:  # (n_classes, n_samples, n_features)
            avg_abs_shap = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
        else:  # (n_samples, n_features)
            avg_abs_shap = np.abs(np.array(shap_values)).mean(axis=0)
        
        # 确保长度匹配
        if len(avg_abs_shap) != len(self.optimal_features):
            print(f"警告: SHAP值长度({len(avg_abs_shap)})与特征数量({len(self.optimal_features)})不匹配")
            min_len = min(len(avg_abs_shap), len(self.optimal_features))
            avg_abs_shap = avg_abs_shap[:min_len]
            features_to_use = self.optimal_features[:min_len]
        else:
            features_to_use = self.optimal_features
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': features_to_use,
            'importance': avg_abs_shap
        }).sort_values('importance', ascending=False)
        
        print(f"\n{model_name} 特征重要性排名:")
        for i, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"{i:2d}. {row['feature']:20s}: {row['importance']:.4f}")
        
        # 保存特征重要性
        importance_df.to_csv(os.path.join(self.results_dir, f'feature_importance_{model_name.lower()}.csv'), 
                           index=False, encoding='utf-8-sig')
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'], alpha=0.7)
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('平均绝对SHAP值')
        plt.title(f'特征重要性排序 - {model_name}')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'feature_importance_plot_{model_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return importance_df
    
    def analyze_feature_interactions(self, shap_values, model_name):
        """深入分析特征交互作用"""
        print(f"\n=== 特征交互作用分析 ({model_name}) ===")
        
        # 处理SHAP值形状
        if len(shap_values.shape) == 3:  # (n_classes, n_samples, n_features)
            shap_values_avg = np.abs(np.array(shap_values)).mean(axis=0)  # 平均所有类别的SHAP值
        else:  # (n_samples, n_features)
            shap_values_avg = np.abs(np.array(shap_values))
        
        # 确保特征数量匹配
        if shap_values_avg.shape[1] != len(self.optimal_features):
            min_len = min(shap_values_avg.shape[1], len(self.optimal_features))
            shap_values_avg = shap_values_avg[:, :min_len]
            features_to_use = self.optimal_features[:min_len]
        else:
            features_to_use = self.optimal_features
        
        # 计算特征间的SHAP值相关性
        try:
            shap_corr = np.corrcoef(shap_values_avg.T)
            
            # 绘制交互热力图
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(shap_corr, dtype=bool))
            sns.heatmap(shap_corr, 
                       mask=mask,
                       xticklabels=features_to_use,
                       yticklabels=features_to_use,
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       square=True,
                       fmt='.2f',
                       cbar_kws={"shrink": .8})
            plt.title(f'SHAP值特征交互热力图 - {model_name}', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'shap_interaction_heatmap_{model_name.lower()}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 寻找强交互特征对
            strong_interactions = []
            for i in range(len(features_to_use)):
                for j in range(i+1, len(features_to_use)):
                    corr_val = shap_corr[i, j]
                    if abs(corr_val) > 0.5:  # 强相关阈值
                        strong_interactions.append({
                            'feature1': features_to_use[i],
                            'feature2': features_to_use[j],
                            'correlation': corr_val
                        })
            
            # 按相关性强度排序
            strong_interactions.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            print(f"发现 {len(strong_interactions)} 对强交互特征:")
            for interaction in strong_interactions[:10]:  # 显示前10对
                print(f"  {interaction['feature1']} <-> {interaction['feature2']}: {interaction['correlation']:.3f}")
            
            # 保存交互分析结果
            if strong_interactions:
                interaction_df = pd.DataFrame(strong_interactions)
                interaction_df.to_csv(os.path.join(self.results_dir, f'feature_interactions_{model_name.lower()}.csv'), 
                                    index=False, encoding='utf-8-sig')
            else:
                print("未发现强交互特征对")
                strong_interactions = []
            
        except Exception as e:
            print(f"特征交互分析失败: {e}")
            strong_interactions = []
        
        return strong_interactions
    
    def generate_improvement_suggestions(self, importance_df, interactions, model_name):
        """基于SHAP分析生成模型改进建议"""
        print(f"\n=== 模型改进建议 ({model_name}) ===")
        
        # 分析高重要性特征
        top_features = importance_df.head(5)['feature'].tolist()
        print(f"前5个最重要特征: {top_features}")
        
        # 分析强交互特征
        strong_interactions = [i for i in interactions if abs(i['correlation']) > 0.5]
        print(f"发现 {len(strong_interactions)} 对强交互特征")
        
        # 生成改进建议
        suggestions = []
        
        # 1. 特征工程建议
        if len(top_features) > 0:
            suggestions.append(f"重点关注 {top_features[0]} 等高频域特征，考虑创建更多相关衍生特征")
        
        # 2. 特征交互建议
        if len(strong_interactions) > 0:
            top_interaction = strong_interactions[0]
            suggestions.append(f"考虑创建 {top_interaction['feature1']} 和 {top_interaction['feature2']} 的交互特征")
        
        # 3. 模型优化建议
        suggestions.append("尝试集成多个模型以提高预测性能")
        suggestions.append("考虑使用更复杂的特征选择方法")
        suggestions.append("增加训练数据以提高模型泛化能力")
        
        # 保存建议
        suggestions_file = os.path.join(self.results_dir, f'improvement_suggestions_{model_name.lower()}.txt')
        with open(suggestions_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {model_name} 模型改进建议 ===\n\n")
            f.write("1. 特征工程建议:\n")
            f.write(f"   - 重点关注 {top_features[0]} 等高频域特征\n")
            f.write("   - 考虑创建更多时频域特征\n")
            f.write("   - 优化故障频率特征的提取方法\n\n")
            
            f.write("2. 特征交互建议:\n")
            if strong_interactions:
                for i, interaction in enumerate(strong_interactions[:3], 1):
                    f.write(f"   - 创建 {interaction['feature1']} 和 {interaction['feature2']} 的交互特征\n")
            else:
                f.write("   - 当前特征交互较弱，可考虑特征组合\n\n")
            
            f.write("3. 模型优化建议:\n")
            f.write("   - 尝试集成多个模型\n")
            f.write("   - 使用更复杂的特征选择方法\n")
            f.write("   - 增加训练数据\n")
            f.write("   - 调整模型超参数\n")
        
        print("改进建议已保存到:", suggestions_file)
        return suggestions
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n=== 生成综合分析报告 ===")
        
        report_path = os.path.join(self.results_dir, 'comprehensive_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 轴承故障诊断树模型与SHAP分析报告 ===\n\n")
            
            # 1. 实验设置
            f.write("1. 实验设置\n")
            f.write(f"   数据集: {self.data_path}\n")
            f.write(f"   交叉验证折数: {self.n_splits}\n")
            f.write(f"   特征数量: {len(self.optimal_features)}\n")
            f.write(f"   类别: {', '.join(self.class_names)}\n\n")
            
            # 2. 模型性能对比
            f.write("2. 模型性能对比\n")
            f.write("   按准确率排序:\n")
            sorted_models = sorted(self.cv_results.items(), 
                                 key=lambda x: x[1]['accuracy']['mean'], reverse=True)
            
            for i, (name, results) in enumerate(sorted_models, 1):
                f.write(f"   {i}. {name:15s}: "
                       f"准确率 {results['accuracy']['mean']:.4f}±{results['accuracy']['std']:.4f}, "
                       f"F1 {results['f1']['mean']:.4f}±{results['f1']['std']:.4f}\n")
            
            # 3. 最佳模型详情
            best_model_name = sorted_models[0][0]
            f.write(f"\n3. 最佳模型: {best_model_name}\n")
            f.write(f"   准确率: {self.cv_results[best_model_name]['accuracy']['mean']:.4f} ± {self.cv_results[best_model_name]['accuracy']['std']:.4f}\n")
            f.write(f"   F1分数: {self.cv_results[best_model_name]['f1']['mean']:.4f} ± {self.cv_results[best_model_name]['f1']['std']:.4f}\n")
            
            # 4. 特征重要性分析
            f.write(f"\n4. 特征重要性分析 (基于{best_model_name})\n")
            f.write("   前10个最重要特征:\n")
            
            # 读取特征重要性文件
            importance_file = os.path.join(self.results_dir, f'feature_importance_{best_model_name.lower()}.csv')
            if os.path.exists(importance_file):
                importance_df = pd.read_csv(importance_file)
                for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                    f.write(f"   {i:2d}. {row['feature']:20s}: {row['importance']:.4f}\n")
            
            # 5. 改进建议
            f.write("\n5. 模型改进建议\n")
            f.write("   基于SHAP分析结果，建议:\n")
            f.write("   - 重点关注高重要性特征，如频域特征和故障频率特征\n")
            f.write("   - 考虑特征间的交互作用，优化特征组合\n")
            f.write("   - 可以尝试集成多个模型提高预测性能\n")
            f.write("   - 进一步收集数据以提高模型泛化能力\n")
        
        print(f"综合分析报告已保存到: {report_path}")
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("=== 开始完整的树模型与SHAP分析 ===")
        
        # 1. 训练树模型
        self.train_tree_models()
        
        # 2. 绘制模型对比
        self.plot_model_comparison()
        
        # 3. 对每个模型进行SHAP分析
        for model_name in self.trained_models.keys():
            print(f"\n对 {model_name} 进行SHAP分析...")
            shap_values = self.shap_analysis(model_name)
            self.analyze_feature_interactions(shap_values, model_name)
        
        # 4. 生成综合报告
        self.generate_comprehensive_report()
        
        print(f"\n=== 分析完成 ===")
        print(f"所有结果已保存到 '{self.results_dir}' 文件夹")
        print("生成的文件包括:")
        print("- model_comparison.png: 模型性能对比")
        print("- shap_summary_*.png: SHAP全局特征重要性")
        print("- shap_bar_*.png: 特征重要性条形图")
        print("- shap_dependence_*.png: 特征依赖图")
        print("- feature_importance_*.png: 特征重要性排序")
        print("- shap_interaction_heatmap_*.png: 特征交互热力图")
        print("- comprehensive_analysis_report.txt: 综合分析报告")

def main():
    """主函数"""
    # 检查数据文件是否存在
    if not os.path.exists('bearing_features.csv'):
        print("错误: 未找到 bearing_features.csv 文件")
        print("请先运行 Q-1改进.py 生成特征数据")
        return
    
    # 创建分析对象
    analyzer = TreeModelsSHAPAnalysis()
    
    # 运行完整分析
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()