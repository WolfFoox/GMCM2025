#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树模型多分类预测与SHAP可解释性分析 - 适配新版特征提取
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
    def __init__(self, data_path=r'../02_特征提取/comprehensive_features.csv', n_splits=5, random_state=42):
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

        # 定义特征组 - 适配新版特征
        self.feature_groups = {
            'time_domain': ['mean', 'std', 'rms', 'max', 'min', 'peak_to_peak',
                            'skewness', 'kurtosis', 'energy', 'power',
                            'impulse_factor', 'crest_factor', 'shape_factor',
                            'impact_energy_ratio', 'cyclostationarity'],

            'freq_domain': ['spectral_centroid', 'spectral_spread', 'spectral_rolloff',
                            'spectral_flux', 'psd_mean', 'psd_std', 'psd_max',
                            'psd_peak_freq', 'order_1_amplitude', 'order_2_amplitude',
                            'order_3_amplitude', 'order_energy'],

            'time_freq_domain': ['envelope_mean', 'envelope_std', 'envelope_kurtosis',
                                 'envelope_skewness', 'vmd_mode_1_energy', 'vmd_mode_1_std',
                                 'vmd_mode_1_kurtosis', 'vmd_mode_2_energy', 'vmd_mode_2_std',
                                 'vmd_mode_2_kurtosis', 'vmd_mode_3_energy', 'vmd_mode_3_std',
                                 'vmd_mode_3_kurtosis', 'vmd_reconstruction_error',
                                 'wavelet_scale_2_energy', 'wavelet_scale_2_std',
                                 'wavelet_scale_4_energy', 'wavelet_scale_4_std',
                                 'wavelet_scale_8_energy', 'wavelet_scale_8_std',
                                 'wavelet_scale_16_energy', 'wavelet_scale_16_std']
        }

    def load_data(self):
        """加载和预处理数据"""
        print("=== 加载数据 ===")
        try:
            df = pd.read_csv(self.data_path)
            # 只使用源域数据进行消融实验
            df = df[df['domain'] == 'Source'].copy()
            print(f"成功加载数据，形状: {df.shape}")
        except FileNotFoundError:
            print(f"错误: 未找到数据文件 {self.data_path}")
            print("请先运行 comprehensive_feature_extraction.py 生成特征数据")
            return

        # 数据清洗
        df = df.fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)

        # 检查必要的列
        if 'label' not in df.columns:
            print("错误: 数据中缺少 'label' 列")
            return

        if 'domain' not in df.columns:
            print("错误: 数据中缺少 'domain' 列")
            return

        # 编码标签
        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['label'])
        self.class_names = le.classes_

        # 分离源域和目标域数据
        self.source_df = df[df['domain'] == 'Source'].copy()
        self.target_df = df[df['domain'] == 'Target'].copy()

        print(f"源域数据: {len(self.source_df)} 样本")
        print(f"目标域数据: {len(self.target_df)} 样本")
        print(f"类别分布: {dict(df['label'].value_counts())}")

        # 获取所有可用特征 - 排除非特征列
        exclude_cols = ['label', 'domain', 'label_encoded']
        self.all_features = [col for col in df.columns if col not in exclude_cols]

        # 过滤存在的特征并确保是数值类型
        self.available_features = []
        for f in self.all_features:
            if f in df.columns:
                if pd.api.types.is_numeric_dtype(df[f]):
                    self.available_features.append(f)
                else:
                    print(f"跳过非数值特征: {f}")

        # 使用源域数据进行训练
        self.X = self.source_df[self.available_features].astype(float)
        self.y = self.source_df['label_encoded']

        # 目标域数据用于域适应分析
        self.X_target = self.target_df[self.available_features].astype(float)
        self.y_target = self.target_df['label_encoded'] if 'label_encoded' in self.target_df.columns else None

        print(f"训练数据形状: {self.X.shape}")
        print(f"目标域数据形状: {self.X_target.shape}")
        print(f"类别: {self.class_names}")
        print(f"可用特征数: {len(self.available_features)}")

        # 检查是否有无效值
        if self.X.isnull().any().any():
            print("警告: 发现NaN值，将用0填充")
            self.X = self.X.fillna(0)

        if np.isinf(self.X).any().any():
            print("警告: 发现无穷值，将用0填充")
            self.X = self.X.replace([np.inf, -np.inf], 0)

    def get_optimal_features(self, method='importance'):
        """基于特征重要性或领域重叠度获取最优特征组合"""
        if method == 'robustness':
            return self.get_robust_features()
        else:
            return self.get_important_features()

    def get_robust_features(self, top_k=15):
        """基于跨域鲁棒性选择特征"""
        robust_features = [
            'spectral_centroid', 'psd_peak_freq', 'order_1_amplitude',
            'envelope_mean', 'envelope_std', 'rms', 'std', 'kurtosis',
            'vmd_mode_1_energy', 'wavelet_scale_4_energy', 'spectral_flux',
            'impact_energy_ratio', 'cyclostationarity', 'order_energy',
            'vmd_reconstruction_error'
        ]

        # 确保特征存在
        available_robust = [f for f in robust_features if f in self.available_features]
        print(f"使用鲁棒特征组合: {len(available_robust)} 个特征")
        return available_robust[:top_k]

    def get_important_features(self, top_k=20):
        """基于特征重要性选择特征"""
        important_features = [
            # 频域重要特征
            'spectral_centroid', 'spectral_spread', 'psd_peak_freq', 'order_1_amplitude',
            'order_energy', 'spectral_flux',

            # 时域重要特征
            'rms', 'std', 'kurtosis', 'impact_energy_ratio', 'cyclostationarity',

            # 时频域重要特征
            'envelope_mean', 'envelope_kurtosis', 'vmd_mode_1_energy',
            'vmd_reconstruction_error', 'wavelet_scale_4_energy',

            # 其他重要特征
            'psd_max', 'peak_to_peak', 'vmd_mode_2_energy', 'wavelet_scale_8_energy'
        ]

        # 确保特征存在
        available_important = [f for f in important_features if f in self.available_features]
        print(f"使用重要性特征组合: {len(available_important)} 个特征")
        return available_important[:top_k]

    def train_tree_models(self, use_robust_features=True):
        """训练三种树模型"""
        print("\n=== 训练树模型 ===")

        # 获取最优特征
        if use_robust_features:
            optimal_features = self.get_optimal_features(method='robustness')
        else:
            optimal_features = self.get_optimal_features(method='importance')

        X_optimal = self.X[optimal_features]

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_optimal)

        # 定义模型
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced'
            )
        }

        # 添加XGBoost模型
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
                verbosity=0
            )

        # 添加LightGBM模型
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=len(self.class_names),
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                class_weight='balanced',
                verbosity=-1
            )

        # 交叉验证评估
        cv_results = {}
        trained_models = {}

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for name, model in models.items():
            print(f"\n训练 {name}...")

            accuracy_scores = []
            f1_scores = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, self.y)):
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val_fold = self.y[train_idx], self.y[val_idx]

                model.fit(X_train_fold, y_train_fold)
                y_pred_fold = model.predict(X_val_fold)

                accuracy_scores.append(accuracy_score(y_val_fold, y_pred_fold))
                f1_scores.append(f1_score(y_val_fold, y_pred_fold, average='weighted'))

            # 训练完整模型
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
                }
            }

            print(f"  准确率: {cv_results[name]['accuracy']['mean']:.4f} ± {cv_results[name]['accuracy']['std']:.4f}")
            print(f"  F1分数: {cv_results[name]['f1']['mean']:.4f} ± {cv_results[name]['f1']['std']:.4f}")

        self.cv_results = cv_results
        self.trained_models = trained_models
        self.scaler = scaler
        self.optimal_features = optimal_features
        self.X_scaled = X_scaled

        # 对目标域数据进行转换
        if len(self.X_target) > 0:
            self.X_target_scaled = self.scaler.transform(self.X_target[optimal_features])
        else:
            self.X_target_scaled = None

        return cv_results, trained_models

    def domain_adaptation_analysis(self):
        """分析模型在目标域上的性能"""
        if self.X_target_scaled is None or self.y_target is None:
            print("目标域数据不可用，跳过域适应分析")
            return

        print("\n=== 域适应性能分析 ===")

        domain_results = {}

        for name, model in self.trained_models.items():
            # 在目标域上的预测
            y_pred_target = model.predict(self.X_target_scaled)
            accuracy_target = accuracy_score(self.y_target, y_pred_target)
            f1_target = f1_score(self.y_target, y_pred_target, average='weighted')

            # 源域性能（训练集上的性能）
            y_pred_source = model.predict(self.X_scaled)
            accuracy_source = accuracy_score(self.y, y_pred_source)
            f1_source = f1_score(self.y, y_pred_source, average='weighted')

            domain_results[name] = {
                'source_accuracy': accuracy_source,
                'source_f1': f1_source,
                'target_accuracy': accuracy_target,
                'target_f1': f1_target,
                'performance_gap': accuracy_source - accuracy_target
            }

            print(f"{name}:")
            print(f"  源域准确率: {accuracy_source:.4f}, 目标域准确率: {accuracy_target:.4f}")
            print(f"  性能差距: {accuracy_source - accuracy_target:.4f}")

        self.domain_results = domain_results
        return domain_results

    def plot_domain_comparison(self):
        """绘制源域和目标域性能对比"""
        if not hasattr(self, 'domain_results'):
            print("未进行域适应分析，跳过对比图")
            return

        model_names = list(self.domain_results.keys())
        source_accuracies = [self.domain_results[name]['source_accuracy'] for name in model_names]
        target_accuracies = [self.domain_results[name]['target_accuracy'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, source_accuracies, width, label='源域', alpha=0.8)
        bars2 = ax.bar(x + width / 2, target_accuracies, width, label='目标域', alpha=0.8)

        ax.set_xlabel('模型')
        ax.set_ylabel('准确率')
        ax.set_title('模型在源域和目标域上的性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'domain_adaptation_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_model_comparison(self):
        """绘制模型性能对比图"""
        print("\n=== 绘制模型性能对比 ===")

        model_names = list(self.cv_results.keys())
        metrics = ['accuracy', 'f1']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ['skyblue', 'lightgreen', 'lightcoral'][:len(model_names)]

        for i, metric in enumerate(metrics):
            ax = axes[i]
            means = [self.cv_results[name][metric]['mean'] for name in model_names]
            stds = [self.cv_results[name][metric]['std'] for name in model_names]

            bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'模型{metric.capitalize()}对比')
            ax.grid(True, alpha=0.3)

            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def shap_analysis(self, model_name='RandomForest', sample_size=1000):
        """SHAP可解释性分析"""
        print(f"\n=== SHAP分析 ({model_name}) ===")

        model = self.trained_models[model_name]

        # 抽样以减少计算时间
        if len(self.X_scaled) > sample_size:
            sample_indices = np.random.choice(len(self.X_scaled), sample_size, replace=False)
            X_sample = self.X_scaled[sample_indices]
        else:
            X_sample = self.X_scaled

        # 创建SHAP解释器
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            print(f"SHAP值计算完成，形状: {np.array(shap_values).shape}")
        except Exception as e:
            print(f"SHAP分析失败: {e}")
            return None, None

        # 1. 全局特征重要性
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample,
                              feature_names=self.optimal_features,
                              class_names=self.class_names,
                              show=False)
            plt.title(f'SHAP全局特征重要性 - {model_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'shap_summary_{model_name.lower()}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("全局特征重要性图生成完成")
        except Exception as e:
            print(f"生成全局特征重要性图失败: {e}")

        # 2. 特征重要性条形图
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample,
                              plot_type="bar",
                              feature_names=self.optimal_features,
                              class_names=self.class_names,
                              show=False)
            plt.title(f'SHAP平均绝对值特征重要性 - {model_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'shap_bar_{model_name.lower()}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("特征重要性条形图生成完成")
        except Exception as e:
            print(f"生成特征重要性条形图失败: {e}")

        # 3. 分析特征重要性
        importance_df = self.analyze_feature_importance(shap_values, model_name)

        return shap_values, importance_df

    def analyze_feature_importance(self, shap_values, model_name):
        """分析特征重要性 - 修复长度不匹配问题"""
        try:
            # 处理多类别SHAP值
            if isinstance(shap_values, list):
                # 多类别情况：shap_values是列表，每个元素对应一个类别的SHAP值矩阵
                shap_array = np.array(shap_values)  # (n_classes, n_samples, n_features)
                avg_abs_shap = np.mean(np.abs(shap_array), axis=(0, 1))  # 平均所有类别和样本
            else:
                # 单类别情况
                shap_array = np.array(shap_values)
                if shap_array.ndim == 3:
                    avg_abs_shap = np.abs(shap_array).mean(axis=0).mean(axis=0)
                else:
                    avg_abs_shap = np.abs(shap_array).mean(axis=0)

            print(f"SHAP值形状: {np.array(shap_values).shape}")
            print(f"特征重要性数组长度: {len(avg_abs_shap)}")
            print(f"特征列表长度: {len(self.optimal_features)}")

            # 确保长度匹配
            min_length = min(len(avg_abs_shap), len(self.optimal_features))
            if min_length == 0:
                print("错误: 特征列表或SHAP值为空")
                return pd.DataFrame()

            # 截取相同长度
            importance_values = avg_abs_shap[:min_length]
            feature_names = self.optimal_features[:min_length]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)

            print(f"\n{model_name} 特征重要性排名 (前15个):")
            for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
                feature_group = self.get_feature_group(row['feature'])
                print(f"{i:2d}. {row['feature']:25s} ({feature_group:10s}): {row['importance']:.4f}")

            # 按特征组分析重要性
            self.analyze_feature_group_importance(importance_df, model_name)

            # 保存特征重要性
            importance_df.to_csv(os.path.join(self.results_dir, f'feature_importance_{model_name.lower()}.csv'),
                                 index=False, encoding='utf-8-sig')

            return importance_df

        except Exception as e:
            print(f"特征重要性分析失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_feature_group(self, feature_name):
        """获取特征所属的组别"""
        for group, features in self.feature_groups.items():
            if feature_name in features:
                return group
        return 'other'

    def analyze_feature_group_importance(self, importance_df, model_name):
        """分析各特征组的重要性"""
        if importance_df.empty:
            print("特征重要性数据为空，跳过组分析")
            return

        importance_df['group'] = importance_df['feature'].apply(self.get_feature_group)
        group_importance = importance_df.groupby('group')['importance'].sum().sort_values(ascending=False)

        print(f"\n{model_name} 特征组重要性:")
        for group, importance in group_importance.items():
            print(f"  {group:15s}: {importance:.4f}")

        # 绘制特征组重要性图
        try:
            plt.figure(figsize=(10, 6))
            group_importance.plot(kind='bar', alpha=0.7)
            plt.title(f'特征组重要性 - {model_name}')
            plt.ylabel('总重要性')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'feature_group_importance_{model_name.lower()}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"绘制特征组重要性图失败: {e}")

    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n=== 生成综合分析报告 ===")

        report_path = os.path.join(self.results_dir, 'comprehensive_analysis_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 轴承故障诊断树模型与SHAP分析报告 ===\n\n")

            f.write("1. 实验设置\n")
            f.write(f"   数据集: {self.data_path}\n")
            f.write(f"   源域样本数: {len(self.source_df)}\n")
            f.write(f"   目标域样本数: {len(self.target_df)}\n")
            f.write(f"   特征数量: {len(self.optimal_features)}\n")
            f.write(f"   类别: {', '.join(self.class_names)}\n\n")

            f.write("2. 模型性能对比\n")
            if hasattr(self, 'cv_results'):
                sorted_models = sorted(self.cv_results.items(),
                                       key=lambda x: x[1]['accuracy']['mean'], reverse=True)

                for i, (name, results) in enumerate(sorted_models, 1):
                    f.write(f"   {i}. {name:15s}: "
                            f"准确率 {results['accuracy']['mean']:.4f}±{results['accuracy']['std']:.4f}\n")

            # 域适应分析结果
            if hasattr(self, 'domain_results'):
                f.write("\n3. 域适应性能分析\n")
                for name, results in self.domain_results.items():
                    f.write(f"   {name}: 源域{results['source_accuracy']:.4f} -> "
                            f"目标域{results['target_accuracy']:.4f} "
                            f"(差距: {results['performance_gap']:.4f})\n")

            f.write("\n4. 特征分析总结\n")
            f.write("   特征组分布:\n")
            for group, features in self.feature_groups.items():
                available_features = [f for f in features if f in self.optimal_features]
                f.write(f"   - {group}: {len(available_features)}个特征\n")

        print(f"综合分析报告已保存到: {report_path}")

    def run_complete_analysis(self, use_robust_features=True):
        """运行完整分析流程"""
        print("=== 开始完整的树模型与SHAP分析 ===")

        # 1. 训练树模型
        self.train_tree_models(use_robust_features=use_robust_features)

        # 2. 域适应分析
        self.domain_adaptation_analysis()
        self.plot_domain_comparison()

        # 3. 绘制模型对比
        self.plot_model_comparison()

        # 4. SHAP分析
        for model_name in list(self.trained_models.keys()):
            print(f"\n对 {model_name} 进行SHAP分析...")
            shap_values, importance_df = self.shap_analysis(model_name)
            if importance_df is not None and not importance_df.empty:
                print(f"{model_name} SHAP分析完成")
            else:
                print(f"{model_name} SHAP分析遇到问题")

        # 5. 生成综合报告
        self.generate_comprehensive_report()

        print(f"\n=== 分析完成 ===")
        print(f"所有结果已保存到 '{self.results_dir}' 文件夹")


def main():
    """主函数"""
    # 检查数据文件是否存在
    data_path = r'../02_特征提取/comprehensive_features.csv'
    if not os.path.exists(data_path):
        print(f"错误: 未找到数据文件 {data_path}")
        print("请先运行 comprehensive_feature_extraction.py 生成特征数据")
        return

    # 创建分析对象
    analyzer = TreeModelsSHAPAnalysis(data_path)

    # 运行完整分析（使用鲁棒特征）
    analyzer.run_complete_analysis(use_robust_features=True)

    return analyzer


if __name__ == "__main__":
    main()