#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的消融实验代码
基于实际提取的特征重新设计特征组合
"""
import os
os.environ["LOKY_BACKEND"] = "multiprocessing"  # 添加这一行
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


class ImprovedAblationStudy:
    def __init__(self, data_path=r'comprehensive_features.csv', n_splits=5, random_state=42):
        self.data_path = data_path
        self.n_splits = n_splits
        self.random_state = random_state
        self.results = {}

        # 基于实际提取的特征重新定义特征组
        self.feature_groups = {
            'time_domain': [
                'mean', 'std', 'rms', 'max', 'min', 'peak_to_peak', 'skewness', 'kurtosis',
                'energy', 'power', 'impulse_factor', 'crest_factor', 'shape_factor',
                'impact_energy_ratio', 'cyclostationarity'
            ],
            'freq_domain': [
                'spectral_centroid', 'spectral_spread', 'spectral_rolloff', 'spectral_flux',
                'psd_mean', 'psd_std', 'psd_max', 'psd_peak_freq',
                'order_1_amplitude', 'order_2_amplitude', 'order_3_amplitude', 'order_energy'
            ],
            'envelope_vmd': [
                'envelope_mean', 'envelope_std', 'envelope_kurtosis', 'envelope_skewness',
                'vmd_mode_1_energy', 'vmd_mode_1_std', 'vmd_mode_1_kurtosis',
                'vmd_mode_2_energy', 'vmd_mode_2_std', 'vmd_mode_2_kurtosis',
                'vmd_mode_3_energy', 'vmd_mode_3_std', 'vmd_mode_3_kurtosis',
                'vmd_reconstruction_error'
            ],
            'wavelet_multiscale': [
                'wavelet_scale_2_energy', 'wavelet_scale_2_std',
                'wavelet_scale_4_energy', 'wavelet_scale_4_std',
                'wavelet_scale_8_energy', 'wavelet_scale_8_std',
                'wavelet_scale_16_energy', 'wavelet_scale_16_std'
            ]
        }

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载和预处理数据"""
        print("=== 加载数据 ===")
        df = pd.read_csv(self.data_path)
        # 只使用源域数据进行消融实验
        df = df[df['domain'] == 'Source'].copy()

        # 数据清洗
        df = df.fillna(0).replace([np.inf, -np.inf], 0)

        # 编码标签
        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['label'])
        self.class_names = le.classes_

        # 获取所有可用特征
        exclude_cols = ['label', 'domain', 'label_encoded']
        self.all_features = [col for col in df.columns if col not in exclude_cols]

        # 过滤存在的特征
        self.available_features = []
        for f in self.all_features:
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
                self.available_features.append(f)

        # 准备数据
        self.X = df[self.available_features].astype(float)
        self.y = df['label_encoded']

        print(f"数据形状: {self.X.shape}")
        print(f"类别分布: {dict(pd.Series(self.y).value_counts())}")
        print(f"可用特征数: {len(self.available_features)}")

        # 验证特征组中的特征是否都存在
        self._validate_feature_groups()

    def _validate_feature_groups(self):
        """验证特征组中的特征是否在数据中存在"""
        print("\n=== 特征组验证 ===")
        for group_name, features in self.feature_groups.items():
            available_count = sum(1 for f in features if f in self.available_features)
            print(f"{group_name}: {available_count}/{len(features)} 个特征可用")

            # 更新特征组，只保留可用的特征
            self.feature_groups[group_name] = [f for f in features if f in self.available_features]

    def get_strategic_feature_combinations(self):
        """设计更合理的特征组合策略"""
        combinations = {}

        # 策略1: 单组特征（评估各组的独立贡献）
        for group_name, features in self.feature_groups.items():
            if features:  # 确保组内有特征
                combinations[f'only_{group_name}'] = features

        # 策略2: 基础组合（时域+频域）
        base_features = []
        for group in ['time_domain', 'freq_domain']:
            base_features.extend(self.feature_groups[group])
        if base_features:
            combinations['time_freq_base'] = base_features

        # 策略3: 传统方法组合（时域+频域+包络）
        traditional_features = base_features.copy()
        traditional_features.extend(self.feature_groups['envelope_vmd'][:4])  # 只取包络特征
        if traditional_features:
            combinations['traditional_methods'] = traditional_features

        # 策略4: 先进方法组合（包络+VMD+小波）
        advanced_features = []
        for group in ['envelope_vmd', 'wavelet_multiscale']:
            advanced_features.extend(self.feature_groups[group])
        if advanced_features:
            combinations['advanced_methods'] = advanced_features

        # 策略5: 完整组合（所有特征）
        all_features = []
        for features in self.feature_groups.values():
            all_features.extend(features)
        if all_features:
            combinations['all_features'] = all_features

        # 策略6: 混合组合（基础+先进）
        hybrid_features = base_features.copy()
        hybrid_features.extend(advanced_features)
        if hybrid_features:
            combinations['hybrid_approach'] = hybrid_features

        # 策略7: 基于特征重要性的组合
        combinations.update(self._get_importance_based_combinations())

        # 策略8: 基于统计检验的组合
        combinations.update(self._get_statistical_combinations())

        print(f"\n定义了 {len(combinations)} 个特征组合")
        return combinations

    def _get_importance_based_combinations(self):
        """基于随机森林特征重要性的组合"""
        combinations = {}

        # 计算特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(self.X, self.y)

        importance_df = pd.DataFrame({
            'feature': self.available_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        # 按重要性选择特征
        thresholds = [0.01, 0.005, 0.001]  # 重要性阈值
        for threshold in thresholds:
            selected_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
            if selected_features:
                combinations[f'importance_threshold_{threshold}'] = selected_features

        # 选择固定数量的重要特征
        for n in [10, 20, 30]:
            if n <= len(importance_df):
                top_features = importance_df.head(n)['feature'].tolist()
                combinations[f'top_{n}_features'] = top_features

        return combinations

    def _get_statistical_combinations(self):
        """基于统计检验的特征组合"""
        combinations = {}

        # 使用ANOVA F-value进行特征选择
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(self.X, self.y)

        scores_df = pd.DataFrame({
            'feature': self.available_features,
            'f_score': selector.scores_
        }).sort_values('f_score', ascending=False)

        # 选择F值高的特征
        for n in [15, 25]:
            if n <= len(scores_df):
                top_features = scores_df.head(n)['feature'].tolist()
                combinations[f'f_score_top_{n}'] = top_features

        return combinations

    def evaluate_combination(self, features, combination_name):
        """评估特征组合的性能"""
        print(f"评估: {combination_name} ({len(features)}个特征)")

        if not features:
            print("  警告: 无可用特征")
            return None

        # 准备数据
        X_subset = self.X[features]

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)

        # 使用随机森林进行评估
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            class_weight='balanced'
        )

        # 交叉验证评估
        cv_scores = {}
        metrics = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro'
        }

        for metric_name, scoring in metrics.items():
            try:
                scores = cross_val_score(rf, X_scaled, self.y, cv=self.n_splits,
                                         scoring=scoring, n_jobs=1)
                cv_scores[metric_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores
                }
            except Exception as e:
                print(f"  警告: {metric_name}评估失败: {e}")
                cv_scores[metric_name] = {'mean': 0, 'std': 0, 'scores': []}

        result = {
            'name': combination_name,
            'features': features,
            'n_features': len(features),
            'scores': cv_scores,
            'feature_groups': self._identify_feature_groups(features)
        }

        print(f"  准确率: {cv_scores['accuracy']['mean']:.4f} ± {cv_scores['accuracy']['std']:.4f}")

        return result

    def _identify_feature_groups(self, features):
        """识别特征所属的组别"""
        groups_included = []
        for group_name, group_features in self.feature_groups.items():
            if any(f in features for f in group_features):
                groups_included.append(group_name)
        return groups_included

    def run_study(self):
        """运行消融实验"""
        print("\n=== 开始消融实验 ===")

        combinations = self.get_strategic_feature_combinations()
        self.results = []

        for name, features in combinations.items():
            result = self.evaluate_combination(features, name)
            if result:
                self.results.append(result)

        # 按准确率排序
        self.results.sort(key=lambda x: x['scores']['accuracy']['mean'], reverse=True)

        return self.results

    def plot_comprehensive_results(self, save_path='improved_ablation_results.png'):
        """绘制综合结果图"""
        if not self.results:
            print("无结果可绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('改进的消融实验结果分析', fontsize=16, fontweight='bold')

        # 1. 性能对比图
        self._plot_performance_comparison(axes[0, 0])

        # 2. 特征数量与性能关系
        self._plot_feature_performance_relationship(axes[0, 1])

        # 3. 特征组贡献分析
        self._plot_group_contribution(axes[1, 0])

        # 4. 多指标热力图
        self._plot_metrics_heatmap(axes[1, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def _plot_performance_comparison(self, ax):
        """绘制性能对比图"""
        names = [r['name'] for r in self.results]
        accuracies = [r['scores']['accuracy']['mean'] for r in self.results]
        errors = [r['scores']['accuracy']['std'] for r in self.results]

        bars = ax.barh(range(len(names)), accuracies, xerr=errors,
                       capsize=5, alpha=0.7, color='skyblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('准确率')
        ax.set_title('各特征组合性能对比')
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{acc:.3f}', va='center', ha='left', fontsize=9)

    def _plot_feature_performance_relationship(self, ax):
        """绘制特征数量与性能关系图"""
        n_features = [r['n_features'] for r in self.results]
        accuracies = [r['scores']['accuracy']['mean'] for r in self.results]

        scatter = ax.scatter(n_features, accuracies, alpha=0.7, s=100,
                             c=accuracies, cmap='viridis')

        # 添加趋势线
        if len(n_features) > 1:
            z = np.polyfit(n_features, accuracies, 2)
            p = np.poly1d(z)
            x_line = np.linspace(min(n_features), max(n_features), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8)

        ax.set_xlabel('特征数量')
        ax.set_ylabel('准确率')
        ax.set_title('特征数量 vs 模型性能')
        ax.grid(True, alpha=0.3)

        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='准确率')

    def _plot_group_contribution(self, ax):
        """绘制特征组贡献分析"""
        group_performance = {}

        for result in self.results:
            for group in result['feature_groups']:
                if group not in group_performance:
                    group_performance[group] = []
                group_performance[group].append(result['scores']['accuracy']['mean'])

        # 计算各组的平均性能
        group_means = {group: np.mean(scores) for group, scores in group_performance.items()}

        groups = list(group_means.keys())
        means = list(group_means.values())

        bars = ax.bar(groups, means, alpha=0.7,
                      color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700'])

        ax.set_ylabel('平均准确率')
        ax.set_title('各特征组贡献分析')
        ax.grid(True, alpha=0.3)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')

    def _plot_metrics_heatmap(self, ax):
        """绘制多指标热力图"""
        metrics_data = []
        for result in self.results[:10]:  # 只显示前10个组合
            row = [
                result['scores']['accuracy']['mean'],
                result['scores']['f1_macro']['mean'],
                result['scores']['precision_macro']['mean'],
                result['scores']['recall_macro']['mean']
            ]
            metrics_data.append(row)

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data,
                                      index=[r['name'] for r in self.results[:10]],
                                      columns=['准确率', 'F1分数', '精确率', '召回率'])

            sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
            ax.set_title('Top 10组合的多指标对比')

    def generate_detailed_report(self):
        """生成详细分析报告"""
        if not self.results:
            return

        print("\n" + "=" * 60)
        print("                   消融实验详细报告")
        print("=" * 60)

        # 最佳组合
        best_result = self.results[0]
        print(f"\n🏆 最佳特征组合: {best_result['name']}")
        print(f"   准确率: {best_result['scores']['accuracy']['mean']:.4f}")
        print(f"   特征数量: {best_result['n_features']}")
        print(f"   包含的特征组: {', '.join(best_result['feature_groups'])}")

        # 各组单独性能
        print(f"\n📊 各特征组单独性能:")
        single_group_results = [r for r in self.results if r['name'].startswith('only_')]
        for result in single_group_results:
            group_name = result['name'].replace('only_', '')
            print(f"   {group_name:15s}: {result['scores']['accuracy']['mean']:.4f}")

        # 策略效果分析
        print(f"\n🎯 特征组合策略效果:")
        strategies = {
            '传统方法': [r for r in self.results if 'traditional' in r['name']],
            '先进方法': [r for r in self.results if 'advanced' in r['name']],
            '混合方法': [r for r in self.results if 'hybrid' in r['name']],
            '统计选择': [r for r in self.results if 'f_score' in r['name']]
        }

        for strategy_name, results in strategies.items():
            if results:
                best_strategy = max(results, key=lambda x: x['scores']['accuracy']['mean'])
                print(f"   {strategy_name:10s}: {best_strategy['scores']['accuracy']['mean']:.4f}")

        # 推荐方案
        print(f"\n💡 推荐方案:")
        print("   1. 如果追求最高性能: 使用最佳组合")
        print("   2. 如果注重可解释性: 使用传统方法组合")
        print("   3. 如果计算资源有限: 使用重要性选择的前20个特征")

        return best_result


def main():
    """主函数"""
    print("=== 改进的轴承故障诊断消融实验 ===")

    # 创建实验对象
    ablation = ImprovedAblationStudy()

    # 运行实验
    results = ablation.run_study()

    # 绘制结果
    ablation.plot_comprehensive_results()

    # 生成报告
    best_result = ablation.generate_detailed_report()

    print("\n=== 实验完成 ===")

    return ablation, results


if __name__ == "__main__":
    ablation, results = main()