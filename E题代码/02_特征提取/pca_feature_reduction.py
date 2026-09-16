import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class IntelligentFeatureReducer:
    def __init__(self, n_components=0.95, feature_groups=None):
        """
        智能特征降维器

        Parameters:
        n_components: 主成分数量或解释方差阈值
        feature_groups: 特征分组信息
        """
        self.n_components = n_components
        self.feature_groups = feature_groups or {}
        self.scaler = StandardScaler()
        self.pca = None
        self.explained_variance_ratio_ = None
        self.optimal_components = None

    def prepare_features(self, df, domain_aware=True):
        """准备特征数据"""
        print("=== 特征数据准备 ===")

        # 基础列排除
        exclude_cols = ['label', 'domain']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # 确保数值类型
        numeric_features = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
            else:
                print(f"跳过非数值特征: {col}")

        X = df[numeric_features].copy()
        y = df['label']
        domains = df['domain'] if 'domain' in df.columns else None

        # 数据清洗
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        print(f"特征数量: {len(numeric_features)}")
        print(f"样本数量: {len(X)}")
        print(f"类别分布: {dict(y.value_counts())}")

        if domain_aware and domains is not None:
            print(f"域分布: {dict(domains.value_counts())}")

        return X, y, domains, numeric_features

    def select_optimal_features(self, X, y, strategy='combined', n_features=30):
        """选择最优特征子集"""
        print(f"\n=== 特征选择策略: {strategy} ===")

        if strategy == 'variance':
            # 基于方差选择
            variances = X.var().sort_values(ascending=False)
            selected_features = variances.head(n_features).index.tolist()

        elif strategy == 'anova':
            # 基于ANOVA F值选择
            selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
            selector.fit(X, y)
            scores = pd.Series(selector.scores_, index=X.columns)
            selected_features = scores.nlargest(n_features).index.tolist()

        elif strategy == 'combined':
            # 组合策略：方差 + ANOVA
            variance_features = X.var().nlargest(n_features // 2).index.tolist()
            selector = SelectKBest(score_func=f_classif, k=min(n_features // 2, X.shape[1]))
            selector.fit(X, y)
            anova_features = pd.Series(selector.scores_, index=X.columns).nlargest(n_features // 2).index.tolist()
            selected_features = list(set(variance_features + anova_features))

        elif strategy == 'group_based' and self.feature_groups:
            # 基于特征组选择
            selected_features = []
            for group_name, features in self.feature_groups.items():
                available_features = [f for f in features if f in X.columns]
                # 从每组选择前几个特征
                if available_features:
                    group_variances = X[available_features].var().sort_values(ascending=False)
                    selected_features.extend(group_variances.head(3).index.tolist())
        else:
            selected_features = X.columns.tolist()

        print(f"选择特征数量: {len(selected_features)}")
        return selected_features

    def determine_optimal_components(self, X_scaled):
        """确定最优主成分数量"""
        # 计算所有主成分
        pca_full = PCA()
        pca_full.fit(X_scaled)

        explained_variance = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # 根据阈值确定成分数量
        if isinstance(self.n_components, float):
            n_components = np.argmax(cumulative_variance >= self.n_components) + 1
            n_components = max(2, min(n_components, len(cumulative_variance)))
        else:
            n_components = min(self.n_components, len(cumulative_variance))

        self.optimal_components = n_components
        print(f"确定最优主成分数量: {n_components}")
        print(f"解释方差: {cumulative_variance[n_components - 1]:.3f}")

        return n_components, explained_variance, cumulative_variance

    def fit_transform(self, X, y=None, domains=None, feature_names=None,
                      feature_selection_strategy='combined'):
        """执行智能特征降维"""

        # 特征选择
        if feature_selection_strategy != 'none':
            selected_features = self.select_optimal_features(X, y, feature_selection_strategy)
            X_selected = X[selected_features]
            feature_names = selected_features
        else:
            X_selected = X
            feature_names = feature_names or X.columns.tolist()

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X_selected)

        # 确定最优主成分数量
        n_components, explained_variance, cumulative_variance = self.determine_optimal_components(X_scaled)

        # PCA降维
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)

        self.explained_variance_ratio_ = explained_variance
        self.cumulative_variance_ratio_ = cumulative_variance
        self.feature_names = feature_names

        print(f"降维完成: {X_selected.shape[1]} -> {X_pca.shape[1]} 个特征")

        return X_pca

    def analyze_feature_contributions(self, top_n=5):
        """分析特征对主成分的贡献"""
        if self.pca is None:
            print("请先执行fit_transform方法")
            return None

        contributions = {}
        components = self.pca.components_

        for i in range(self.optimal_components):
            # 计算每个特征对当前主成分的贡献
            feature_weights = pd.Series(np.abs(components[i]), index=self.feature_names)
            top_features = feature_weights.nlargest(top_n)

            contributions[f'PC{i + 1}'] = {
                'explained_variance': self.explained_variance_ratio_[i],
                'top_features': top_features.to_dict()
            }

        return contributions

    def plot_comprehensive_analysis(self, X_pca, y, domains=None, save_path='pca_analysis_comprehensive.png'):
        """绘制全面的PCA分析图"""
        fig = plt.figure(figsize=(20, 16))
        # fig.suptitle('PCA降维分析报告', fontsize=10, fontweight='bold')

        # 创建子图布局
        gs = plt.GridSpec(3, 3, figure=fig)

        # 1. 方差解释图
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_variance_explained(ax1)

        # 2. 累积方差图
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_cumulative_variance(ax2)

        # 3. PCA散点图（按类别）
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_pca_by_class(ax3, X_pca, y)

        # 4. PCA散点图（按域）
        if domains is not None:
            ax4 = fig.add_subplot(gs[1, 0])
            self._plot_pca_by_domain(ax4, X_pca, domains)

        # 5. 特征贡献热力图
        ax5 = fig.add_subplot(gs[1, 1:])
        self._plot_feature_contribution_heatmap(ax5)

        # 6. t-SNE对比图
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_tsne_comparison(ax6, X_pca, y)

        # 7. 特征组贡献分析
        ax7 = fig.add_subplot(gs[2, 1:])
        self._plot_group_contributions(ax7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def _plot_variance_explained(self, ax):
        """绘制方差解释图"""
        explained = self.explained_variance_ratio_[:self.optimal_components]
        bars = ax.bar(range(1, len(explained) + 1), explained, alpha=0.7, color='lightblue')
        ax.set_xlabel('主成分')
        ax.set_ylabel('解释方差比')
        ax.set_title('各主成分解释方差')
        ax.grid(True, alpha=0.3)

        for i, (bar, var) in enumerate(zip(bars, explained)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{var:.3f}', ha='center', va='bottom', fontsize=8)

    def _plot_cumulative_variance(self, ax):
        """绘制累积方差图"""
        cumulative = self.cumulative_variance_ratio_[:self.optimal_components]
        ax.plot(range(1, len(cumulative) + 1), cumulative, 'ro-', linewidth=2, markersize=6)
        ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='85%阈值')
        ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%阈值')
        ax.set_xlabel('主成分数量')
        ax.set_ylabel('累积解释方差比')
        ax.set_title('累积解释方差')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_pca_by_class(self, ax, X_pca, y):
        """按类别绘制PCA散点图"""
        unique_labels = y.unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = y == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[colors[i]], label=label, alpha=0.7, s=50)

        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.2%})')
        ax.set_title('PCA: 按类别分布')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_pca_by_domain(self, ax, X_pca, domains):
        """按域绘制PCA散点图"""
        domain_colors = {'Source': 'blue', 'Target': 'red'}

        for domain, color in domain_colors.items():
            mask = domains == domain
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=color, label=domain, alpha=0.7, s=50)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA: 源域 vs 目标域')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_feature_contribution_heatmap(self, ax):
        """绘制特征贡献热力图"""
        if self.pca is None:
            return

        # 选择前几个主成分和重要特征
        n_components_show = min(6, self.optimal_components)
        n_features_show = min(10, len(self.feature_names))

        # 计算特征重要性
        importance_matrix = np.abs(self.pca.components_[:n_components_show, :])

        # 选择最重要的特征
        feature_importance = importance_matrix.sum(axis=0)
        top_feature_indices = np.argsort(feature_importance)[-n_features_show:]
        top_feature_names = [self.feature_names[i] for i in top_feature_indices]

        # 创建热力图数据
        heatmap_data = importance_matrix[:, top_feature_indices]

        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(top_feature_names)))
        ax.set_xticklabels(top_feature_names, rotation=45, ha='right')
        ax.set_yticks(range(n_components_show))
        ax.set_yticklabels([f'PC{i + 1}' for i in range(n_components_show)])
        ax.set_title('特征对主成分的贡献热力图')
        plt.colorbar(im, ax=ax)

    def _plot_tsne_comparison(self, ax, X_pca, y):
        """绘制t-SNE对比图"""
        try:
            # 使用PCA降维后的数据进行t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_pca) - 1))
            X_tsne = tsne.fit_transform(X_pca)

            unique_labels = y.unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = y == label
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                           c=[colors[i]], label=label, alpha=0.7, s=30)

            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title('t-SNE可视化对比')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f't-SNE计算失败\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('t-SNE可视化')

    def _plot_group_contributions(self, ax):
        """绘制特征组贡献分析"""
        if not self.feature_groups or self.pca is None:
            ax.text(0.5, 0.5, '无特征组信息',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('特征组贡献分析')
            return

        # 计算每组特征的总贡献
        group_contributions = {}
        components = np.abs(self.pca.components_)

        for group_name, features in self.feature_groups.items():
            group_features = [f for f in features if f in self.feature_names]
            if group_features:
                feature_indices = [self.feature_names.index(f) for f in group_features]
                group_contribution = components[:, feature_indices].sum(axis=1).mean()
                group_contributions[group_name] = group_contribution

        if group_contributions:
            groups = list(group_contributions.keys())
            contributions = list(group_contributions.values())

            bars = ax.bar(groups, contributions, alpha=0.7,
                          color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700'])
            ax.set_ylabel('平均贡献度')
            ax.set_title('各特征组对主成分的贡献')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            for bar, contrib in zip(bars, contributions):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{contrib:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, '无可用特征组数据',
                    ha='center', va='center', transform=ax.transAxes)


def intelligent_feature_reduction(input_file='comprehensive_features.csv',
                                  output_file='reduced_features_intelligent.csv',
                                  feature_groups=None,
                                  n_components=0.95,
                                  feature_selection_strategy='combined'):
    """智能特征降维主函数"""

    print("=== 智能特征降维分析 ===")

    # 读取数据
    df = pd.read_csv(input_file)
    # 只使用源域数据进行消融实验
    df = df[df['domain'] == 'Source'].copy()
    print(f"原始数据形状: {df.shape}")

    # 定义特征组（基于之前的分类）
    if feature_groups is None:
        feature_groups = {
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

    # 创建降维器
    reducer = IntelligentFeatureReducer(
        n_components=n_components,
        feature_groups=feature_groups
    )

    # 准备数据
    X, y, domains, feature_names = reducer.prepare_features(df)

    # 执行降维
    X_pca = reducer.fit_transform(
        X, y, domains, feature_names,
        feature_selection_strategy=feature_selection_strategy
    )

    # 创建结果DataFrame
    pca_columns = [f'PC{i + 1}' for i in range(X_pca.shape[1])]
    df_reduced = pd.DataFrame(X_pca, columns=pca_columns)
    df_reduced['label'] = y.values
    if domains is not None:
        df_reduced['domain'] = domains.values

    # 保存结果
    df_reduced.to_csv(output_file, index=False)
    print(f"降维后数据保存至: {output_file}")

    # 生成分析报告
    print("\n=== 降维分析报告 ===")
    print(f"原始特征数: {len(feature_names)}")
    print(f"降维后特征数: {X_pca.shape[1]}")
    print(f"累积解释方差: {reducer.cumulative_variance_ratio_[X_pca.shape[1] - 1]:.3f}")

    # 特征贡献分析
    contributions = reducer.analyze_feature_contributions(top_n=3)
    if contributions:
        print("\n各主成分的重要特征:")
        for pc, info in contributions.items():
            print(f"{pc} (方差解释: {info['explained_variance']:.3f}):")
            for feature, weight in info['top_features'].items():
                print(f"  {feature}: {weight:.3f}")

    # 绘制综合分析图
    reducer.plot_comprehensive_analysis(X_pca, y, domains)

    # 给出建议
    print("\n=== 降维建议 ===")
    n_components_used = X_pca.shape[1]
    variance_explained = reducer.cumulative_variance_ratio_[n_components_used - 1]

    if variance_explained >= 0.95:
        print("✅ 降维效果优秀：95%以上的方差被保留")
    elif variance_explained >= 0.85:
        print("⚠️  降维效果良好：85%-95%的方差被保留")
    else:
        print("❌ 降维效果一般：考虑调整参数或使用其他方法")

    if n_components_used <= 10:
        print("✅ 特征维度适中，适合机器学习模型")
    else:
        print("⚠️  特征维度较高，可考虑进一步降维")

    return df_reduced, reducer


if __name__ == "__main__":
    # 执行智能特征降维
    df_reduced, reducer = intelligent_feature_reduction(
        input_file='comprehensive_features.csv',
        output_file='reduced_features_intelligent.csv',
        n_components=0.95,  # 保留95%的方差
        feature_selection_strategy='combined'
    )