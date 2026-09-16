import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class FeatureReducer:
    def __init__(self, n_components=12):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.feature_names = None
        self.explained_variance_ratio_ = None
        
    def fit_transform(self, X, feature_names=None):
        """对特征进行PCA降维"""
        self.feature_names = feature_names
        
        # 数据预处理
        X_clean = X.fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # PCA降维
        X_pca = self.pca.fit_transform(X_scaled)
        
        # 保存解释方差比
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        
        return X_pca
    
    def get_feature_importance(self, original_features):
        """获取原始特征对主成分的贡献度"""
        components = self.pca.components_
        feature_importance = {}
        
        for i in range(self.n_components):
            component_importance = {}
            for j, feature in enumerate(original_features):
                component_importance[feature] = abs(components[i, j])
            
            # 按重要性排序
            sorted_features = sorted(component_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            feature_importance[f'PC{i+1}'] = sorted_features[:5]  # 前5个重要特征
            
        return feature_importance
    
    def plot_pca_analysis(self, X_pca, labels, save_path='pca_analysis.png'):
        """绘制PCA分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PCA降维分析结果', fontsize=16)
        
        # 1. 主成分贡献度
        ax1 = axes[0, 0]
        explained_variance = self.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        x_pos = range(1, len(explained_variance) + 1)
        bars = ax1.bar(x_pos, explained_variance, alpha=0.7, color='skyblue')
        ax1.set_xlabel('主成分')
        ax1.set_ylabel('解释方差比')
        ax1.set_title('各主成分解释方差比')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, var in zip(bars, explained_variance):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{var:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 累积解释方差比
        ax2 = axes[0, 1]
        ax2.plot(x_pos, cumulative_variance, 'ro-', linewidth=2, markersize=6)
        ax2.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='85%阈值')
        ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%阈值')
        ax2.set_xlabel('主成分数量')
        ax2.set_ylabel('累积解释方差比')
        ax2.set_title('累积解释方差比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. PCA散点图
        ax3 = axes[1, 0]
        unique_labels = np.unique(labels)
        colors = ['green', 'blue', 'red', 'orange']
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[i % len(colors)], label=label, alpha=0.7, s=50)
        
        ax3.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
        ax3.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
        ax3.set_title('PCA前两个主成分分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 特征重要性热力图
        ax4 = axes[1, 1]
        if self.feature_names is not None:
            # 计算前几个主成分的特征载荷
            n_show = min(5, self.n_components)
            components_to_show = self.pca.components_[:n_show]
            
            im = ax4.imshow(components_to_show, cmap='RdBu_r', aspect='auto')
            ax4.set_xticks(range(len(self.feature_names)))
            ax4.set_xticklabels(self.feature_names, rotation=45, ha='right')
            ax4.set_yticks(range(n_show))
            ax4.set_yticklabels([f'PC{i+1}' for i in range(n_show)])
            ax4.set_title('主成分特征载荷')
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def reduce_features_with_pca(input_file='bearing_features.csv', 
                           output_file='reduced_features.csv',
                           n_components=12):
    """使用PCA对特征进行降维（结合消融实验结果）"""
    
    print("=== 开始PCA特征降维分析 ===")
    
    # 读取数据
    df = pd.read_csv(input_file)
    print(f"原始数据形状: {df.shape}")
    
    # 分离特征和标签
    feature_cols = [col for col in df.columns 
                   if col not in ['label', 'filename', 'rpm', 'sampling_rate']]
    X = df[feature_cols]
    y = df['label']
    
    print(f"原始特征数: {len(feature_cols)}")
    print(f"样本数: {len(X)}")
    
    # 检查消融实验结果
    try:
        ablation_df = pd.read_csv('ablation_results.csv')
        best_combo = ablation_df.iloc[0]  # 最佳组合
        print(f"\n消融实验最佳特征组合: {best_combo['combination_name']}")
        print(f"最佳特征数量: {best_combo['n_features']}")
        print(f"最佳准确率: {best_combo['accuracy_mean']:.4f}")
        
        # 如果最佳特征数量较少，建议不进行PCA降维
        if best_combo['n_features'] <= 15:
            print("\n建议: 消融实验已找到最优特征组合，PCA降维可能不必要")
            print("推荐直接使用消融实验的最优特征组合")
            
            # 仍然进行PCA分析，但主要用于可视化
            print("\n继续进行PCA分析用于特征关系可视化...")
            
    except FileNotFoundError:
        print("未找到消融实验结果，继续进行PCA分析...")
    
    # 创建特征降维器
    reducer = FeatureReducer(n_components=n_components)
    
    # 执行PCA降维
    X_pca = reducer.fit_transform(X, feature_cols)
    
    print(f"降维后特征数: {X_pca.shape[1]}")
    print(f"解释方差比: {reducer.explained_variance_ratio_}")
    print(f"累积解释方差比: {np.cumsum(reducer.explained_variance_ratio_)}")
    
    # 创建降维后的DataFrame
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    df_reduced = pd.DataFrame(X_pca, columns=pca_columns)
    df_reduced['label'] = y
    df_reduced['filename'] = df['filename']
    df_reduced['rpm'] = df['rpm']
    df_reduced['sampling_rate'] = df['sampling_rate']
    
    # 保存结果
    df_reduced.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"降维后数据已保存到: {output_file}")
    
    # 绘制分析图
    reducer.plot_pca_analysis(X_pca, y, 'pca_analysis.png')
    
    # 显示特征重要性
    feature_importance = reducer.get_feature_importance(feature_cols)
    print("\n=== 各主成分的重要特征 ===")
    for pc, features in feature_importance.items():
        print(f"\n{pc}:")
        for feature, importance in features:
            print(f"  {feature}: {importance:.3f}")
    
    # 生成建议
    print("\n=== PCA分析建议 ===")
    cumulative_var = np.cumsum(reducer.explained_variance_ratio_)
    if cumulative_var[4] >= 0.85:  # 前5个主成分解释85%方差
        print("前5个主成分已能解释85%以上的方差，建议使用前5个主成分")
    elif cumulative_var[9] >= 0.95:  # 前10个主成分解释95%方差
        print("前10个主成分已能解释95%以上的方差，建议使用前10个主成分")
    else:
        print("需要更多主成分才能解释足够的方差")
    
    return df_reduced, reducer

if __name__ == "__main__":
    # 执行PCA降维
    df_reduced, reducer = reduce_features_with_pca(
        input_file='bearing_features.csv',
        output_file='reduced_features.csv',
        n_components=12
    )
