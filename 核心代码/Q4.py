class InterpretabilityAnalyzer:
    """可解释性分析器核心版本"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = RobustScaler()
        self.source_model = None
        self.target_model = None

    def load_data(self):
        """加载数据"""
        print("Loading data for interpretability analysis...")
        self.df = pd.read_csv(self.data_path)

        # 分离源域和目标域数据
        self.source_df = self.df[self.df['domain'] == 'Source'].copy()
        self.target_df = self.df[self.df['domain'] == 'Target'].copy()

        # 准备特征和标签
        feature_cols = [col for col in self.df.columns if col not in ['domain', 'label']]
        self.feature_names = feature_cols

        self.X_source = self.source_df[feature_cols]
        self.y_source = self.source_df['label']
        self.X_target = self.target_df[feature_cols]

        print(f"Source domain: {len(self.X_source)} samples")
        print(f"Target domain: {len(self.X_target)} samples")
        print(f"Features: {len(feature_cols)}")

        return self.X_source, self.y_source, self.X_target

    def train_models(self):
        """训练源域和目标域模型"""
        print("Training source and target domain models...")

        # 训练源域模型
        X_source_scaled = self.scaler.fit_transform(self.X_source)
        self.source_model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        self.source_model.fit(X_source_scaled, self.y_source)

        # 生成目标域伪标签
        X_target_scaled = self.scaler.transform(self.X_target)
        target_predictions = self.source_model.predict(X_target_scaled)
        target_probabilities = self.source_model.predict_proba(X_target_scaled)

        # 筛选高置信度样本
        confidence_threshold = 0.8
        max_probs = np.max(target_probabilities, axis=1)
        high_conf_mask = max_probs >= confidence_threshold

        if np.sum(high_conf_mask) > 0:
            # 训练目标域模型
            X_pseudo = X_target_scaled[high_conf_mask]
            y_pseudo = target_predictions[high_conf_mask]

            X_combined = np.vstack([X_source_scaled, X_pseudo])
            y_combined = np.hstack([self.y_source, y_pseudo])

            self.target_model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
            self.target_model.fit(X_combined, y_combined)
        else:
            self.target_model = self.source_model

        return target_predictions, target_probabilities

    def plot_feature_importance_analysis(self, save_path):
        """特征重要性分析"""
        print("Generating feature importance analysis...")
        feature_importance = self.source_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 条形图
        ax1 = axes[0, 0]
        top_features = importance_df.head(20)
        bars = ax1.barh(range(len(top_features)), top_features['importance'],
                        color=sns.color_palette("Set2", len(top_features)))
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=10)
        ax1.set_xlabel('Feature Importance')
        ax1.set_title('Top 15 Feature Importance (Source Domain)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 累积图
        ax2 = axes[0, 1]
        cumulative_importance = np.cumsum(importance_df['importance'])
        ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance,
                 marker='o', linewidth=2, markersize=4, color=sns.color_palette("Set2")[1])
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance')
        ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 分布直方图
        ax3 = axes[1, 0]
        ax3.hist(feature_importance, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(feature_importance), color='red', linestyle='--',
                    label=f'Mean: {np.mean(feature_importance):.3f}')
        ax3.set_xlabel('Feature Importance')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Feature Importance Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 热力图
        ax4 = axes[1, 1]
        top_15_features = importance_df.head(15)
        importance_matrix = top_15_features['importance'].values.reshape(1, -1)
        im = ax4.imshow(importance_matrix, cmap='viridis', aspect='auto')
        ax4.set_xticks(range(len(top_15_features)))
        ax4.set_xticklabels(top_15_features['feature'], rotation=45, ha='right', fontsize=10)
        ax4.set_title('Top 15 Features Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax4, shrink=0.8)

        plt.tight_layout()
        plt.savefig(f'{save_path}pre_hoc_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        return importance_df

    def plot_transfer_process_visualization(self, save_path):
        """迁移过程可视化"""
        print("Generating transfer process visualization...")

        # 准备数据
        X_source_scaled = self.scaler.fit_transform(self.X_source)
        X_target_scaled = self.scaler.transform(self.X_target)

        # 使用t-SNE进行降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_source_tsne = tsne.fit_transform(X_source_scaled)
        X_target_tsne = tsne.fit_transform(X_target_scaled)

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 1. 源域数据分布
        ax1 = axes[0, 0]
        for i, label in enumerate(np.unique(self.y_source)):
            mask = self.y_source == label
            ax1.scatter(X_source_tsne[mask, 0], X_source_tsne[mask, 1],
                        label=f'Class {label}', alpha=0.8, s=60)
        ax1.set_title('Source Domain Distribution (t-SNE)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 目标域数据分布
        ax2 = axes[0, 1]
        target_predictions = self.target_model.predict(X_target_scaled)
        for i, label in enumerate(np.unique(target_predictions)):
            mask = target_predictions == label
            ax2.scatter(X_target_tsne[mask, 0], X_target_tsne[mask, 1],
                        label=f'Class {label}', alpha=0.8, s=80, marker='^')
        ax2.set_title('Target Domain Distribution (t-SNE)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 域间分布对比
        ax3 = axes[1, 0]
        for i, label in enumerate(np.unique(self.y_source)):
            mask = self.y_source == label
            ax3.scatter(X_source_tsne[mask, 0], X_source_tsne[mask, 1],
                        label=f'Source Class {label}', alpha=0.6, s=50)
        for i, label in enumerate(np.unique(target_predictions)):
            mask = target_predictions == label
            ax3.scatter(X_target_tsne[mask, 0], X_target_tsne[mask, 1],
                        label=f'Target Class {label}', alpha=0.9, s=70, marker='^')
        ax3.set_title('Source vs Target Domain Comparison', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 域间距离分析
        ax4 = axes[1, 1]
        from scipy.spatial.distance import cdist
        distances = []
        for i in range(len(X_target_tsne)):
            target_point = X_target_tsne[i].reshape(1, -1)
            dists = cdist(target_point, X_source_tsne)
            min_dist = np.min(dists)
            distances.append(min_dist)

        ax4.hist(distances, bins=20, alpha=0.8, edgecolor='black')
        mean_dist = np.mean(distances)
        ax4.axvline(mean_dist, color='red', linestyle='--', label=f'Mean: {mean_dist:.2f}')
        ax4.set_xlabel('Distance to Nearest Source Sample')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Domain Distance Distribution', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}during_hoc_transfer_process.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_decision_process_analysis(self, save_path):
        """决策过程分析可视化"""
        print("Generating decision process analysis...")

        # 准备数据
        X_source_scaled = self.scaler.fit_transform(self.X_source)
        X_target_scaled = self.scaler.transform(self.X_target)

        # 获取预测结果
        target_predictions = self.target_model.predict(X_target_scaled)
        target_probabilities = self.target_model.predict_proba(X_target_scaled)

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 1. 决策边界可视化
        ax1 = axes[0, 0]
        pca = PCA(n_components=2)
        X_combined = np.vstack([X_source_scaled, X_target_scaled])
        X_combined_pca = pca.fit_transform(X_combined)

        X_source_pca = X_combined_pca[:len(X_source_scaled)]
        X_target_pca = X_combined_pca[len(X_source_scaled):]

        for i, label in enumerate(np.unique(self.y_source)):
            mask = self.y_source == label
            ax1.scatter(X_source_pca[mask, 0], X_source_pca[mask, 1],
                        label=f'Source {label}', alpha=0.6, s=50)

        for i, label in enumerate(np.unique(target_predictions)):
            mask = target_predictions == label
            ax1.scatter(X_target_pca[mask, 0], X_target_pca[mask, 1],
                        label=f'Target {label}', alpha=0.8, s=80, marker='^')

        ax1.set_title('Decision Boundary Visualization', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 预测置信度分析
        ax2 = axes[0, 1]
        max_probs = np.max(target_probabilities, axis=1)
        ax2.hist(max_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(max_probs), color='red', linestyle='--',
                    label=f'Mean: {np.mean(max_probs):.3f}')
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 决策路径分析
        ax3 = axes[1, 0]
        class_names = self.target_model.classes_
        decision_paths = []

        for i, class_name in enumerate(class_names):
            class_mask = target_predictions == class_name
            if np.sum(class_mask) > 0:
                class_probs = target_probabilities[class_mask, i]
                decision_paths.append((class_name, np.mean(class_probs), np.std(class_probs)))

        if decision_paths:
            class_names_plot = [path[0] for path in decision_paths]
            mean_probs = [path[1] for path in decision_paths]
            std_probs = [path[2] for path in decision_paths]

            bars = ax3.bar(class_names_plot, mean_probs, yerr=std_probs,
                           alpha=0.7, capsize=5)
            ax3.set_ylabel('Average Probability')
            ax3.set_title('Decision Path Analysis by Class', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)

        # 4. 决策不确定性分析
        ax4 = axes[1, 1]
        entropy = -np.sum(target_probabilities * np.log(target_probabilities + 1e-8), axis=1)
        ax4.hist(entropy, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.axvline(np.mean(entropy), color='red', linestyle='--',
                    label=f'Mean: {np.mean(entropy):.3f}')
        ax4.set_xlabel('Decision Entropy')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Decision Uncertainty Analysis', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}post_hoc_decision_process.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_complete_analysis(self, save_path='./interpretability_results/'):
        """运行完整的可解释性分析"""
        import os
        os.makedirs(save_path, exist_ok=True)

        print("=== Starting Complete Interpretability Analysis ===")

        # 加载数据
        self.load_data()

        # 训练模型
        self.train_models()

        # 特征重要性分析
        self.plot_feature_importance_analysis(save_path)

        # 迁移过程可视化
        self.plot_transfer_process_visualization(save_path)

        # 决策过程分析
        self.plot_decision_process_analysis(save_path)

        print("=== Analysis Complete ===")
        print(f"Results saved to: {save_path}")


def main():
    """主函数"""
    # 创建分析器实例
    analyzer = InterpretabilityAnalyzer('../02_特征提取/final_features.csv')

    # 运行完整分析
    analyzer.run_complete_analysis()

    return analyzer

