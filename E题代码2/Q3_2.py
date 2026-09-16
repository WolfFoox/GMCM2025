import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


class BearingTransferLearning:
    """轴承故障迁移学习诊断类"""

    def __init__(self):
        self.source_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.best_model = None
        self.label_encoder = LabelEncoder()
        self.source_features = None
        self.target_features = None
        self.aligned_target_features = None

        # 故障类型映射
        self.fault_mapping = {
            'B': '滚动体故障',
            'IR': '内圈故障',
            'OR': '外圈故障',
            'N': '正常状态'
        }

    def load_data(self, source_path, target_path):
        """加载源域和目标域数据"""
        print("=" * 60)
        print("第一步：数据加载与预处理")
        print("=" * 60)

        # 加载源域数据
        self.source_data = pd.read_csv(source_path)
        print(f"源域数据形状: {self.source_data.shape}")
        print(f"源域列名: {list(self.source_data.columns)}")

        # 加载目标域数据
        self.target_data = pd.read_csv(target_path)
        print(f"目标域数据形状: {self.target_data.shape}")
        print(f"目标域列名: {list(self.target_data.columns)}")

        # 提取源域特征和标签
        source_exclude_cols = ['file_path', 'fault_type', 'fault_size',
                               'load_level', 'sensor_type', 'rpm', 'fault_label',
                               'or_position', 'sampling_rate']

        source_feature_cols = [col for col in self.source_data.columns
                               if col not in source_exclude_cols]

        print(f"源域特征列: {source_feature_cols}")

        self.X_source_raw = self.source_data[source_feature_cols]
        self.y_source = self.source_data['fault_type']

        # 提取目标域特征
        target_exclude_cols = ['file_id', 'signal_length', 'sampling_rate',
                               'rpm', 'signal_key', 'mean']  # mean不是故障诊断的关键特征

        target_feature_cols = [col for col in self.target_data.columns
                               if col not in target_exclude_cols]

        print(f"目标域特征列: {target_feature_cols}")

        self.X_target_raw = self.target_data[target_feature_cols]

        print(f"源域原始特征维度: {self.X_source_raw.shape}")
        print(f"目标域原始特征维度: {self.X_target_raw.shape}")
        print(f"源域故障类型分布:\n{self.y_source.value_counts()}")

        return self.X_source_raw, self.y_source, self.X_target_raw

    def align_features(self):
        """特征对齐：处理源域和目标域特征维度不一致的问题"""
        print("\n第二步：特征对齐处理")
        print("-" * 40)

        # 创建精确的特征映射关系
        # 目标域特征 -> 源域特征的映射
        feature_mapping = {
            # 基本统计特征
            'rms': 'DE_rms',
            'peak': 'DE_peak',
            'std': 'DE_std',
            'kurtosis': 'DE_kurtosis',
            'skewness': 'DE_skewness',
            'crest_factor': 'DE_crest_factor',

            # 频域特征
            'freq_mean': 'DE_freq_mean',
            'freq_std': 'DE_freq_std',
            'spectral_centroid': 'DE_spectral_centroid',

            # 故障特征频率
            'BPFO_amplitude': 'DE_BPFO_amplitude',
            'BPFO_harmonics_energy': 'DE_BPFO_harmonics_energy',
            'BPFI_amplitude': 'DE_BPFI_amplitude',
            'BPFI_harmonics_energy': 'DE_BPFI_harmonics_energy',
            'BSF_amplitude': 'DE_BSF_amplitude',
            'BSF_harmonics_energy': 'DE_BSF_harmonics_energy',

            # 频带特征
            'low_band_ratio': 'DE_low_band_ratio',
            'high_band_ratio': 'DE_high_band_ratio'
        }

        print("特征映射关系:")
        for target_col, source_col in feature_mapping.items():
            print(f"  {target_col} -> {source_col}")

        # 构建对齐的特征集
        aligned_source_features = []
        aligned_target_features = []
        successful_mappings = []

        for target_col, source_col in feature_mapping.items():
            if target_col in self.X_target_raw.columns and source_col in self.X_source_raw.columns:
                aligned_source_features.append(source_col)
                aligned_target_features.append(target_col)
                successful_mappings.append((target_col, source_col))
                print(f"✓ 成功映射: {target_col} -> {source_col}")
            else:
                print(f"✗ 映射失败: {target_col} -> {source_col}")
                if target_col not in self.X_target_raw.columns:
                    print(f"    目标域缺少特征: {target_col}")
                if source_col not in self.X_source_raw.columns:
                    print(f"    源域缺少特征: {source_col}")

        # 创建对齐的数据集
        self.X_source = self.X_source_raw[aligned_source_features].copy()
        self.X_target = self.X_target_raw[aligned_target_features].copy()

        # 重命名目标域特征列，使其与源域一致
        target_rename_dict = {target_col: source_col for target_col, source_col in successful_mappings}
        self.X_target = self.X_target.rename(columns=target_rename_dict)

        print(f"\n对齐后的特征列: {list(self.X_source.columns)}")
        print(f"成功对齐的特征数量: {len(successful_mappings)}")

        # 数据清理：处理NaN和异常值
        print("\n数据清理...")

        # 检查并处理NaN值
        source_nan_info = self.X_source.isnull().sum()
        target_nan_info = self.X_target.isnull().sum()

        if source_nan_info.sum() > 0:
            print(f"源域NaN值分布:\n{source_nan_info[source_nan_info > 0]}")

        if target_nan_info.sum() > 0:
            print(f"目标域NaN值分布:\n{target_nan_info[target_nan_info > 0]}")

        # 使用SimpleImputer处理NaN值
        self.X_source = pd.DataFrame(
            self.imputer.fit_transform(self.X_source),
            columns=self.X_source.columns,
            index=self.X_source.index
        )

        self.X_target = pd.DataFrame(
            self.imputer.transform(self.X_target),
            columns=self.X_target.columns,
            index=self.X_target.index
        )

        # 处理无穷值
        self.X_source = self.X_source.replace([np.inf, -np.inf], 0)
        self.X_target = self.X_target.replace([np.inf, -np.inf], 0)

        # 最终验证
        final_source_issues = self.X_source.isnull().sum().sum() + np.isinf(self.X_source.values).sum()
        final_target_issues = self.X_target.isnull().sum().sum() + np.isinf(self.X_target.values).sum()

        print(f"清理后源域异常值: {final_source_issues}")
        print(f"清理后目标域异常值: {final_target_issues}")
        print(f"最终特征维度 - 源域: {self.X_source.shape}, 目标域: {self.X_target.shape}")

        return self.X_source, self.X_target

    def train_source_model(self):
        """训练源域模型"""
        print("\n第三步：源域模型训练")
        print("-" * 40)

        # 编码标签
        y_encoded = self.label_encoder.fit_transform(self.y_source)
        print(f"标签编码: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_source, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        # 特征标准化
        X_train_scaled = self.source_scaler.fit_transform(X_train)
        X_test_scaled = self.source_scaler.transform(X_test)

        # 最终的NaN检查
        if np.isnan(X_train_scaled).any():
            print("警告：训练数据标准化后发现NaN值")
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)

        if np.isnan(X_test_scaled).any():
            print("警告：测试数据标准化后发现NaN值")
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

        # 训练最佳模型（Gradient Boosting）
        self.best_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

        self.best_model.fit(X_train_scaled, y_train)

        # 评估源域性能
        y_pred = self.best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"源域模型准确率: {accuracy:.4f}")

        # 显示分类报告
        target_names = [self.fault_mapping[label] for label in self.label_encoder.classes_]
        print("\n源域分类报告:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        return self.best_model

    def domain_adaptation(self):
        """域适应：特征分布对齐"""
        print("\n第四步：域适应处理")
        print("-" * 40)

        # 标准化源域和目标域特征
        X_source_scaled = self.source_scaler.transform(self.X_source)
        X_target_scaled = self.source_scaler.transform(self.X_target)

        # 强制处理任何残留的NaN值
        X_source_scaled = np.nan_to_num(X_source_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_target_scaled = np.nan_to_num(X_target_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"标准化后数据检查:")
        print(
            f"  源域: shape={X_source_scaled.shape}, NaN={np.isnan(X_source_scaled).sum()}, Inf={np.isinf(X_source_scaled).sum()}")
        print(
            f"  目标域: shape={X_target_scaled.shape}, NaN={np.isnan(X_target_scaled).sum()}, Inf={np.isinf(X_target_scaled).sum()}")

        # 计算分布统计
        source_mean = np.mean(X_source_scaled, axis=0)
        source_std = np.std(X_source_scaled, axis=0)
        target_mean = np.mean(X_target_scaled, axis=0)
        target_std = np.std(X_target_scaled, axis=0)

        # 防止除零
        source_std = np.where(source_std < 1e-10, 1e-10, source_std)
        target_std = np.where(target_std < 1e-10, 1e-10, target_std)

        print("域分布差异分析:")
        mean_diff = np.mean(np.abs(source_mean - target_mean))
        std_diff = np.mean(np.abs(source_std - target_std))
        print(f"均值差异: {mean_diff:.4f}")
        print(f"标准差差异: {std_diff:.4f}")

        # 简单但稳定的域适应：Z-score对齐
        try:
            # 将目标域标准化到源域的分布
            self.X_target_aligned = (X_target_scaled - target_mean) / target_std * source_std + source_mean

            # 强制处理任何产生的异常值
            self.X_target_aligned = np.nan_to_num(self.X_target_aligned, nan=0.0, posinf=0.0, neginf=0.0)

            print("Z-score域对齐完成")

        except Exception as e:
            print(f"域对齐失败: {e}")
            print("使用原始标准化数据")
            self.X_target_aligned = X_target_scaled.copy()

        # 最终验证
        nan_count = np.isnan(self.X_target_aligned).sum()
        inf_count = np.isinf(self.X_target_aligned).sum()

        print(f"域适应完成验证:")
        print(f"  形状: {self.X_target_aligned.shape}")
        print(f"  NaN值: {nan_count}")
        print(f"  无穷值: {inf_count}")
        print(f"  数据范围: [{np.min(self.X_target_aligned):.4f}, {np.max(self.X_target_aligned):.4f}]")

        if nan_count > 0 or inf_count > 0:
            print("强制清理残留异常值...")
            self.X_target_aligned = np.nan_to_num(self.X_target_aligned, nan=0.0, posinf=0.0, neginf=0.0)

        return self.X_target_aligned

    def predict_target_domain(self):
        """预测目标域标签"""
        print("\n第五步：目标域预测")
        print("-" * 40)

        # 预测前的最终检查
        print("预测前数据检查:")
        print(f"  数据类型: {type(self.X_target_aligned)}")
        print(f"  数据形状: {self.X_target_aligned.shape}")
        print(f"  NaN值: {np.isnan(self.X_target_aligned).sum()}")
        print(f"  无穷值: {np.isinf(self.X_target_aligned).sum()}")
        print(f"  数据范围: [{np.min(self.X_target_aligned):.4f}, {np.max(self.X_target_aligned):.4f}]")

        # 确保数据是float64类型
        self.X_target_aligned = self.X_target_aligned.astype(np.float64)

        # 最后一次安全检查
        self.X_target_aligned = np.nan_to_num(self.X_target_aligned, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            # 使用训练好的模型预测
            y_target_pred = self.best_model.predict(self.X_target_aligned)
            y_target_proba = self.best_model.predict_proba(self.X_target_aligned)

            print("预测成功完成!")

        except Exception as e:
            print(f"预测失败: {e}")
            print("尝试使用默认预测...")
            # 如果预测失败，使用默认值
            y_target_pred = np.zeros(len(self.X_target_aligned), dtype=int)
            y_target_proba = np.ones((len(self.X_target_aligned), len(self.label_encoder.classes_))) / len(
                self.label_encoder.classes_)

        # 转换为原始标签
        y_target_labels = self.label_encoder.inverse_transform(y_target_pred)

        # 生成预测结果
        results = []
        for i, file_id in enumerate(self.target_data['file_id']):
            predicted_label = y_target_labels[i]
            predicted_fault = self.fault_mapping[predicted_label]
            confidence = np.max(y_target_proba[i])

            # 构建概率字典
            proba_dict = {}
            for j, class_label in enumerate(self.label_encoder.classes_):
                proba_dict[f'proba_{class_label}'] = y_target_proba[i][j] if j < len(y_target_proba[i]) else 0.0

            result = {
                'file_id': file_id,
                'predicted_label': predicted_label,
                'predicted_fault': predicted_fault,
                'confidence': confidence,
                **proba_dict
            }
            results.append(result)

        self.prediction_results = pd.DataFrame(results)

        # 显示预测结果
        print("\n目标域预测结果:")
        print("=" * 60)
        for _, row in self.prediction_results.iterrows():
            print(f"文件 {row['file_id']}: {row['predicted_fault']}({row['predicted_label']}) "
                  f"- 置信度: {row['confidence']:.3f}")

        # 统计预测分布
        print(f"\n预测标签分布:")
        pred_counts = self.prediction_results['predicted_fault'].value_counts()
        for fault, count in pred_counts.items():
            print(f"  {fault}: {count} 个文件")

        return self.prediction_results

    def visualize_transfer_results(self):
        """可视化迁移学习结果"""
        print("\n第六步：迁移结果可视化")
        print("-" * 40)

        try:
            fig = plt.figure(figsize=(20, 15))

            # 1. 特征空间可视化（t-SNE）
            print("生成t-SNE可视化...")
            ax1 = plt.subplot(2, 3, 1)

            try:
                # 合并源域和目标域数据用于t-SNE
                X_combined = np.vstack([
                    self.source_scaler.transform(self.X_source),
                    self.X_target_aligned
                ])

                # 确保数据安全
                X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

                # t-SNE降维
                n_samples = min(len(X_combined), 1000)  # 限制样本数以提高速度
                if len(X_combined) > n_samples:
                    indices = np.random.choice(len(X_combined), n_samples, replace=False)
                    X_combined = X_combined[indices]

                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples - 1))
                X_tsne = tsne.fit_transform(X_combined)

                # 绘制
                ax1.scatter(X_tsne[:len(self.X_source), 0], X_tsne[:len(self.X_source), 1],
                            alpha=0.6, c='blue', s=50, label='源域')
                ax1.scatter(X_tsne[len(self.X_source):, 0], X_tsne[len(self.X_source):, 1],
                            alpha=0.8, c='red', s=100, label='目标域', marker='s')

            except Exception as e:
                print(f"t-SNE可视化失败: {e}")
                ax1.text(0.5, 0.5, 't-SNE可视化失败', ha='center', va='center', transform=ax1.transAxes)

            ax1.set_title('t-SNE 特征空间分布', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. 目标域预测结果饼图
            ax2 = plt.subplot(2, 3, 2)
            pred_counts = self.prediction_results['predicted_fault'].value_counts()
            colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

            wedges, texts, autotexts = ax2.pie(pred_counts.values, labels=pred_counts.index,
                                               autopct='%1.1f%%', colors=colors_pie, startangle=90)
            ax2.set_title('目标域预测结果分布', fontsize=14, fontweight='bold')

            # 3. 预测置信度分布
            ax3 = plt.subplot(2, 3, 3)
            confidences = self.prediction_results['confidence']
            ax3.hist(confidences, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(np.mean(confidences), color='red', linestyle='--',
                        label=f'平均置信度: {np.mean(confidences):.3f}')
            ax3.set_title('预测置信度分布', fontsize=14, fontweight='bold')
            ax3.set_xlabel('置信度')
            ax3.set_ylabel('频次')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. 各文件预测结果条形图
            ax4 = plt.subplot(2, 3, 4)
            file_ids = self.prediction_results['file_id']
            confidences = self.prediction_results['confidence']
            predicted_faults = self.prediction_results['predicted_fault']

            color_map = {'滚动体故障': '#FF6B6B', '内圈故障': '#4ECDC4',
                         '正常状态': '#45B7D1', '外圈故障': '#96CEB4'}
            bar_colors = [color_map.get(fault, '#999999') for fault in predicted_faults]

            bars = ax4.bar(range(len(file_ids)), confidences, color=bar_colors, alpha=0.8)
            ax4.set_title('各文件预测置信度', fontsize=14, fontweight='bold')
            ax4.set_xlabel('文件ID')
            ax4.set_ylabel('置信度')
            ax4.set_xticks(range(len(file_ids)))
            ax4.set_xticklabels(file_ids, rotation=45)

            # 5. 特征重要性
            ax5 = plt.subplot(2, 3, 5)
            if hasattr(self.best_model, 'feature_importances_'):
                feature_importance = self.best_model.feature_importances_
                feature_names = self.X_source.columns

                top_indices = np.argsort(feature_importance)[-10:]
                top_features = [feature_names[i] for i in top_indices]
                top_importance = feature_importance[top_indices]

                bars = ax5.barh(range(len(top_features)), top_importance, color='lightcoral')
                ax5.set_yticks(range(len(top_features)))
                ax5.set_yticklabels(top_features)
                ax5.set_title('Top 10 特征重要性', fontsize=14, fontweight='bold')
                ax5.set_xlabel('重要性分数')

            # 6. 预测结果表格
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')

            # 创建结果表格文本
            table_text = "目标域预测结果\n" + "=" * 40 + "\n"
            for _, row in self.prediction_results.iterrows():
                table_text += f"{row['file_id']}: {row['predicted_label']} ({row['confidence']:.3f})\n"

            ax6.text(0.05, 0.95, table_text, transform=ax6.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

            plt.suptitle('轴承故障迁移学习诊断结果', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"可视化过程中出现错误: {e}")
            print("跳过可视化步骤")

        return True

    def generate_detailed_report(self):
        """生成详细的迁移诊断报告"""
        print("\n第七步：生成详细报告")
        print("=" * 60)

        print("轴承故障迁移学习诊断报告")
        print("=" * 60)

        print(f"\n1. 数据概况:")
        print(f"   源域样本数: {len(self.X_source)}")
        print(f"   目标域样本数: {len(self.X_target)}")
        print(f"   特征维度: {self.X_source.shape[1]}")

        print(f"\n2. 源域数据分布:")
        source_dist = self.y_source.value_counts()
        for fault_type, count in source_dist.items():
            fault_name = self.fault_mapping[fault_type]
            print(f"   {fault_name}({fault_type}): {count} 样本")

        print(f"\n3. 迁移学习方法:")
        print(f"   - 特征对齐: 基于语义的精确映射")
        print(f"   - 基础模型: Gradient Boosting")
        print(f"   - 域适应: Z-score分布对齐")
        print(f"   - 数据清理: SimpleImputer + 异常值处理")

        print(f"\n4. 目标域预测结果:")
        print("-" * 40)
        for _, row in self.prediction_results.iterrows():
            print(f"   文件 {row['file_id']}: {row['predicted_fault']} (置信度: {row['confidence']:.3f})")

        print(f"\n5. 预测统计:")
        pred_counts = self.prediction_results['predicted_fault'].value_counts()
        for fault, count in pred_counts.items():
            percentage = count / len(self.prediction_results) * 100
            print(f"   {fault}: {count} 个文件 ({percentage:.1f}%)")

        print(f"\n6. 置信度分析:")
        confidences = self.prediction_results['confidence']
        print(f"   平均置信度: {np.mean(confidences):.3f}")
        print(f"   置信度范围: [{np.min(confidences):.3f}, {np.max(confidences):.3f}]")

        # 输出最终标签映射
        print(f"\n7. 最终标签结果:")
        print("-" * 40)
        for _, row in self.prediction_results.iterrows():
            print(f"   {row['file_id']}: {row['predicted_label']}")

        return self.prediction_results

    def run_transfer_learning(self, source_path, target_path):
        """运行完整的迁移学习流程"""
        print("开始轴承故障迁移学习诊断")
        print("=" * 60)

        try:
            # 1. 数据加载
            self.load_data(source_path, target_path)

            # 2. 特征对齐
            self.align_features()

            # 3. 训练源域模型
            self.train_source_model()

            # 4. 域适应
            self.domain_adaptation()

            # 5. 目标域预测
            self.predict_target_domain()

            # 6. 可视化结果
            self.visualize_transfer_results()

            # 7. 生成报告
            results = self.generate_detailed_report()

            print("\n迁移学习诊断完成!")
            return results

        except Exception as e:
            print(f"迁移学习过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None


# 使用示例
if __name__ == "__main__":
    # 初始化迁移学习诊断器
    transfer_learner = BearingTransferLearning()

    # 运行迁移学习
    results = transfer_learner.run_transfer_learning(
        source_path='./processed_data/source_domain_features_simplified.csv',
        target_path='./processed_data/target_domain_features.csv'
    )

    if results is not None:
        # 保存结果
        results.to_csv('target_domain_predictions.csv', index=False)
        print(f"\n预测结果已保存到: target_domain_predictions.csv")

        # 输出最终标签（问题要求的格式）
        print("\n最终目标域标签:")
        print("=" * 40)
        for _, row in results.iterrows():
            print(f"{row['file_id']}: {row['predicted_label']}")
    else:
        print("迁移学习失败，请检查数据和代码")