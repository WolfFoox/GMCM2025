import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, \
    precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


# 强制设置中文字体 - 更可靠的方法
def setup_chinese_font():
    """设置中文字体显示"""
    # 清除字体缓存
    try:
        fm._rebuild()
    except:
        pass

    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 中文字体优先级列表
    chinese_fonts = [
        'SimHei',  # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',  # 宋体
        'KaiTi',  # 楷体
        'FangSong',  # 仿宋
        'STHeiti',  # 华文黑体
        'STSong',  # 华文宋体
        'PingFang SC',  # 苹果字体
        'Hiragino Sans GB',  # 苹果字体
        'WenQuanYi Micro Hei',  # Linux字体
        'Droid Sans Fallback'  # Android字体
    ]

    # 寻找可用的中文字体
    found_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            found_font = font
            print(f"使用中文字体: {font}")
            break

    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font, 'DejaVu Sans', 'Arial']
    else:
        print("警告: 未找到中文字体，将使用英文标签")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        return False

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100

    # 测试中文显示
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
        plt.close(fig)
        return True
    except:
        print("中文字体设置失败，将使用英文标签")
        return False


# 设置字体
USE_CHINESE = setup_chinese_font()


class BearingFaultDiagnosis:
    """轴承故障诊断模型训练与评估类"""

    def __init__(self, data_path):
        """
        初始化诊断器

        Args:
            data_path: 特征数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.models = {}
        self.results = {}
        self.use_smote = False

        # 故障类型映射 - 支持中英文
        if USE_CHINESE:
            self.fault_mapping = {
                'B': '滚动体故障',
                'IR': '内圈故障',
                'OR': '外圈故障',
                'N': '正常状态'
            }
        else:
            self.fault_mapping = {
                'B': 'Ball Fault',
                'IR': 'Inner Race Fault',
                'OR': 'Outer Race Fault',
                'N': 'Normal'
            }

    def load_and_explore_data(self):
        """加载和探索数据"""
        print("=" * 50)
        print("1. 数据加载与探索")
        print("=" * 50)

        # 加载数据
        self.data = pd.read_csv(self.data_path)
        print(f"数据形状: {self.data.shape}")

        # 基本信息
        print(f"\n故障类型分布:")
        fault_counts = self.data['fault_type'].value_counts()
        for fault, count in fault_counts.items():
            fault_name = self.fault_mapping.get(fault, fault)
            print(f"  {fault_name}({fault}): {count} 样本")

        # 检查数据不平衡程度
        class_ratio = fault_counts.max() / fault_counts.min()
        print(f"\n类别不平衡比例: {class_ratio:.2f}:1")
        if class_ratio > 3:
            print("  检测到严重类别不平衡，建议使用SMOTE")
            self.use_smote = True

        # 检查缺失值
        missing_data = self.data.isnull().sum()
        missing_features = missing_data[missing_data > 0]
        if len(missing_features) > 0:
            print(f"\n缺失值较多的特征数量: {len(missing_features)}")
        else:
            print("\n无缺失值")

        # 传感器类型分布
        print(f"\n传感器类型分布:")
        sensor_counts = self.data['sensor_type'].value_counts()
        for sensor, count in sensor_counts.items():
            print(f"  {sensor}: {count}")

        return self.data

    def visualize_data_distribution(self):
        """可视化数据分布"""
        print("\n2. 数据分布可视化")
        print("=" * 50)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 设置图表标题 - 支持中英文
        if USE_CHINESE:
            main_title = '轴承故障数据分布分析'
            titles = ['故障类型分布', '传感器类型 vs 故障类型', '转速(RPM)分布',
                      '故障尺寸分布', '载荷等级分布', 'DE峭度特征分布（按故障类型）']
            labels = {'sensor': '传感器类型', 'samples': '样本数', 'rpm': 'RPM', 'freq': '频次',
                      'fault_size': '故障尺寸', 'load': '载荷等级 (马力)', 'kurtosis': '峭度值'}
        else:
            main_title = 'Bearing Fault Data Distribution Analysis'
            titles = ['Fault Type Distribution', 'Sensor Type vs Fault Type', 'RPM Distribution',
                      'Fault Size Distribution', 'Load Level Distribution', 'DE Kurtosis Distribution by Fault Type']
            labels = {'sensor': 'Sensor Type', 'samples': 'Sample Count', 'rpm': 'RPM', 'freq': 'Frequency',
                      'fault_size': 'Fault Size', 'load': 'Load Level (HP)', 'kurtosis': 'Kurtosis Value'}

        fig.suptitle(main_title, fontsize=16, fontweight='bold')

        # 1. 故障类型分布
        fault_counts = self.data['fault_type'].value_counts()
        fault_labels = [self.fault_mapping.get(f, f) for f in fault_counts.index]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        wedges, texts, autotexts = axes[0, 0].pie(fault_counts.values, labels=fault_labels,
                                                  autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title(titles[0], fontweight='bold')

        # 2. 传感器类型vs故障类型
        sensor_fault = pd.crosstab(self.data['sensor_type'], self.data['fault_type'])
        x_pos = np.arange(len(sensor_fault.index))
        width = 0.8 / len(sensor_fault.columns)

        for i, fault_type in enumerate(sensor_fault.columns):
            axes[0, 1].bar(x_pos + i * width, sensor_fault[fault_type],
                           width=width, label=self.fault_mapping.get(fault_type, fault_type),
                           color=colors[i])

        axes[0, 1].set_title(titles[1], fontweight='bold')
        axes[0, 1].set_xlabel(labels['sensor'])
        axes[0, 1].set_ylabel(labels['samples'])
        axes[0, 1].set_xticks(x_pos + width * (len(sensor_fault.columns) - 1) / 2)
        axes[0, 1].set_xticklabels(sensor_fault.index)
        axes[0, 1].legend()

        # 3. 转速分布
        rpm_data = self.data['rpm'].dropna()
        axes[0, 2].hist(rpm_data, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 2].set_title(titles[2], fontweight='bold')
        axes[0, 2].set_xlabel(labels['rpm'])
        axes[0, 2].set_ylabel(labels['freq'])

        # 4. 故障尺寸分布
        fault_size_counts = self.data['fault_size'].value_counts().sort_index()
        axes[1, 0].bar(range(len(fault_size_counts)), fault_size_counts.values, color='lightcoral')
        axes[1, 0].set_title(titles[3], fontweight='bold')
        axes[1, 0].set_xlabel(labels['fault_size'])
        axes[1, 0].set_ylabel(labels['samples'])
        axes[1, 0].set_xticks(range(len(fault_size_counts)))
        axes[1, 0].set_xticklabels([f'0.{int(x):03d}' if pd.notna(x) else 'N/A' for x in fault_size_counts.index])

        # 5. 载荷等级分布
        load_counts = self.data['load_level'].value_counts().sort_index()
        axes[1, 1].bar(range(len(load_counts)), load_counts.values, color='lightgreen')
        axes[1, 1].set_title(titles[4], fontweight='bold')
        axes[1, 1].set_xlabel(labels['load'])
        axes[1, 1].set_ylabel(labels['samples'])
        axes[1, 1].set_xticks(range(len(load_counts)))
        axes[1, 1].set_xticklabels([f'{int(x)}' if pd.notna(x) else 'N/A' for x in load_counts.index])

        # 6. 关键特征的故障类型分布（以DE_kurtosis为例）
        for i, fault in enumerate(self.data['fault_type'].unique()):
            fault_data = self.data[self.data['fault_type'] == fault]['DE_kurtosis'].dropna()
            fault_name = self.fault_mapping.get(fault, fault)
            axes[1, 2].hist(fault_data, alpha=0.6, label=fault_name, color=colors[i], bins=15)

        axes[1, 2].set_title(titles[5], fontweight='bold')
        axes[1, 2].set_xlabel(labels['kurtosis'])
        axes[1, 2].set_ylabel(labels['freq'])
        axes[1, 2].legend()

        plt.tight_layout()
        plt.show()

    def prepare_features(self, feature_selection_method='rfe', n_features=20):
        """特征工程和预处理"""
        print("\n3. 特征工程与预处理")
        print("=" * 50)

        # 提取特征列（排除非特征列）
        feature_columns = [col for col in self.data.columns
                           if col not in ['file_path', 'fault_type', 'fault_size',
                                          'load_level', 'sensor_type', 'rpm', 'fault_label',
                                          'or_position', 'sampling_rate']]

        print(f"原始特征数量: {len(feature_columns)}")

        # 准备特征和标签
        self.X = self.data[feature_columns].fillna(0)  # 填充缺失值
        self.y = self.data['fault_type']

        # 标签编码
        label_encoder = LabelEncoder()
        self.y_encoded = label_encoder.fit_transform(self.y)

        print(f"类别标签映射:")
        for i, class_name in enumerate(label_encoder.classes_):
            fault_name = self.fault_mapping.get(class_name, class_name)
            print(f"  {i}: {fault_name}({class_name})")

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.3, random_state=42, stratify=self.y_encoded
        )

        print(f"\n数据集划分:")
        print(f"  训练集: {self.X_train.shape[0]} 样本")
        print(f"  测试集: {self.X_test.shape[0]} 样本")

        # 特征标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # SMOTE处理类别不平衡
        if self.use_smote:
            print("\n应用SMOTE处理类别不平衡...")

            # 检查每个类别的样本数量
            unique_train, counts_train = np.unique(self.y_train, return_counts=True)
            min_samples = np.min(counts_train)

            print("训练集中各类别样本数:")
            for label, count in zip(unique_train, counts_train):
                class_name = label_encoder.classes_[label]
                fault_name = self.fault_mapping.get(class_name, class_name)
                print(f"  {fault_name}: {count}")

            # 检查是否可以应用SMOTE
            if min_samples >= 6:
                # 有足够样本，使用默认SMOTE
                smote = SMOTE(random_state=42)
                self.X_train_scaled, self.y_train = smote.fit_resample(self.X_train_scaled, self.y_train)
                print(f"SMOTE成功应用，训练集大小: {self.X_train_scaled.shape[0]} 样本")
            elif min_samples >= 2:
                # 样本数较少，使用调整后的SMOTE参数
                k_neighbors = min_samples - 1
                print(f"样本数较少，调整SMOTE参数 k_neighbors={k_neighbors}")
                smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
                self.X_train_scaled, self.y_train = smote.fit_resample(self.X_train_scaled, self.y_train)
                print(f"SMOTE成功应用，训练集大小: {self.X_train_scaled.shape[0]} 样本")
            else:
                # 样本数太少，不能使用SMOTE
                print("警告: 某些类别样本数太少（<2），跳过SMOTE处理")
                print("建议: 考虑使用类别权重或其他方法处理不平衡问题")
                self.use_smote = False

            # 如果成功应用SMOTE，打印结果
            if self.use_smote:
                unique, counts = np.unique(self.y_train, return_counts=True)
                print("SMOTE后各类别样本数:")
                for label, count in zip(unique, counts):
                    class_name = label_encoder.classes_[label]
                    fault_name = self.fault_mapping.get(class_name, class_name)
                    print(f"  {fault_name}: {count}")

        # 特征选择
        if feature_selection_method == 'rfe':
            # 使用递归特征消除
            estimator = RandomForestClassifier(random_state=42)
            self.feature_selector = RFE(estimator, n_features_to_select=n_features)
            self.X_train_selected = self.feature_selector.fit_transform(self.X_train_scaled, self.y_train)
            self.X_test_selected = self.feature_selector.transform(self.X_test_scaled)

            # 获取选中的特征名称
            selected_features = np.array(feature_columns)[self.feature_selector.support_]
            print(f"\n使用RFE选择的{n_features}个关键特征:")
            for i, feature in enumerate(selected_features, 1):
                print(f"  {i:2d}. {feature}")

        elif feature_selection_method == 'univariate':
            # 使用单变量特征选择
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            self.X_train_selected = self.feature_selector.fit_transform(self.X_train_scaled, self.y_train)
            self.X_test_selected = self.feature_selector.transform(self.X_test_scaled)

            # 获取特征得分
            feature_scores = self.feature_selector.scores_
            selected_features = np.array(feature_columns)[self.feature_selector.get_support()]

            print(f"\n使用单变量选择的{n_features}个关键特征:")
            for i, (feature, score) in enumerate(zip(selected_features,
                                                     feature_scores[self.feature_selector.get_support()]), 1):
                print(f"  {i:2d}. {feature} (F-score: {score:.2f})")
        else:
            # 不进行特征选择
            self.X_train_selected = self.X_train_scaled
            self.X_test_selected = self.X_test_scaled
            print(f"\n使用全部{len(feature_columns)}个特征")

        return self.X_train_selected, self.X_test_selected, self.y_train, self.y_test

    def initialize_models(self):
        """初始化多个分类模型"""
        print("\n4. 初始化分类模型")
        print("=" * 50)

        # 计算类别权重以处理不平衡问题
        unique_labels, counts = np.unique(self.y_train, return_counts=True)
        total_samples = len(self.y_train)
        n_classes = len(unique_labels)

        # 计算平衡权重
        class_weights = {}
        for label, count in zip(unique_labels, counts):
            class_weights[label] = total_samples / (n_classes * count)

        print("类别权重设置:")
        for label, weight in class_weights.items():
            print(f"  类别{label}: {weight:.3f}")

        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10, min_samples_split=5,
                class_weight='balanced'  # 自动处理类别不平衡
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', random_state=42, probability=True,
                class_weight='balanced'  # 自动处理类别不平衡
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42, learning_rate=0.1
                # GradientBoosting不支持class_weight，但对不平衡相对鲁棒
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, multi_class='ovr',
                class_weight='balanced'  # 自动处理类别不平衡
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5, weights='distance'  # 使用距离权重
            ),
            'Multi-layer Perceptron': MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
                # 可以通过调整训练权重处理不平衡
            )
        }

        print("初始化的模型:")
        for i, model_name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {model_name}")

        if not self.use_smote:
            print("\n注意: 由于未使用SMOTE，已为支持的模型启用类别权重平衡")

        return self.models

    def train_and_evaluate_models(self):
        """训练和评估所有模型"""
        print("\n5. 模型训练与评估")
        print("=" * 50)

        self.results = {}

        for model_name, model in self.models.items():
            print(f"\n训练 {model_name}...")

            # 训练模型
            model.fit(self.X_train_selected, self.y_train)

            # 预测
            y_pred = model.predict(self.X_test_selected)
            y_pred_proba = model.predict_proba(self.X_test_selected) if hasattr(model, 'predict_proba') else None

            # 评估指标
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            precision, recall, f1_per_class, support = precision_recall_fscore_support(
                self.y_test, y_pred, average=None
            )

            # 交叉验证
            cv_scores = cross_val_score(model, self.X_train_selected, self.y_train,
                                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                        scoring='accuracy')

            # 混淆矩阵
            cm = confusion_matrix(self.y_test, y_pred)

            # 保存结果
            self.results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'f1_per_class': f1_per_class,
                'support': support,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': cm,
                'classification_report': classification_report(self.y_test, y_pred,
                                                               target_names=[self.fault_mapping[label]
                                                                             for label in ['B', 'IR', 'N', 'OR']],
                                                               output_dict=True)
            }

            print(f"  测试集准确率: {accuracy:.4f}")
            print(f"  加权F1分数: {f1:.4f}")
            print(f"  交叉验证: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        return self.results

    def visualize_model_comparison(self):
        """可视化模型对比结果"""
        print("\n6. 模型性能对比可视化")
        print("=" * 50)

        # 准备数据
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]

        # 设置标题和标签 - 支持中英文
        if USE_CHINESE:
            main_title = '轴承故障诊断模型性能综合对比'
            titles = ['测试集准确率对比', '加权F1分数对比', '5折交叉验证准确率',
                      '各类别F1分数', '模型稳定性对比', '模型性能排名']
            labels = {'accuracy': '准确率', 'f1': 'F1分数', 'model': '模型', 'stability': '稳定性分数'}
            class_names = ['滚动体故障', '内圈故障', '正常状态', '外圈故障']
            rank_headers = ['排名', '模型', '准确率', 'F1分数']
        else:
            main_title = 'Bearing Fault Diagnosis Model Performance Comparison'
            titles = ['Test Accuracy Comparison', 'Weighted F1-Score Comparison', '5-Fold Cross Validation Accuracy',
                      'Per-Class F1-Score', 'Model Stability Comparison', 'Model Performance Ranking']
            labels = {'accuracy': 'Accuracy', 'f1': 'F1-Score', 'model': 'Model', 'stability': 'Stability Score'}
            class_names = ['Ball Fault', 'Inner Race Fault', 'Normal', 'Outer Race Fault']
            rank_headers = ['Rank', 'Model', 'Accuracy', 'F1-Score']

        # 创建图形
        fig = plt.figure(figsize=(20, 12))

        # 1. 准确率对比
        ax1 = plt.subplot(2, 3, 1)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
        bars1 = ax1.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title(titles[0], fontsize=14, fontweight='bold')
        ax1.set_ylabel(labels['accuracy'], fontsize=12)
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # 添加数值标签
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. F1分数对比
        ax2 = plt.subplot(2, 3, 2)
        bars2 = ax2.bar(model_names, f1_scores, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title(titles[1], fontsize=14, fontweight='bold')
        ax2.set_ylabel(labels['f1'], fontsize=12)
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # 添加数值标签
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. 交叉验证结果对比
        ax3 = plt.subplot(2, 3, 3)
        x_pos = np.arange(len(model_names))
        ax3.bar(x_pos, cv_means, yerr=cv_stds, color=colors, alpha=0.8,
                edgecolor='black', capsize=5)
        ax3.set_title(titles[2], fontsize=14, fontweight='bold')
        ax3.set_ylabel(labels['accuracy'], fontsize=12)
        ax3.set_xlabel(labels['model'], fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.set_ylim(0, 1)

        # 4. 各类别F1分数对比（选择最佳模型）
        ax4 = plt.subplot(2, 3, 4)
        best_model_name = model_names[np.argmax(accuracies)]
        best_model_results = self.results[best_model_name]

        class_f1_scores = best_model_results['f1_per_class']

        bars4 = ax4.bar(class_names, class_f1_scores, color=colors[:4], alpha=0.8, edgecolor='black')
        ax4.set_title(f'{best_model_name} - {titles[3]}', fontsize=14, fontweight='bold')
        ax4.set_ylabel(labels['f1'], fontsize=12)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

        # 添加数值标签
        for bar, f1 in zip(bars4, class_f1_scores):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')

        # 5. 模型稳定性对比（交叉验证标准差）
        ax5 = plt.subplot(2, 3, 5)
        stability_scores = [1 - std for std in cv_stds]  # 稳定性 = 1 - 标准差
        bars5 = ax5.bar(model_names, stability_scores, color=colors, alpha=0.8, edgecolor='black')
        ax5.set_title(titles[4], fontsize=14, fontweight='bold')
        ax5.set_ylabel(labels['stability'], fontsize=12)
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')

        # 添加数值标签
        for bar, stability in zip(bars5, stability_scores):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{stability:.3f}', ha='center', va='bottom', fontweight='bold')

        # 6. 模型排名表（以文本形式显示）
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        # 创建排名数据
        performance_data = []
        for i, name in enumerate(model_names):
            performance_data.append([
                name,
                f"{accuracies[i]:.4f}",
                f"{f1_scores[i]:.4f}",
                f"{cv_means[i]:.3f}+/-{cv_stds[i]:.3f}"
            ])

        # 按准确率排序
        performance_data.sort(key=lambda x: float(x[1]), reverse=True)

        # 创建表格
        table_text = f"{titles[5]}\n" + "=" * 50 + "\n"
        table_text += f"{rank_headers[0]:<4} {rank_headers[1]:<18} {rank_headers[2]:<8} {rank_headers[3]:<8}\n"
        table_text += "-" * 50 + "\n"

        for rank, (name, acc, f1, cv) in enumerate(performance_data, 1):
            table_text += f"{rank:<4} {name:<18} {acc:<8} {f1:<8}\n"

        ax6.text(0.05, 0.95, table_text, transform=ax6.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

        plt.suptitle(main_title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # 打印性能排名表
        if USE_CHINESE:
            print("\n模型性能排名:")
            column_names = ['模型', '测试准确率', 'F1分数', '交叉验证']
        else:
            print("\nModel Performance Ranking:")
            column_names = ['Model', 'Test Accuracy', 'F1-Score', 'Cross Validation']

        performance_df = pd.DataFrame({
            column_names[0]: model_names,
            column_names[1]: [f"{acc:.4f}" for acc in accuracies],
            column_names[2]: [f"{f1:.4f}" for f1 in f1_scores],
            column_names[3]: [f"{cv:.4f}+/-{std:.4f}" for cv, std in zip(cv_means, cv_stds)]
        })

        # 按准确率排序
        performance_df = performance_df.sort_values(column_names[1], ascending=False)
        performance_df.index = range(1, len(performance_df) + 1)

        print(performance_df.to_string())

    def visualize_confusion_matrices(self):
        """可视化混淆矩阵"""
        print("\n7. 混淆矩阵分析")
        print("=" * 50)

        # 选择表现最好的3个模型
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        best_models_idx = np.argsort(accuracies)[-3:]
        best_models = [model_names[i] for i in best_models_idx]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 设置标题和标签 - 支持中英文
        if USE_CHINESE:
            main_title = 'Top 3 模型混淆矩阵对比'
            class_names = ['滚动体故障', '内圈故障', '正常状态', '外圈故障']
            pred_label = '预测类别'
            true_label = '真实类别'
            acc_label = '准确率'
        else:
            main_title = 'Top 3 Models Confusion Matrix Comparison'
            class_names = ['Ball Fault', 'Inner Race Fault', 'Normal', 'Outer Race Fault']
            pred_label = 'Predicted Class'
            true_label = 'True Class'
            acc_label = 'Accuracy'

        fig.suptitle(main_title, fontsize=16, fontweight='bold')

        for i, model_name in enumerate(best_models):
            cm = self.results[model_name]['confusion_matrix']

            # 归一化混淆矩阵
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            im = axes[i].imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[i].set_title(f'{model_name}\n({acc_label}: {self.results[model_name]["accuracy"]:.3f})',
                              fontsize=12, fontweight='bold')

            # 添加文本标注
            thresh = cm_norm.max() / 2.
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    axes[i].text(col, row, f'{cm[row, col]}\n({cm_norm[row, col]:.2f})',
                                 ha="center", va="center",
                                 color="white" if cm_norm[row, col] > thresh else "black",
                                 fontweight='bold')

            axes[i].set_xticks(range(len(class_names)))
            axes[i].set_yticks(range(len(class_names)))
            axes[i].set_xticklabels(class_names, rotation=45, ha='right')
            axes[i].set_yticklabels(class_names)
            axes[i].set_xlabel(pred_label, fontsize=10)
            axes[i].set_ylabel(true_label, fontsize=10)

        plt.tight_layout()
        plt.show()

    def detailed_classification_report(self):
        """详细分类报告"""
        print("\n8. 详细分类性能报告")
        print("=" * 50)

        # 找到最佳模型
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        best_model_name = model_names[np.argmax(accuracies)]

        # 设置标签 - 支持中英文
        if USE_CHINESE:
            best_model_label = "最佳模型"
            test_acc_label = "测试集准确率"
            weighted_f1_label = "加权F1分数"
            detail_report_label = "详细分类报告"
            class_names = ['滚动体故障', '内圈故障', '正常状态', '外圈故障']
            headers = ['故障类型', '精确率', '召回率', 'F1分数', '支持数']
            avg_labels = ['宏平均', '加权平均']
        else:
            best_model_label = "Best Model"
            test_acc_label = "Test Accuracy"
            weighted_f1_label = "Weighted F1-Score"
            detail_report_label = "Detailed Classification Report"
            class_names = ['Ball Fault', 'Inner Race Fault', 'Normal', 'Outer Race Fault']
            headers = ['Fault Type', 'Precision', 'Recall', 'F1-Score', 'Support']
            avg_labels = ['Macro Avg', 'Weighted Avg']

        print(f"{best_model_label}: {best_model_name}")
        print(f"{test_acc_label}: {self.results[best_model_name]['accuracy']:.4f}")
        print(f"{weighted_f1_label}: {self.results[best_model_name]['f1_score']:.4f}")

        print(f"\n{detail_report_label}:")
        print("-" * 70)

        # 获取分类报告
        best_results = self.results[best_model_name]
        precision = best_results['precision']
        recall = best_results['recall']
        f1_per_class = best_results['f1_per_class']
        support = best_results['support']

        print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10}")
        print("-" * 70)

        for i, class_name in enumerate(class_names):
            print(f"{class_name:<15} {precision[i]:<10.3f} {recall[i]:<10.3f} "
                  f"{f1_per_class[i]:<10.3f} {support[i]:<10.0f}")

        print("-" * 70)
        # 计算宏平均和加权平均
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1_per_class)

        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1_per_class, weights=support)

        print(f"{avg_labels[0]:<15} {macro_precision:<10.3f} {macro_recall:<10.3f} "
              f"{macro_f1:<10.3f} {np.sum(support):<10.0f}")
        print(f"{avg_labels[1]:<15} {weighted_precision:<10.3f} {weighted_recall:<10.3f} "
              f"{weighted_f1:<10.3f} {np.sum(support):<10.0f}")

        return best_model_name, self.results[best_model_name]

    def run_full_analysis(self, feature_selection_method='rfe', n_features=20):
        """运行完整的诊断分析流程"""
        print("开始轴承故障诊断分析")
        print("=" * 60)

        # 1. 数据加载和探索
        self.load_and_explore_data()

        # 2. 数据分布可视化
        self.visualize_data_distribution()

        # 3. 特征工程
        self.prepare_features(feature_selection_method, n_features)

        # 4. 初始化模型
        self.initialize_models()

        # 5. 训练和评估
        self.train_and_evaluate_models()

        # 6. 模型对比可视化
        self.visualize_model_comparison()

        # 7. 混淆矩阵
        self.visualize_confusion_matrices()

        # 8. 详细报告
        best_model_name, best_model_results = self.detailed_classification_report()

        print("\n分析完成!")
        print(f"推荐使用模型: {best_model_name}")

        return best_model_name, best_model_results


# 使用示例
if __name__ == "__main__":
    # 初始化诊断器
    diagnosis = BearingFaultDiagnosis('./processed_data/source_domain_features_simplified.csv')

    # 运行完整分析
    best_model_name, best_results = diagnosis.run_full_analysis(
        feature_selection_method='rfe',  # 'rfe', 'univariate', 'none'
        n_features=20
    )

    print(f"\n最终推荐: {best_model_name}")
    print(f"   准确率: {best_results['accuracy']:.4f}")
    print(f"   F1分数: {best_results['f1_score']:.4f}")
    print(f"   交叉验证: {best_results['cv_mean']:.4f} +/- {best_results['cv_std']:.4f}")