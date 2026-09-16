#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
源域数据集分类器
基于clean_features.csv构建轴承故障分类模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SourceDomainClassifier:
    """源域数据集分类器"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.scaler = RobustScaler()
        self.best_model = None
        self.best_score = 0
        
    def load_data(self):
        """加载数据"""
        print("Loading source domain data...")
        self.df = pd.read_csv(self.data_path)
        
        # 只使用源域数据
        self.source_df = self.df[self.df['domain'] == 'Source'].copy()
        # self.source_df = self.df

        print(f"Source domain data shape: {self.source_df.shape}")
        print(f"Label distribution:")
        print(self.source_df['label'].value_counts())
        
        return self.source_df
    
    def prepare_features(self):
        """准备特征"""
        # 特征列（排除domain和label）
        feature_cols = [col for col in self.source_df.columns 
                       if col not in ['domain', 'label']]
        
        self.X = self.source_df[feature_cols]
        self.y = self.source_df['label']
        
        print(f"Features: {len(feature_cols)}")
        print(f"Feature names: {feature_cols}")
        
        return self.X, self.y
    
    def handle_class_imbalance(self):
        """处理类别不平衡"""
        print("\n=== Class Imbalance Analysis ===")
        
        # 计算类别权重
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(self.y), 
                                           y=self.y)
        self.class_weight_dict = dict(zip(np.unique(self.y), class_weights))
        
        print("Class weights:")
        for label, weight in self.class_weight_dict.items():
            print(f"  {label}: {weight:.3f}")
        
        # 检查类别不平衡情况
        min_samples = self.y.value_counts().min()
        print(f"Minimum samples per class: {min_samples}")
        
        # 如果某个类别样本太少，使用简单的过采样
        if min_samples < 10:
            print(f"Warning: Class imbalance detected (min samples: {min_samples})")
            print("Using class_weight='balanced' in models to handle imbalance")
        
        return self.X, self.y
    
    def split_data(self):
        """分割数据"""
        # 分层抽样，确保每个类别在训练集和测试集中都有代表
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nData split:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """训练多种模型"""
        print("\n=== Training Models ===")
        
        # 定义模型
        models = {
            # 1. 随机森林
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced'
            ),
            # 2. 梯度提升
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            # 3. 支持向量机
            'SVM': SVC(
                kernel='rbf',
                random_state=42,
                class_weight='balanced',
                probability=True
            ),
            # 4. 逻辑回归
            'Logistic Regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            ),
            # 增加新模型
            # 5. 多层感知机
            'MLP' : MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
            ),
            # 6. 极端随机树
            'ExtraTrees' : ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
            ),
            # 7. K近邻
            'KNN' : KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
            ),
            # 8. 朴素贝叶斯
            'NaiveBayes' : GaussianNB(),
            # 9. CatBoost分类器
            'CatBoost' : CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            loss_function='MultiClass',
            verbose=100
            )

        }
        
        # 训练和评估模型
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # 训练模型
            model.fit(self.X_train_scaled, self.y_train)
            
            # 预测
            y_pred = model.predict(self.X_test_scaled)
            
            # 计算准确率
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # 交叉验证
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # 保存模型
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_scores': cv_scores,
                'predictions': y_pred
            }
            
            # 更新最佳模型
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = name
        
        print(f"\nBest model: {self.best_model} (Accuracy: {self.best_score:.4f})")
        
        return self.models
    
    def hyperparameter_tuning(self):
        """超参数调优"""
        print(f"\n=== Hyperparameter Tuning for {self.best_model} ===")
        
        if self.best_model == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
            
        elif self.best_model == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=42)
            
        elif self.best_model == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
            model = SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True)
            
        else:  # Logistic Regression
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        
        # 网格搜索
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # 使用最佳参数重新训练
        best_model = grid_search.best_estimator_
        y_pred_tuned = best_model.predict(self.X_test_scaled)
        tuned_accuracy = accuracy_score(self.y_test, y_pred_tuned)
        
        print(f"Tuned test accuracy: {tuned_accuracy:.4f}")
        
        # 更新最佳模型
        self.models[self.best_model]['model'] = best_model
        self.models[self.best_model]['accuracy'] = tuned_accuracy
        self.models[self.best_model]['predictions'] = y_pred_tuned
        
        return best_model
    
    def evaluate_model(self, model_name=None):
        """评估模型"""
        if model_name is None:
            model_name = self.best_model
        
        model_info = self.models[model_name]
        y_pred = model_info['predictions']
        
        print(f"\n=== Model Evaluation: {model_name} ===")
        
        # 分类报告
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # 计算各种指标
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, y_pred, average=None
        )
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'Class': np.unique(self.y_test),
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        print(f"\nDetailed Metrics:")
        print(results_df)
        
        return results_df, cm
    
    def plot_confusion_matrix(self, model_name=None, save_path=None):
        """绘制混淆矩阵"""
        if model_name is None:
            model_name = self.best_model
        
        model_info = self.models[model_name]
        y_pred = model_info['predictions']
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                   xticklabels=np.unique(self.y_test),
                   yticklabels=np.unique(self.y_test))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model_name=None, top_n=15, save_path=None):
        """绘制特征重要性"""
        if model_name is None:
            model_name = self.best_model
        
        model = self.models[model_name]['model']
        
        # 检查模型是否有feature_importances_属性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.X.columns
            
            # 创建特征重要性DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # 绘制前top_n个特征
            plt.figure(figsize=(10, 8))
            top_features = feature_importance_df.head(top_n)
            sns.set_palette("crest")  # 设置全局调色板
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'Feature Importance - {model_name} (Top {top_n})')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_importance_df
        else:
            print(f"{model_name} does not have feature_importances_ attribute")
            return None
    
    def plot_model_comparison(self, save_path=None):
        """绘制模型比较图"""
        model_names = list(self.models.keys())
        accuracies = [self.models[name]['accuracy'] for name in model_names]
        cv_means = [self.models[name]['cv_scores'].mean() for name in model_names]
        cv_stds = [self.models[name]['cv_scores'].std() for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.set_palette("mako")  # 全局使用 mako 色盘
        bars1 = ax.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        bars2 = ax.bar(x + width/2, cv_means, width, yerr=cv_stds, 
                      label='CV Mean ± Std', alpha=0.8, capsize=5)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, save_path=None):
        """绘制类别分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        colors = sns.color_palette("Set2")
        # 原始数据分布
        original_counts = self.source_df['label'].value_counts()
        ax1.pie(original_counts.values, labels=original_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.set_title('Original Class Distribution')
        
        # 训练数据分布
        train_counts = self.y_train.value_counts()
        ax2.pie(train_counts.values, labels=train_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Training Set Class Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_target_domain(self, target_data=None):
        """预测目标域数据"""
        if target_data is None:
            # 使用原始数据中的目标域数据
            target_df = self.df[self.df['domain'] == 'Target'].copy()
            target_X = target_df.drop(['domain', 'label'], axis=1)
        else:
            target_X = target_data
        
        # 标准化
        target_X_scaled = self.scaler.transform(target_X)
        
        # 使用最佳模型预测
        best_model = self.models[self.best_model]['model']
        predictions = best_model.predict(target_X_scaled)
        probabilities = best_model.predict_proba(target_X_scaled)
        
        print(f"\n=== Target Domain Predictions ===")
        print(f"Predicted classes: {np.unique(predictions, return_counts=True)}")
        
        return predictions, probabilities

def main():
    """主函数"""
    # 创建分类器 - 使用新的49维特征数据
    classifier = SourceDomainClassifier('../02_特征提取/final_features.csv')
    
    # 加载数据
    classifier.load_data()
    
    # 准备特征
    classifier.prepare_features()
    
    # 处理类别不平衡
    classifier.handle_class_imbalance()
    
    # 分割数据
    classifier.split_data()
    
    # 训练模型
    classifier.train_models()
    
    # 超参数调优
    classifier.hyperparameter_tuning()
    
    # 评估最佳模型
    classifier.evaluate_model()
    
    # 绘制图表
    classifier.plot_class_distribution('../04_结果可视化/class_distribution.png')
    classifier.plot_model_comparison('../04_结果可视化/model_comparison.png')
    classifier.plot_confusion_matrix(save_path='../04_结果可视化/confusion_matrix.png')
    classifier.plot_feature_importance(save_path='../04_结果可视化/feature_importance.png')
    
    # 预测目标域
    classifier.predict_target_domain()
    
    return classifier

if __name__ == "__main__":
    classifier = main()
