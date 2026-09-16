#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级迁移学习方法
实现基于深度学习的迁移学习，包括域适应和对抗训练
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedTransferLearning:
    """高级迁移学习系统"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = RobustScaler()
        self.source_model = None
        self.target_model = None
        self.domain_classifier = None
        
    def load_data(self):
        """加载数据"""
        print("Loading comprehensive features data...")
        self.df = pd.read_csv(self.data_path)
        
        # 分离源域和目标域数据
        self.source_df = self.df[self.df['domain'] == 'Source'].copy()
        self.target_df = self.df[self.df['domain'] == 'Target'].copy()
        
        print(f"Source domain: {len(self.source_df)} samples")
        print(f"Target domain: {len(self.target_df)} samples")
        
        # 准备特征和标签
        feature_cols = [col for col in self.df.columns if col not in ['domain', 'label']]
        self.feature_names = feature_cols
        
        self.X_source = self.source_df[feature_cols]
        self.y_source = self.source_df['label']
        self.X_target = self.target_df[feature_cols]
        
        print(f"Features: {len(feature_cols)}")
        print(f"Source labels: {dict(self.y_source.value_counts())}")
        
        return self.X_source, self.y_source, self.X_target
    
    def adversarial_domain_adaptation(self, X_source, X_target, y_source, alpha=0.1, max_iter=100):
        """对抗域适应"""
        print("\n=== Adversarial Domain Adaptation ===")
        
        # 标准化数据
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)
        
        # 创建域标签
        domain_labels = np.hstack([np.zeros(len(X_source_scaled)), np.ones(len(X_target_scaled))])
        X_combined = np.vstack([X_source_scaled, X_target_scaled])
        
        # 简化的对抗训练
        # 使用梯度反转层的思想
        print("Training adversarial domain classifier...")
        
        # 训练域分类器
        self.domain_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.domain_classifier.fit(X_combined, domain_labels)
        
        # 计算域分类准确率
        domain_pred = self.domain_classifier.predict(X_combined)
        domain_acc = accuracy_score(domain_labels, domain_pred)
        print(f"Domain classifier accuracy: {domain_acc:.4f}")
        
        # 如果域分类器准确率太高，说明域间差异大
        if domain_acc > 0.8:
            print("Large domain gap detected, applying domain adaptation...")
            
            # 使用域分类器的特征重要性进行特征选择
            feature_importance = self.domain_classifier.feature_importances_
            
            # 选择对域分类贡献小的特征（域不变特征）
            domain_invariant_mask = feature_importance < np.percentile(feature_importance, 30)
            
            print(f"Selected {np.sum(domain_invariant_mask)} domain-invariant features")
            
            # 使用域不变特征
            X_source_adapted = X_source_scaled[:, domain_invariant_mask]
            X_target_adapted = X_target_scaled[:, domain_invariant_mask]
            
        else:
            print("Small domain gap, using original features")
            X_source_adapted = X_source_scaled
            X_target_adapted = X_target_scaled
        
        return X_source_adapted, X_target_adapted
    
    def progressive_domain_adaptation(self, X_source, X_target, y_source):
        """渐进式域适应"""
        print("\n=== Progressive Domain Adaptation ===")
        
        # 标准化数据
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)
        
        # 第一阶段：在源域上训练基础模型
        print("Stage 1: Training base model on source domain...")
        base_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        base_model.fit(X_source_scaled, y_source)
        
        # 第二阶段：在目标域上生成伪标签
        print("Stage 2: Generating pseudo-labels on target domain...")
        target_predictions = base_model.predict(X_target_scaled)
        target_probabilities = base_model.predict_proba(X_target_scaled)
        
        # 筛选高置信度样本
        confidence_threshold = 0.8
        max_probs = np.max(target_probabilities, axis=1)
        high_conf_mask = max_probs >= confidence_threshold
        
        print(f"High confidence samples: {np.sum(high_conf_mask)}")
        
        if np.sum(high_conf_mask) > 0:
            # 第三阶段：使用伪标签进行微调
            print("Stage 3: Fine-tuning with pseudo-labels...")
            
            X_pseudo = X_target_scaled[high_conf_mask]
            y_pseudo = target_predictions[high_conf_mask]
            
            # 结合源域和伪标签数据
            X_combined = np.vstack([X_source_scaled, X_pseudo])
            y_combined = np.hstack([y_source, y_pseudo])
            
            # 训练最终模型
            self.target_model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
            self.target_model.fit(X_combined, y_combined)
            
            print("Progressive adaptation completed")
        else:
            print("No high confidence samples, using base model")
            self.target_model = base_model
        
        return target_predictions, target_probabilities
    
    def multi_task_learning(self, X_source, X_target, y_source):
        """多任务学习"""
        print("\n=== Multi-task Learning ===")
        
        # 标准化数据
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)
        
        # 创建多任务数据
        # 任务1：源域分类
        # 任务2：域分类
        # 任务3：目标域分类（使用伪标签）
        
        # 训练源域分类器
        source_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        source_classifier.fit(X_source_scaled, y_source)
        
        # 生成目标域伪标签
        target_predictions = source_classifier.predict(X_target_scaled)
        target_probabilities = source_classifier.predict_proba(X_target_scaled)
        
        # 多任务学习：结合源域分类和域适应
        X_combined = np.vstack([X_source_scaled, X_target_scaled])
        y_combined = np.hstack([y_source, target_predictions])
        
        # 创建域标签
        domain_labels = np.hstack([np.zeros(len(X_source_scaled)), np.ones(len(X_target_scaled))])
        
        # 训练多任务模型
        self.target_model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        self.target_model.fit(X_combined, y_combined)
        
        print("Multi-task learning completed")
        
        return target_predictions, target_probabilities
    
    def evaluate_transfer_methods(self):
        """评估不同迁移方法"""
        print("\n=== Evaluating Transfer Methods ===")
        
        # 准备数据
        X_source, y_source, X_target = self.load_data()
        
        methods = {
            'Adversarial': self.adversarial_domain_adaptation,
            'Progressive': self.progressive_domain_adaptation,
            'Multi-task': self.multi_task_learning
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            print(f"\n--- {method_name} Method ---")
            
            try:
                if method_name == 'Adversarial':
                    X_source_adapted, X_target_adapted = method_func(X_source, X_target, y_source)
                    # 在适应后的特征上训练模型
                    self.target_model = RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight='balanced'
                    )
                    self.target_model.fit(X_source_adapted, y_source)
                    predictions = self.target_model.predict(X_target_adapted)
                    probabilities = self.target_model.predict_proba(X_target_adapted)
                else:
                    predictions, probabilities = method_func(X_source, X_target, y_source)
                
                # 评估结果
                unique, counts = np.unique(predictions, return_counts=True)
                confidence = np.mean(np.max(probabilities, axis=1))
                
                results[method_name] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'distribution': dict(zip(unique, counts)),
                    'confidence': confidence
                }
                
                print(f"Predictions: {dict(zip(unique, counts))}")
                print(f"Mean confidence: {confidence:.3f}")
                
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                results[method_name] = None
        
        return results
    
    def plot_transfer_comparison(self, results, save_path=None):
        """绘制迁移方法对比"""
        print("\n=== Generating Transfer Method Comparison ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测分布对比
        ax1 = axes[0, 0]
        method_names = list(results.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (method, result) in enumerate(results.items()):
            if result is not None:
                distribution = result['distribution']
                labels = list(distribution.keys())
                values = list(distribution.values())
                ax1.bar([x + i*0.25 for x in range(len(labels))], values, 
                       width=0.25, label=method, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Predicted Classes')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Prediction Distribution Comparison')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 置信度对比
        ax2 = axes[0, 1]
        confidences = [result['confidence'] for result in results.values() if result is not None]
        method_names_valid = [name for name, result in results.items() if result is not None]
        
        bars = ax2.bar(method_names_valid, confidences, color=colors[:len(confidences)], alpha=0.8)
        ax2.set_ylabel('Mean Confidence')
        ax2.set_title('Confidence Comparison')
        ax2.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, conf in zip(bars, confidences):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom')
        
        ax2.grid(True, alpha=0.3)
        
        # 3. 特征空间可视化（使用最佳方法）
        ax3 = axes[1, 0]
        best_method = max(results.items(), key=lambda x: x[1]['confidence'] if x[1] else 0)
        
        if best_method[1] is not None:
            # 使用PCA降维
            X_combined = np.vstack([self.scaler.transform(self.X_source), 
                                  self.scaler.transform(self.X_target)])
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_combined)
            
            # 源域数据
            X_source_pca = X_pca[:len(self.X_source)]
            X_target_pca = X_pca[len(self.X_source):]
            
            # 绘制源域数据
            for i, label in enumerate(np.unique(self.y_source)):
                mask = self.y_source == label
                ax3.scatter(X_source_pca[mask, 0], X_source_pca[mask, 1], 
                           label=f'Source {label}', alpha=0.6, s=50)
            
            # 绘制目标域数据
            target_predictions = best_method[1]['predictions']
            for i, label in enumerate(np.unique(target_predictions)):
                mask = target_predictions == label
                ax3.scatter(X_target_pca[mask, 0], X_target_pca[mask, 1], 
                           marker='^', label=f'Target {label}', alpha=0.8, s=80)
            
            ax3.set_xlabel('First Principal Component')
            ax3.set_ylabel('Second Principal Component')
            ax3.set_title(f'Feature Space - {best_method[0]} Method')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 方法性能总结
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 创建性能总结表
        summary_data = []
        for method, result in results.items():
            if result is not None:
                summary_data.append([
                    method,
                    f"{result['confidence']:.3f}",
                    f"{len(result['predictions'])}",
                    f"{len(result['distribution'])}"
                ])
        
        table = ax4.table(cellText=summary_data,
                         colLabels=['Method', 'Confidence', 'Samples', 'Classes'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Transfer Method Performance Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_final_results(self, results, save_path='../06_迁移结果/'):
        """保存最终结果"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 选择最佳方法
        best_method = max(results.items(), key=lambda x: x[1]['confidence'] if x[1] else 0)
        
        if best_method[1] is not None:
            # 创建最终结果DataFrame
            final_results = pd.DataFrame({
                'sample_id': range(len(best_method[1]['predictions'])),
                'predicted_label': best_method[1]['predictions'],
                'confidence': np.max(best_method[1]['probabilities'], axis=1),
                'method': best_method[0]
            })
            
            # 添加各类别的概率
            class_names = self.target_model.classes_
            for i, class_name in enumerate(class_names):
                final_results[f'prob_{class_name}'] = best_method[1]['probabilities'][:, i]
            
            # 保存结果
            results_path = os.path.join(save_path, 'final_transfer_results.csv')
            final_results.to_csv(results_path, index=False)
            
            print(f"\nFinal results saved to: {results_path}")
            print(f"Best method: {best_method[0]}")
            print(f"Mean confidence: {best_method[1]['confidence']:.3f}")
            
            return final_results
        else:
            print("No valid results to save")
            return None

def main():
    """主函数"""
    # 创建高级迁移学习系统
    transfer_system = AdvancedTransferLearning('../02_特征提取/comprehensive_features.csv')
    
    # 评估不同迁移方法
    results = transfer_system.evaluate_transfer_methods()
    
    # 绘制对比图
    transfer_system.plot_transfer_comparison(results, '../04_结果可视化/advanced_transfer_comparison.png')
    
    # 保存最终结果
    final_results = transfer_system.save_final_results(results)
    
    return transfer_system, results, final_results

if __name__ == "__main__":
    transfer_system, results, final_results = main()
