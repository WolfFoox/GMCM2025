#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迁移学习诊断系统
实现三层迁移策略：特征空间对齐、模型参数迁移、伪标签自训练
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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TransferLearningDiagnosis:
    """迁移学习诊断系统"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = RobustScaler()
        self.source_model = None
        self.target_model = None
        self.alignment_method = None
        
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
    
    def mmd_alignment(self, X_source, X_target, gamma=1.0):
        """最大均值差异对齐"""
        print("Applying MMD (Maximum Mean Discrepancy) alignment...")
        
        # 计算MMD核矩阵
        def rbf_kernel(X, Y, gamma):
            X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
            dist = X_norm + Y_norm - 2 * np.dot(X, Y.T)
            return np.exp(-gamma * dist)
        
        # 计算MMD距离
        K_ss = rbf_kernel(X_source, X_source, gamma)
        K_tt = rbf_kernel(X_target, X_target, gamma)
        K_st = rbf_kernel(X_source, X_target, gamma)
        
        mmd = np.mean(K_ss) + np.mean(K_tt) - 2 * np.mean(K_st)
        print(f"MMD distance before alignment: {mmd:.6f}")
        
        # 简单的分布对齐：将目标域数据向源域分布对齐
        X_source_mean = np.mean(X_source, axis=0)
        X_source_std = np.std(X_source, axis=0)
        X_target_mean = np.mean(X_target, axis=0)
        X_target_std = np.std(X_target, axis=0)
        
        # 避免除零
        X_target_std[X_target_std == 0] = 1
        X_source_std[X_source_std == 0] = 1
        
        # 标准化对齐
        X_target_aligned = (X_target - X_target_mean) / X_target_std * X_source_std + X_source_mean
        
        # 计算对齐后的MMD距离
        K_ss_aligned = rbf_kernel(X_source, X_source, gamma)
        K_tt_aligned = rbf_kernel(X_target_aligned, X_target_aligned, gamma)
        K_st_aligned = rbf_kernel(X_source, X_target_aligned, gamma)
        
        mmd_aligned = np.mean(K_ss_aligned) + np.mean(K_tt_aligned) - 2 * np.mean(K_st_aligned)
        print(f"MMD distance after alignment: {mmd_aligned:.6f}")
        print(f"MMD reduction: {((mmd - mmd_aligned) / mmd * 100):.2f}%")
        
        return X_target_aligned, mmd, mmd_aligned
    
    # def coral_alignment(self, X_source, X_target,
    #                     eps=1e-1,
    #                     shrinkage='lw',        # 'lw' | 'oas' | float in (0,1) | None
    #                     pca_components=None,   # int or None
    #                     blend=1.0              # 0..1, 1 means full alignment
    #                     ):
    #     """相关对齐（CORAL）增强版
    #     参数：
    #     - eps: 协方差对角正则，增大更稳但可能欠对齐
    #     - shrinkage: 协方差收缩（'lw'、'oas'、(0,1) 实数或 None）
    #     - pca_components: 先 PCA 降维后再做 CORAL
    #     - blend: 部分对齐强度，1=完全对齐，0=不对齐
    #     """
    #     print("Applying CORAL (Correlation Alignment) with params:",
    #           f"eps={eps}, shrinkage={shrinkage}, pca_components={pca_components}, blend={blend}")
    #
    #     import numpy as np
    #     from sklearn.decomposition import PCA
    #     from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
    #
    #     Xs = np.asarray(X_source, dtype=np.float64)
    #     Xt = np.asarray(X_target, dtype=np.float64)
    #
    #     # 可选：PCA降维以稳定协方差估计
    #     pca = None
    #     if pca_components is not None and isinstance(pca_components, int) and pca_components > 0 and pca_components < Xs.shape[1]:
    #         pca = PCA(n_components=pca_components, svd_solver='auto', random_state=42)
    #         Xs = pca.fit_transform(Xs)
    #         Xt = pca.transform(Xt)
    #
    #     # 协方差估计（支持收缩）
    #     def estimate_cov(X):
    #         if shrinkage == 'lw':
    #             est = LedoitWolf().fit(X)
    #             return est.covariance_
    #         elif shrinkage == 'oas':
    #             est = OAS().fit(X)
    #             return est.covariance_
    #         elif isinstance(shrinkage, (float, int)) and 0 < float(shrinkage) < 1:
    #             est = ShrunkCovariance(shrinkage=float(shrinkage)).fit(X)
    #             return est.covariance_
    #         else:
    #             return np.cov(X.T)
    #
    #     Cs = estimate_cov(Xs)
    #     Ct = estimate_cov(Xt)
    #
    #     # 对角正则以提升数值稳定性
    #     d = Xs.shape[1]
    #     Cs = Cs + eps * np.eye(d)
    #     Ct = Ct + eps * np.eye(d)
    #
    #     # 对齐前的 CORAL 距离
    #     coral_distance = np.linalg.norm(Cs - Ct, 'fro')**2 / (4.0 * d * d)
    #     print(f"CORAL distance before alignment: {coral_distance:.6f}")
    #
    #     # 对称矩阵幂（特征分解实现）
    #     def sym_matrix_power(mat, power):
    #         eigvals, eigvecs = np.linalg.eigh(mat)
    #         eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
    #         return (eigvecs * (eigvals**power)) @ eigvecs.T
    #
    #     # A = Cs^{-1/2} * Ct^{1/2}
    #     Cs_inv_sqrt = sym_matrix_power(Cs, -0.5)
    #     Ct_sqrt     = sym_matrix_power(Ct,  0.5)
    #     A = Cs_inv_sqrt @ Ct_sqrt
    #
    #     # 应用对齐；可选部分对齐（blend）
    #     Xt_aligned_full = Xt @ A
    #     Xt_aligned = (1.0 - blend) * Xt + blend * Xt_aligned_full
    #
    #     # 对齐后的距离（在当前空间中评估）
    #     Ct_aligned = estimate_cov(Xt_aligned) + eps * np.eye(d)
    #     coral_distance_aligned = np.linalg.norm(Cs - Ct_aligned, 'fro')**2 / (4.0 * d * d)
    #     print(f"CORAL distance after alignment:  {coral_distance_aligned:.6f}")
    #     if coral_distance > 1e-12:
    #         reduction = (coral_distance - coral_distance_aligned) / coral_distance * 100.0
    #         print(f"CORAL reduction: {reduction:.2f}%")
    #
    #     # 若做了PCA，回到原空间
    #     if pca is not None:
    #         Xt_aligned = pca.inverse_transform(Xt_aligned)
    #
    #     return Xt_aligned, coral_distance, coral_distance_aligned

    def coral_alignment(self, X_source, X_target,
                        eps=1e-5,
                        shrinkage='lw',
                        pca_components='auto',
                        blend=0.5,  # 降低默认混合系数
                        n_iter=1,
                        power_compensation=True,
                        adaptive_eps=True,
                        metric='frobenius',
                        early_stopping=False,  # 添加早停机制
                        convergence_threshold=1e-4  # 收敛阈值
                        ):
        """修复版CORAL对齐算法"""

        print("Applying Fixed CORAL with params:",
              f"eps={eps}, shrinkage={shrinkage}, blend={blend}, n_iter={n_iter}")

        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance

        Xs = np.asarray(X_source, dtype=np.float64)
        Xt = np.asarray(X_target, dtype=np.float64)

        # 数据预处理：中心化
        Xs_mean = np.mean(Xs, axis=0)
        Xt_mean = np.mean(Xt, axis=0)
        Xs_centered = Xs - Xs_mean
        Xt_centered = Xt - Xt_mean

        # 记录原始特征能量
        original_source_energy = np.mean(np.linalg.norm(Xs_centered, axis=1) ** 2)
        original_target_energy = np.mean(np.linalg.norm(Xt_centered, axis=1) ** 2)

        # PCA降维
        pca = None
        if pca_components is not None:
            if pca_components == 'auto':
                # 基于特征值自动选择
                pca_temp = PCA().fit(Xs_centered)
                explained_variance = np.cumsum(pca_temp.explained_variance_ratio_)
                pca_components = np.argmax(explained_variance >= 0.95) + 1
                pca_components = min(pca_components, Xs_centered.shape[1] - 1)

            if pca_components < Xs_centered.shape[1]:
                pca = PCA(n_components=pca_components, svd_solver='full', random_state=42)
                Xs_reduced = pca.fit_transform(Xs_centered)
                Xt_reduced = pca.transform(Xt_centered)
            else:
                Xs_reduced = Xs_centered.copy()
                Xt_reduced = Xt_centered.copy()
        else:
            Xs_reduced = Xs_centered.copy()
            Xt_reduced = Xt_centered.copy()

        d = Xs_reduced.shape[1]

        # 修复的协方差估计函数
        def stable_cov_estimate(X, method=shrinkage):
            n_samples, n_features = X.shape

            # 基础协方差估计
            if method == 'lw' and n_samples > n_features:
                est = LedoitWolf().fit(X)
                cov = est.covariance_
            elif method == 'oas' and n_samples > n_features:
                est = OAS().fit(X)
                cov = est.covariance_
            elif isinstance(method, (float, int)) and 0 < float(method) < 1:
                est = ShrunkCovariance(shrinkage=float(method)).fit(X)
                cov = est.covariance_
            else:
                cov = np.cov(X.T)

            # 自适应正则化
            if adaptive_eps:
                trace_cov = np.trace(cov)
                adaptive_eps_value = max(eps, 1e-10 * trace_cov / n_features)
            else:
                adaptive_eps_value = eps

            # 确保正定
            cov_reg = cov + adaptive_eps_value * np.eye(n_features)
            return cov_reg, adaptive_eps_value

        # 距离计算函数
        def calculate_distance(C1, C2):
            return np.linalg.norm(C1 - C2, 'fro') ** 2 / (4.0 * d * d)

        # 初始距离
        Cs_initial, eps_used = stable_cov_estimate(Xs_reduced)
        Ct_initial, _ = stable_cov_estimate(Xt_reduced)
        distance_before = calculate_distance(Cs_initial, Ct_initial)
        print(f"CORAL distance before alignment: {distance_before:.6f}")

        # 修复的矩阵幂运算 - 使用更稳定的方法
        def stable_matrix_power(mat, power):
            """使用SVD的稳定矩阵幂运算"""
            try:
                # 尝试SVD方法，比特征分解更稳定
                U, s, Vt = np.linalg.svd(mat, full_matrices=False)
                s_powered = np.clip(s, 1e-12, None) ** power
                return U @ np.diag(s_powered) @ Vt
            except:
                # 备用方法：特征分解
                eigvals, eigvecs = np.linalg.eigh(mat)
                eigvals = np.clip(eigvals, 1e-12, None)
                return (eigvecs * (eigvals ** power)) @ eigvecs.T

        # 关键修复：使用固定的源域协方差，只变换目标域
        Cs_fixed, _ = stable_cov_estimate(Xs_reduced)
        Xt_current = Xt_reduced.copy()

        distances = [distance_before]
        best_distance = distance_before
        best_Xt = Xt_current.copy()
        convergence_count = 0

        print("Starting iterative alignment...")

        for iteration in range(n_iter):
            try:
                # 计算当前目标域协方差
                Ct_current, current_eps = stable_cov_estimate(Xt_current)

                # 计算单步变换矩阵（相对于固定源域）
                Cs_inv_sqrt = stable_matrix_power(Cs_fixed, -0.5)
                Ct_sqrt = stable_matrix_power(Ct_current, 0.5)
                A_step = Cs_inv_sqrt @ Ct_sqrt

                # 应用变换（使用较小的混合系数）
                Xt_transformed = Xt_current @ A_step

                # 功率补偿
                if power_compensation:
                    current_energy = np.mean(np.linalg.norm(Xt_transformed, axis=1) ** 2)
                    if current_energy > 0:
                        energy_ratio = np.sqrt(original_target_energy / current_energy)
                        Xt_transformed = Xt_transformed * energy_ratio

                # 渐进式混合（关键修复）
                current_blend = min(blend, 0.3 + 0.1 * iteration)  # 渐进增加混合系数
                Xt_next = (1.0 - current_blend) * Xt_current + current_blend * Xt_transformed

                # 计算新距离
                Ct_next, _ = stable_cov_estimate(Xt_next)
                current_distance = calculate_distance(Cs_fixed, Ct_next)
                distances.append(current_distance)

                print(f"Iteration {iteration + 1}: distance = {current_distance:.6f}, blend = {current_blend:.3f}")

                # 早停机制
                if early_stopping:
                    if current_distance < best_distance:
                        best_distance = current_distance
                        best_Xt = Xt_next.copy()
                        convergence_count = 0
                    else:
                        convergence_count += 1
                        if convergence_count >= 2:  # 连续2次没有改进就停止
                            print(f"Early stopping at iteration {iteration + 1}")
                            break

                # 收敛检查
                if iteration > 0:
                    improvement = distances[-2] - current_distance
                    if abs(improvement) < convergence_threshold and improvement > 0:
                        print(f"Converged at iteration {iteration + 1}")
                        break

                Xt_current = Xt_next

            except Exception as e:
                print(f"Iteration {iteration + 1} failed: {e}")
                break

        # 使用最佳结果
        if early_stopping and best_distance < distances[-1]:
            Xt_final = best_Xt
            final_distance = best_distance
            print("Using best iteration result")
        else:
            Xt_final = Xt_current
            final_distance = distances[-1]

        # 最终结果
        reduction = (distance_before - final_distance) / distance_before * 100.0
        print(f"CORAL distance after alignment:  {final_distance:.6f}")
        print(f"CORAL reduction: {reduction:.2f}%")

        # 回到原空间
        if pca is not None:
            Xt_aligned = pca.inverse_transform(Xt_final) + Xt_mean
        else:
            Xt_aligned = Xt_final + Xt_mean

        return Xt_aligned, distance_before, final_distance

    def feature_alignment(self, method='coral', **kwargs):
        """修复的特征对齐接口"""
        print(f"\n=== Feature Space Alignment using {method.upper()} ===")

        X_source_scaled = self.scaler.fit_transform(self.X_source)
        X_target_scaled = self.scaler.transform(self.X_target)

        if method.lower() == 'coral':
            # 使用更保守的默认参数
            default_kwargs = {
                'blend': 0.3,  # 更小的混合系数
                'n_iter': 3,  # 更少的迭代次数
                'early_stopping': True
            }
            # 合并用户参数和默认参数
            for k, v in default_kwargs.items():
                if k not in kwargs:
                    kwargs[k] = v

            X_target_aligned, distance_before, distance_after = self.coral_alignment(
                X_source_scaled, X_target_scaled, **kwargs
            )
        elif method.lower() == 'mmd':
            X_target_aligned, distance_before, distance_after = self.mmd_alignment(
                X_source_scaled, X_target_scaled, gamma=0.01
            )
        else:
            raise ValueError("Method must be 'mmd' or 'coral'")

        self.alignment_method = method
        self.X_target_aligned = X_target_aligned

        return X_target_aligned, distance_before, distance_after


    def feature_alignment(self, method='coral', **kwargs):
        """增强的特征空间对齐接口"""
        print(f"\n=== Enhanced Feature Space Alignment using {method.upper()} ===")

        # 标准化数据
        X_source_scaled = self.scaler.fit_transform(self.X_source)
        X_target_scaled = self.scaler.transform(self.X_target)

        if method.lower() == 'mmd':
            # 可以类似增强MMD对齐
            X_target_aligned, distance_before, distance_after = self.mmd_alignment(
                X_source_scaled, X_target_scaled, **kwargs
            )
            alignment_info = {}
        elif method.lower() == 'coral':
            X_target_aligned, distance_before, distance_after = self.coral_alignment(
                X_source_scaled, X_target_scaled, **kwargs
            )
        else:
            raise ValueError("Method must be 'mmd' or 'coral'")

        self.alignment_method = method
        self.X_target_aligned = X_target_aligned
        self.alignment_info = alignment_info

        return X_target_aligned, distance_before, distance_after, alignment_info
    
    def feature_alignment(self, method='mmd'):
        """特征空间对齐"""
        print(f"\n=== Feature Space Alignment using {method.upper()} ===")
        
        # 标准化数据
        X_source_scaled = self.scaler.fit_transform(self.X_source)
        X_target_scaled = self.scaler.transform(self.X_target)
        
        if method.lower() == 'mmd':
            X_target_aligned, distance_before, distance_after = self.mmd_alignment(
                X_source_scaled, X_target_scaled,gamma=0.01
            )
        elif method.lower() == 'coral':
            X_target_aligned, distance_before, distance_after = self.coral_alignment(
                X_source_scaled, X_target_scaled
            )
        else:
            raise ValueError("Method must be 'mmd' or 'coral'")
        
        self.alignment_method = method
        self.X_target_aligned = X_target_aligned
        
        return X_target_aligned, distance_before, distance_after
    
    def train_source_model(self):
        """训练源域模型"""
        print("\n=== Training Source Domain Model ===")
        
        # 分割源域数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_source, self.y_source, test_size=0.3, random_state=42, stratify=self.y_source
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练多个模型
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, class_weight='balanced', max_iter=1000
            ),
            'SVM': SVC(
                kernel='rbf', random_state=42, class_weight='balanced', probability=True
            )
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            print(f"{name}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.source_model = best_model
        print(f"Best source model: {best_model.__class__.__name__} (Score: {best_score:.4f})")
        
        return best_model, best_score
    
    def hierarchical_transfer(self, X_target_aligned):
        """分层迁移策略"""
        print("\n=== Hierarchical Transfer Learning ===")
        
        # 使用源域模型在目标域上进行预测
        X_target_scaled = self.scaler.transform(self.X_target)
        source_predictions = self.source_model.predict(X_target_scaled)
        source_probabilities = self.source_model.predict_proba(X_target_scaled)
        
        print(f"Source model predictions on target domain:")
        unique, counts = np.unique(source_predictions, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} samples")
        
        # 筛选高置信度样本作为伪标签
        confidence_threshold = 0.8
        max_probs = np.max(source_probabilities, axis=1)
        high_confidence_mask = max_probs >= confidence_threshold
        
        print(f"High confidence samples (>{confidence_threshold}): {np.sum(high_confidence_mask)}")
        
        if np.sum(high_confidence_mask) > 0:
            # 使用高置信度样本进行微调
            X_pseudo = X_target_scaled[high_confidence_mask]
            y_pseudo = source_predictions[high_confidence_mask]
            
            # 结合源域数据训练目标域模型
            X_combined = np.vstack([self.scaler.transform(self.X_source), X_pseudo])
            y_combined = np.hstack([self.y_source, y_pseudo])
            
            # 训练目标域模型
            self.target_model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
            self.target_model.fit(X_combined, y_combined)
            
            print("Target domain model trained with pseudo-labels")
        else:
            print("No high confidence samples found, using source model directly")
            self.target_model = self.source_model
        
        return source_predictions, source_probabilities, high_confidence_mask
    
    def pseudo_label_self_training(self, X_target_aligned, max_iterations=5):
        """伪标签自训练机制"""
        print("\n=== Pseudo-label Self-training ===")
        
        # 初始化
        current_model = self.source_model
        X_target_scaled = self.scaler.transform(self.X_target)
        
        # 自训练迭代
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}:")
            
            # 预测目标域数据
            predictions = current_model.predict(X_target_scaled)
            probabilities = current_model.predict_proba(X_target_scaled)
            
            # 计算置信度
            max_probs = np.max(probabilities, axis=1)
            confidence_threshold = 0.7 + iteration * 0.05  # 逐渐降低阈值
            
            high_confidence_mask = max_probs >= confidence_threshold
            n_high_conf = np.sum(high_confidence_mask)
            
            print(f"  Confidence threshold: {confidence_threshold:.2f}")
            print(f"  High confidence samples: {n_high_conf}")
            
            if n_high_conf == 0:
                print("  No high confidence samples, stopping iteration")
                break
            
            # 使用高置信度样本更新模型
            X_pseudo = X_target_scaled[high_confidence_mask]
            y_pseudo = predictions[high_confidence_mask]
            
            # 结合源域数据
            X_combined = np.vstack([self.scaler.transform(self.X_source), X_pseudo])
            y_combined = np.hstack([self.y_source, y_pseudo])
            
            # 训练新模型
            new_model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
            new_model.fit(X_combined, y_combined)
            
            # 检查是否收敛
            new_predictions = new_model.predict(X_target_scaled)
            if np.array_equal(predictions, new_predictions):
                print("  Model converged, stopping iteration")
                break
            
            current_model = new_model
        
        self.target_model = current_model
        final_predictions = current_model.predict(X_target_scaled)
        final_probabilities = current_model.predict_proba(X_target_scaled)
        
        return final_predictions, final_probabilities
    
    def evaluate_transfer_performance(self):
        """评估迁移性能"""
        print("\n=== Transfer Performance Evaluation ===")
        
        # 在目标域上的预测
        X_target_scaled = self.scaler.transform(self.X_target)
        target_predictions = self.target_model.predict(X_target_scaled)
        target_probabilities = self.target_model.predict_proba(X_target_scaled)
        
        # 预测结果统计
        unique, counts = np.unique(target_predictions, return_counts=True)
        print("Final target domain predictions:")
        for label, count in zip(unique, counts):
            percentage = count / len(target_predictions) * 100
            print(f"  {label}: {count} samples ({percentage:.1f}%)")
        
        # 置信度分析
        max_probs = np.max(target_probabilities, axis=1)
        print(f"\nConfidence analysis:")
        print(f"  Mean confidence: {np.mean(max_probs):.3f}")
        print(f"  Std confidence: {np.std(max_probs):.3f}")
        print(f"  Min confidence: {np.min(max_probs):.3f}")
        print(f"  Max confidence: {np.max(max_probs):.3f}")
        
        return target_predictions, target_probabilities
    
    def plot_transfer_visualization(self, save_path=None):
        """绘制迁移结果可视化"""
        print("\n=== Generating Transfer Visualization ===")
        
        # 获取预测结果
        X_target_scaled = self.scaler.transform(self.X_target)
        target_predictions = self.target_model.predict(X_target_scaled)
        target_probabilities = self.target_model.predict_proba(X_target_scaled)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测结果分布
        ax1 = axes[0, 0]
        unique, counts = np.unique(target_predictions, return_counts=True)
        # colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        colors = ['#E63946', '#F1FAEE', '#A8DADC', '#1D3557']
        ax1.pie(counts, labels=unique, autopct='%1.1f%%', colors=colors[:len(unique)])
        ax1.set_title('Target Domain Prediction Distribution')
        
        # 2. 置信度分布
        ax2 = axes[0, 1]
        max_probs = np.max(target_probabilities, axis=1)
        ax2.hist(max_probs, bins=20, alpha=0.7, color='orange', edgecolor='skyblue')
        ax2.axvline(np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.3f}')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Prediction Confidence Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 特征空间对比（使用前两个主成分）
        ax3 = axes[1, 0]
        from sklearn.decomposition import PCA
        
        # 源域数据
        X_source_scaled = self.scaler.transform(self.X_source)
        pca = PCA(n_components=2)
        X_source_pca = pca.fit_transform(X_source_scaled)
        
        # 目标域数据
        X_target_pca = pca.transform(X_target_scaled)

        # 简洁的颜色方案，源域和目标域使用相同颜色但不同深浅
        base_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # 红, 蓝, 绿, 橙

        # 绘制源域数据
        for i, label in enumerate(np.unique(self.y_source)):
            mask = self.y_source == label
            ax3.scatter(X_source_pca[mask, 0], X_source_pca[mask, 1],
                        color=base_colors[i % len(base_colors)],
                        label=f'Source {label}', alpha=0.6, s=50)

        # 绘制目标域数据 - 使用相同颜色但标记不同
        for i, label in enumerate(np.unique(target_predictions)):
            mask = target_predictions == label
            ax3.scatter(X_target_pca[mask, 0], X_target_pca[mask, 1],
                        color=base_colors[i % len(base_colors)],
                        marker='^', label=f'Target {label}', alpha=0.8, s=80)
        
        # # 绘制源域数据
        # for i, label in enumerate(np.unique(self.y_source)):
        #     mask = self.y_source == label
        #     ax3.scatter(X_source_pca[mask, 0], X_source_pca[mask, 1],
        #                label=f'Source {label}', alpha=0.6, s=50)
        #
        # # 绘制目标域数据
        # for i, label in enumerate(np.unique(target_predictions)):
        #     mask = target_predictions == label
        #     ax3.scatter(X_target_pca[mask, 0], X_target_pca[mask, 1],
        #                marker='^', label=f'Target {label}', alpha=0.8, s=80)
        
        ax3.set_xlabel('First Principal Component')
        ax3.set_ylabel('Second Principal Component')
        ax3.set_title('Feature Space: Source vs Target')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 迁移学习效果对比
        ax4 = axes[1, 1]
        
        # 源域模型在目标域的预测
        source_predictions = self.source_model.predict(X_target_scaled)
        source_unique, source_counts = np.unique(source_predictions, return_counts=True)
        
        # 目标域模型预测
        target_unique, target_counts = np.unique(target_predictions, return_counts=True)
        
        x = np.arange(len(source_unique))
        width = 0.35
        
        ax4.bar(x - width/2, source_counts, width, label='Source Model', alpha=0.8, color='#FF6B6B')
        ax4.bar(x + width/2, target_counts, width, label='Target Model', alpha=0.8, color='#4ECDC4')
        
        ax4.set_xlabel('Predicted Classes')
        ax4.set_ylabel('Number of Samples')
        ax4.set_title('Transfer Learning Effect Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(source_unique)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, predictions, probabilities, save_path='../06_迁移结果/'):
        """保存迁移结果"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'sample_id': range(len(predictions)),
            'predicted_label': predictions,
            'confidence': np.max(probabilities, axis=1)
        })
        
        # 添加各类别的概率
        class_names = self.target_model.classes_
        for i, class_name in enumerate(class_names):
            results_df[f'prob_{class_name}'] = probabilities[:, i]
        
        # 保存结果
        results_path = os.path.join(save_path, 'target_domain_predictions.csv')
        results_df.to_csv(results_path, index=False)
        
        print(f"\nResults saved to: {results_path}")
        print(f"Total samples: {len(predictions)}")
        
        return results_df

def main():
    """主函数"""
    # 创建迁移学习诊断系统
    transfer_system = TransferLearningDiagnosis('../02_特征提取/final_features.csv')
    
    # 加载数据
    transfer_system.load_data()
    
    # 特征空间对齐
    X_target_aligned, distance_before, distance_after = transfer_system.feature_alignment(method='coral')
    
    # 训练源域模型
    source_model, source_score = transfer_system.train_source_model()
    
    # 分层迁移
    source_predictions, source_probabilities, high_conf_mask = transfer_system.hierarchical_transfer(X_target_aligned)
    
    # 伪标签自训练
    final_predictions, final_probabilities = transfer_system.pseudo_label_self_training(X_target_aligned)
    
    # 评估迁移性能
    target_predictions, target_probabilities = transfer_system.evaluate_transfer_performance()
    
    # 可视化结果
    transfer_system.plot_transfer_visualization('../04_结果可视化/transfer_learning_results.png')
    
    # 保存结果
    results_df = transfer_system.save_results(target_predictions, target_probabilities, '../06_迁移结果/')
    
    return transfer_system, results_df

if __name__ == "__main__":
    transfer_system, results = main()
