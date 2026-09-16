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
        self.df = pd.read_csv(self.data_path)
        self.source_df = self.df[self.df['domain'] == 'Source'].copy()
        self.target_df = self.df[self.df['domain'] == 'Target'].copy()

        feature_cols = [col for col in self.df.columns if col not in ['domain', 'label']]
        self.feature_names = feature_cols

        self.X_source = self.source_df[feature_cols]
        self.y_source = self.source_df['label']
        self.X_target = self.target_df[feature_cols]

        return self.X_source, self.y_source, self.X_target

    def mmd_alignment(self, X_source, X_target, gamma=1.0):
        """最大均值差异对齐"""

        def rbf_kernel(X, Y, gamma):
            X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
            dist = X_norm + Y_norm - 2 * np.dot(X, Y.T)
            return np.exp(-gamma * dist)

        K_ss = rbf_kernel(X_source, X_source, gamma)
        K_tt = rbf_kernel(X_target, X_target, gamma)
        K_st = rbf_kernel(X_source, X_target, gamma)
        mmd = np.mean(K_ss) + np.mean(K_tt) - 2 * np.mean(K_st)

        X_source_mean = np.mean(X_source, axis=0)
        X_source_std = np.std(X_source, axis=0)
        X_target_mean = np.mean(X_target, axis=0)
        X_target_std = np.std(X_target, axis=0)

        X_target_std[X_target_std == 0] = 1
        X_source_std[X_source_std == 0] = 1

        X_target_aligned = (X_target - X_target_mean) / X_target_std * X_source_std + X_source_mean

        return X_target_aligned, mmd, mmd

    def coral_alignment(self, X_source, X_target, eps=1e-5, blend=0.3, n_iter=3):
        """CORAL特征对齐（简化版）"""
        Xs = np.asarray(X_source, dtype=np.float64)
        Xt = np.asarray(X_target, dtype=np.float64)

        Xs_mean = np.mean(Xs, axis=0)
        Xt_mean = np.mean(Xt, axis=0)
        Xs_centered = Xs - Xs_mean
        Xt_centered = Xt - Xt_mean

        Cs = np.cov(Xs_centered.T) + eps * np.eye(Xs.shape[1])
        Ct = np.cov(Xt_centered.T) + eps * np.eye(Xt.shape[1])

        def stable_matrix_power(mat, power):
            U, s, Vt = np.linalg.svd(mat, full_matrices=False)
            s_powered = np.clip(s, 1e-12, None) ** power
            return U @ np.diag(s_powered) @ Vt

        Cs_inv_sqrt = stable_matrix_power(Cs, -0.5)
        Ct_sqrt = stable_matrix_power(Ct, 0.5)
        A = Cs_inv_sqrt @ Ct_sqrt

        Xt_aligned_full = Xt_centered @ A
        Xt_aligned = (1.0 - blend) * Xt_centered + blend * Xt_aligned_full + Xt_mean

        return Xt_aligned, 0, 0  # 简化返回

    def feature_alignment(self, method='coral', **kwargs):
        """特征空间对齐接口"""
        X_source_scaled = self.scaler.fit_transform(self.X_source)
        X_target_scaled = self.scaler.transform(self.X_target)

        if method.lower() == 'mmd':
            X_target_aligned, before, after = self.mmd_alignment(X_source_scaled, X_target_scaled)
        elif method.lower() == 'coral':
            X_target_aligned, before, after = self.coral_alignment(X_source_scaled, X_target_scaled, **kwargs)
        else:
            raise ValueError("Method must be 'mmd' or 'coral'")

        self.alignment_method = method
        self.X_target_aligned = X_target_aligned

        return X_target_aligned, before, after

    def train_source_model(self):
        """训练源域模型"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_source, self.y_source, test_size=0.3, random_state=42, stratify=self.y_source
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'SVM': SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True)
        }

        best_model, best_score = None, 0
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            if score > best_score:
                best_score, best_model = score, model

        self.source_model = best_model
        return best_model, best_score

    def hierarchical_transfer(self, X_target_aligned):
        """分层迁移策略"""
        X_target_scaled = self.scaler.transform(self.X_target)
        source_predictions = self.source_model.predict(X_target_scaled)
        source_probabilities = self.source_model.predict_proba(X_target_scaled)

        confidence_threshold = 0.8
        max_probs = np.max(source_probabilities, axis=1)
        high_confidence_mask = max_probs >= confidence_threshold

        if np.sum(high_confidence_mask) > 0:
            X_pseudo = X_target_scaled[high_confidence_mask]
            y_pseudo = source_predictions[high_confidence_mask]

            X_combined = np.vstack([self.scaler.transform(self.X_source), X_pseudo])
            y_combined = np.hstack([self.y_source, y_pseudo])

            self.target_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            self.target_model.fit(X_combined, y_combined)
        else:
            self.target_model = self.source_model

        return source_predictions, source_probabilities, high_confidence_mask

    def pseudo_label_self_training(self, X_target_aligned, max_iterations=5):
        """伪标签自训练"""
        current_model = self.source_model
        X_target_scaled = self.scaler.transform(self.X_target)

        for iteration in range(max_iterations):
            predictions = current_model.predict(X_target_scaled)
            probabilities = current_model.predict_proba(X_target_scaled)
            max_probs = np.max(probabilities, axis=1)
            confidence_threshold = 0.7 + iteration * 0.05

            high_confidence_mask = max_probs >= confidence_threshold
            if np.sum(high_confidence_mask) == 0:
                break

            X_pseudo = X_target_scaled[high_confidence_mask]
            y_pseudo = predictions[high_confidence_mask]

            X_combined = np.vstack([self.scaler.transform(self.X_source), X_pseudo])
            y_combined = np.hstack([self.y_source, y_pseudo])

            new_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            new_model.fit(X_combined, y_combined)

            if np.array_equal(predictions, new_model.predict(X_target_scaled)):
                break
            current_model = new_model

        self.target_model = current_model
        return current_model.predict(X_target_scaled), current_model.predict_proba(X_target_scaled)

    def evaluate_transfer_performance(self):
        """评估迁移性能"""
        X_target_scaled = self.scaler.transform(self.X_target)
        target_predictions = self.target_model.predict(X_target_scaled)
        target_probabilities = self.target_model.predict_proba(X_target_scaled)

        print("Final target domain predictions:")
        unique, counts = np.unique(target_predictions, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} samples ({count / len(target_predictions) * 100:.1f}%)")

        return target_predictions, target_probabilities


def main():
    """主流程示例"""
    transfer_system = TransferLearningDiagnosis('../02_特征提取/final_features.csv')
    transfer_system.load_data()
    X_target_aligned, _, _ = transfer_system.feature_alignment(method='coral')
    transfer_system.train_source_model()
    transfer_system.hierarchical_transfer(X_target_aligned)
    predictions, probabilities = transfer_system.evaluate_transfer_performance()
    return transfer_system, predictions

