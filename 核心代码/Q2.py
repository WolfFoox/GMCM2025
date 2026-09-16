# 数据加载与预处理
def load_data(self):
    """加载源域数据"""
    self.df = pd.read_csv(self.data_path)
    self.source_df = self.df[self.df['domain'] == 'Source'].copy()

def prepare_features(self):
    """准备特征数据"""
    feature_cols = [col for col in self.source_df.columns
                   if col not in ['domain', 'label']]
    self.X = self.source_df[feature_cols]
    self.y = self.source_df['label']

# 类别不平衡处理
def handle_class_imbalance(self):
    """计算类别权重处理不平衡数据"""
    class_weights = compute_class_weight('balanced',
                                       classes=np.unique(self.y),
                                       y=self.y)
    self.class_weight_dict = dict(zip(np.unique(self.y), class_weights))

# 数据分割与标准化
def split_data(self):
    """分层分割数据并标准化"""
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
    )
    # 使用RobustScaler标准化
    self.X_train_scaled = self.scaler.fit_transform(self.X_train)
    self.X_test_scaled = self.scaler.transform(self.X_test)

# 多模型训练
def train_models(self):
    """训练9种不同的分类模型"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
        'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True),
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': GaussianNB(),
        'CatBoost': CatBoostClassifier(iterations=500)
    }


