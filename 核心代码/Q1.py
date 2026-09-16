# 主控流程 - 构建完整数据集
def build_comprehensive_dataset(self):
    """构建全面的数据集 - 核心主流程"""
    source_features, target_features = [], []
    source_labels = []

    # 处理源域数据
    for filename, data_info in self.source_data.items():
        segments = self.preprocess_signal(data_info['data'], data_info['sampling_rate'])
        for segment in segments[:3]:
            features = self.extract_all_features(segment)  # ⭐核心调用
            source_features.append(features)
            source_labels.append(data_info['fault_type'])

    # 处理目标域数据
    for filename, data_info in self.target_data.items():
        segments = self.preprocess_signal(data_info['data'], data_info['sampling_rate'])
        for segment in segments[:3]:
            features = self.extract_all_features(segment)
            target_features.append(features)

    # 创建DataFrame
    source_df = pd.DataFrame(source_features)
    source_df['domain'] = 'Source'
    source_df['label'] = source_labels
    target_df = pd.DataFrame(target_features)
    target_df['domain'] = 'Target'
    self.dataset = pd.concat([source_df, target_df], ignore_index=True)
    return self.dataset

# 特征提取总入口
def extract_all_features(self, signal_data):
    """提取所有特征 - 特征提取核心"""
    features = {}
    features.update(self.extract_time_domain_features(signal_data))      # 时域
    features.update(self.extract_frequency_domain_features(signal_data)) # 频域
    features.update(self.extract_time_frequency_features(signal_data))   # 时频域
    return features

# 信号预处理核心
def preprocess_signal(self, signal_data, original_sr):
    """信号预处理 - 跨域适配关键步骤"""
    # 1. 重采样到目标采样率（消除硬件差异）
    if original_sr != self.target_sr:
        num_samples = int(len(signal_data) * self.target_sr / original_sr)
        signal_data = resample(signal_data, num_samples)

    # 2. 滤波去噪
    nyquist = self.target_sr / 2
    cutoff = nyquist * 0.8
    b, a = signal.butter(4, cutoff / nyquist, btype='low')
    signal_data = signal.filtfilt(b, a, signal_data)

    # 3. 标准化（消除幅值差异）
    signal_data = (signal_data - np.mean(signal_data)) / np.std(signal_data)

    # 4. 分段处理
    return self._segment_signal(signal_data)

# 时域特征提取核心
def extract_time_domain_features(self, signal_data):
    """时域特征 - 故障诊断最有效特征"""
    features = {}

    # 基础统计特征
    features.update({
        'mean': np.mean(signal_data),
        'std': np.std(signal_data),
        'rms': np.sqrt(np.mean(signal_data ** 2)),
        'kurtosis': stats.kurtosis(signal_data),  # ⭐故障敏感指标
        'skewness': stats.skew(signal_data),
    })

    # 冲击特征（轴承故障关键指标）
    signal_abs = np.abs(signal_data)
    threshold = np.mean(signal_abs) + 3 * np.std(signal_abs)
    impact_indices = signal_abs > threshold
    features['impact_energy_ratio'] = np.sum(signal_data[impact_indices] ** 2) / features['energy']

    return features

# 频域特征提取核心
def extract_frequency_domain_features(self, signal_data):
    """频域特征 - 旋转机械故障诊断核心"""
    features = {}

    # FFT分析
    fft_data = fft(signal_data)
    freqs = fftfreq(len(signal_data), 1 / self.target_sr)
    magnitude = np.abs(fft_data)

    # 频谱特征
    positive_freqs = freqs[:len(freqs) // 2]
    positive_magnitude = magnitude[:len(magnitude) // 2]

    spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
    features.update({
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': self._calculate_spectral_rolloff(positive_freqs, positive_magnitude),
    })

    # 阶次分析（跨域鲁棒性关键）
    features.update(self._order_analysis(signal_data))

    return features
