import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.signal import resample, welch, find_peaks, hilbert
from scipy.fft import fft, fftfreq
import os
import glob
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveFeatureExtractor:
    
    def __init__(self, data_path, target_sr=32000):
        self.data_path = data_path
        self.target_sr = target_sr
        self.segment_length = 2048
        self.overlap_ratio = 0.5
        
        # 配置
        self.fault_types = {'B': 'Ball_Fault', 'IR': 'Inner_Race_Fault', 
                           'OR': 'Outer_Race_Fault', 'N': 'Normal'}
        self.loads = {'0': '0hp', '1': '1hp', '2': '2hp', '3': '3hp'}
        
    def load_data(self):
        """加载所有数据"""
        print("Loading data...")
        self.source_data = self._load_domain_data("源域数据集", is_source=True)
        self.target_data = self._load_domain_data("目标域数据集", is_source=False)
        print(f"Loaded: {len(self.source_data)} source files, {len(self.target_data)} target files")
        
    def _load_domain_data(self, domain_path, is_source=True):
        """统一的数据加载函数"""
        data = {}
        full_path = os.path.join(self.data_path, domain_path)
        
        if is_source:
            # 源域数据加载
            for sr_dir in ["12kHz_DE_data", "48kHz_DE_data"]:
                sr_path = os.path.join(full_path, sr_dir)
                if os.path.exists(sr_path):
                    sampling_rate = 12000 if "12kHz" in sr_dir else 48000
                    data.update(self._load_fault_data(sr_path, sampling_rate))
            
            # 正常数据
            normal_path = os.path.join(full_path, "48kHz_Normal_data")
            if os.path.exists(normal_path):
                data.update(self._load_normal_data(normal_path, 48000))
        else:
            # 目标域数据加载
            for mat_file in glob.glob(os.path.join(full_path, "*.mat")):
                try:
                    signal_data = self._extract_signal_data(loadmat(mat_file))
                    if signal_data is not None:
                        filename = os.path.basename(mat_file)
                        data[filename] = {
                            'data': signal_data,
                            'sampling_rate': self.target_sr,
                            'location': 'DE'
                        }
                except Exception as e:
                    print(f"Failed to load: {mat_file}, Error: {e}")
        
        return data
    
    def _load_fault_data(self, base_path, sampling_rate):
        """加载故障数据"""
        data = {}
        for fault_type in ['B', 'IR', 'OR']:
            fault_path = os.path.join(base_path, fault_type)
            if os.path.exists(fault_path):
                for load_dir in os.listdir(fault_path):
                    load_path = os.path.join(fault_path, load_dir)
                    if os.path.isdir(load_path):
                        for mat_file in glob.glob(os.path.join(load_path, "*.mat")):
                            try:
                                signal_data = self._extract_signal_data(loadmat(mat_file))
                                if signal_data is not None:
                                    filename = os.path.basename(mat_file)
                                    key = f"{sampling_rate}Hz_{fault_type}_{load_dir}_{filename}"
                                    data[key] = {
                                        'data': signal_data,
                                        'fault_type': self.fault_types[fault_type],
                                        'sampling_rate': sampling_rate,
                                        'location': 'DE'
                                    }
                            except Exception as e:
                                print(f"Failed to load: {mat_file}, Error: {e}")
        return data
    
    def _load_normal_data(self, normal_path, sampling_rate):
        """加载正常数据"""
        data = {}
        for mat_file in glob.glob(os.path.join(normal_path, "*.mat")):
            try:
                signal_data = self._extract_signal_data(loadmat(mat_file))
                if signal_data is not None:
                    filename = os.path.basename(mat_file)
                    data[f"{sampling_rate}Hz_Normal_{filename}"] = {
                        'data': signal_data,
                        'fault_type': 'Normal',
                        'sampling_rate': sampling_rate,
                        'location': 'DE'
                    }
            except Exception as e:
                print(f"Failed to load: {mat_file}, Error: {e}")
        return data
    
    def _extract_signal_data(self, mat_data):
        """提取信号数据"""
        possible_keys = ['X', 'x', 'data', 'signal', 'vibration', 'DE', 'FE', 'BA']
        
        for key in possible_keys:
            if key in mat_data:
                data = mat_data[key]
                if isinstance(data, np.ndarray):
                    return data.flatten() if data.ndim > 1 else data
        
        # 尝试其他字段
        for key in mat_data.keys():
            if not key.startswith('__'):
                data = mat_data[key]
                if isinstance(data, np.ndarray) and data.size > 1000:
                    return data.flatten() if data.ndim > 1 else data
        return None
    
    def preprocess_signal(self, signal_data, original_sr):
        """信号预处理 - 重采样 + 特征尺度归一化"""
        # 1. 重采样到目标采样率
        if original_sr != self.target_sr:
            num_samples = int(len(signal_data) * self.target_sr / original_sr)
            signal_data = resample(signal_data, num_samples)
        
        # 2. 去噪处理
        nyquist = self.target_sr / 2
        cutoff = nyquist * 0.8
        b, a = signal.butter(4, cutoff / nyquist, btype='low')
        signal_data = signal.filtfilt(b, a, signal_data)
        
        # 3. 特征尺度归一化
        signal_data = (signal_data - np.mean(signal_data)) / np.std(signal_data)
        
        # 4. 分段
        return self._segment_signal(signal_data)
    
    def _segment_signal(self, signal_data):
        """信号分段"""
        segment_length = self.segment_length
        overlap_length = int(segment_length * self.overlap_ratio)
        step_length = segment_length - overlap_length
        
        segments = []
        for i in range(0, len(signal_data) - segment_length + 1, step_length):
            segment = signal_data[i:i + segment_length]
            if len(segment) == segment_length:
                segments.append(segment)
        return segments
    
    def extract_time_domain_features(self, signal_data):
        """时域特征提取"""
        features = {}
        
        # 基础统计特征
        features.update({
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2)),
            'max': np.max(signal_data),
            'min': np.min(signal_data),
            'peak_to_peak': np.max(signal_data) - np.min(signal_data),
            'skewness': stats.skew(signal_data),
            'kurtosis': stats.kurtosis(signal_data),
            'energy': np.sum(signal_data**2),
            'power': np.mean(signal_data**2)
        })
        
        # 波形因子
        features.update({
            'impulse_factor': features['max'] / np.mean(np.abs(signal_data)),
            'crest_factor': features['max'] / features['rms'],
            'shape_factor': features['rms'] / np.mean(np.abs(signal_data))
        })
        
        # 冲击指标 - 重点关注跨域鲁棒性
        signal_abs = np.abs(signal_data)
        threshold = np.mean(signal_abs) + 3 * np.std(signal_abs)
        impact_indices = signal_abs > threshold
        features['impact_energy_ratio'] = np.sum(signal_data[impact_indices]**2) / features['energy']
        
        # 循环平稳性度量 - 反映周期性冲击
        features['cyclostationarity'] = self._calculate_cyclostationarity(signal_data)
        
        return features
    
    def _calculate_cyclostationarity(self, signal_data):
        """计算循环平稳性度量"""
        # 简化的循环平稳性计算
        # 使用自相关函数的周期性来度量
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # 寻找周期性峰值
        peaks, _ = find_peaks(autocorr[:len(autocorr)//4], height=np.max(autocorr)*0.1)
        if len(peaks) > 1:
            period = np.mean(np.diff(peaks))
            return 1.0 / period if period > 0 else 0
        return 0
    
    def extract_frequency_domain_features(self, signal_data):
        """频域特征提取"""
        features = {}
        
        # FFT分析
        fft_data = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.target_sr)
        magnitude = np.abs(fft_data)
        
        # 只取正频率部分
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # 基础频域特征
        spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
        features.update({
            'spectral_centroid': spectral_centroid,
            'spectral_spread': np.sqrt(np.sum(((positive_freqs - spectral_centroid)**2) * positive_magnitude) / np.sum(positive_magnitude)),
            'spectral_rolloff': self._calculate_spectral_rolloff(positive_freqs, positive_magnitude),
            'spectral_flux': self._calculate_spectral_flux(positive_magnitude)
        })
        
        # 功率谱密度分析
        freqs_psd, psd = welch(signal_data, fs=self.target_sr, nperseg=min(512, len(signal_data)//4))
        features.update({
            'psd_mean': np.mean(psd),
            'psd_std': np.std(psd),
            'psd_max': np.max(psd),
            'psd_peak_freq': freqs_psd[np.argmax(psd)]
        })
        
        # 阶次分析 - 缓解转速差异
        features.update(self._order_analysis(signal_data))
        
        return features
    
    def _calculate_spectral_rolloff(self, freqs, magnitude, rolloff_threshold=0.85):
        """计算频谱滚降点"""
        cumsum = np.cumsum(magnitude)
        rolloff_index = np.where(cumsum >= rolloff_threshold * cumsum[-1])[0]
        return freqs[rolloff_index[0]] if len(rolloff_index) > 0 else freqs[-1]
    
    def _calculate_spectral_flux(self, magnitude):
        """计算频谱通量"""
        if len(magnitude) < 2:
            return 0
        diff = np.diff(magnitude)
        return np.sum(np.abs(diff))
    
    def _order_analysis(self, signal_data):
        """阶次分析 - 频率归一化到转速"""
        features = {}
        
        # 简化的阶次分析
        # 假设转速为600rpm (10Hz)
        assumed_rpm = 600
        fundamental_freq = assumed_rpm / 60  # 10Hz
        
        # 计算主要阶次频率
        order_freqs = [fundamental_freq * i for i in range(1, 21)]  # 1-20阶
        
        # FFT分析
        fft_data = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.target_sr)
        magnitude = np.abs(fft_data)
        
        # 计算各阶次幅值
        order_amplitudes = []
        for order_freq in order_freqs:
            # 找到最接近的频率索引
            freq_idx = np.argmin(np.abs(freqs - order_freq))
            if freq_idx < len(magnitude):
                order_amplitudes.append(magnitude[freq_idx])
            else:
                order_amplitudes.append(0)
        
        features['order_1_amplitude'] = order_amplitudes[0]
        features['order_2_amplitude'] = order_amplitudes[1]
        features['order_3_amplitude'] = order_amplitudes[2]
        features['order_energy'] = np.sum(np.array(order_amplitudes)**2)
        
        return features
    
    def extract_time_frequency_features(self, signal_data):
        """时频域特征提取 - 使用VMD和包络分析"""
        features = {}
        
        # 包络分析
        envelope = np.abs(hilbert(signal_data))
        features.update({
            'envelope_mean': np.mean(envelope),
            'envelope_std': np.std(envelope),
            'envelope_kurtosis': stats.kurtosis(envelope),
            'envelope_skewness': stats.skew(envelope)
        })
        
        # 简化的VMD特征
        features.update(self._vmd_features(signal_data))
        
        # 小波变换特征
        features.update(self._wavelet_features(signal_data))
        
        return features
    
    def _vmd_features(self, signal_data):
        """变分模态分解特征"""
        features = {}
        
        # 简化的VMD实现
        # 使用经验模态分解(EMD)的简化版本
        try:
            # 这里使用简化的分解方法
            # 实际应用中可以使用PyEMD库
            
            # 分解为3个模态
            modes = self._simple_emd(signal_data, num_modes=3)
            
            for i, mode in enumerate(modes):
                features[f'vmd_mode_{i+1}_energy'] = np.sum(mode**2)
                features[f'vmd_mode_{i+1}_std'] = np.std(mode)
                features[f'vmd_mode_{i+1}_kurtosis'] = stats.kurtosis(mode)
            
            # 重构误差
            reconstructed = np.sum(modes, axis=0)
            features['vmd_reconstruction_error'] = np.mean((signal_data - reconstructed)**2)
            
        except Exception as e:
            print(f"VMD feature extraction failed: {e}")
            # 使用默认值
            for i in range(3):
                features[f'vmd_mode_{i+1}_energy'] = 0
                features[f'vmd_mode_{i+1}_std'] = 0
                features[f'vmd_mode_{i+1}_kurtosis'] = 0
            features['vmd_reconstruction_error'] = 0
        
        return features
    
    def _simple_emd(self, signal_data, num_modes=3):
        """简化的经验模态分解"""
        modes = []
        residual = signal_data.copy()
        
        for _ in range(num_modes):
            # 简化的分解过程
            # 实际应用中应该使用完整的EMD算法
            
            # 找到局部极值
            maxima_indices = find_peaks(residual)[0]
            minima_indices = find_peaks(-residual)[0]
            
            if len(maxima_indices) > 1 and len(minima_indices) > 1:
                # 插值得到上下包络
                maxima_values = residual[maxima_indices]
                minima_values = residual[minima_indices]
                
                # 简化的包络估计
                upper_envelope = np.interp(np.arange(len(residual)), maxima_indices, maxima_values)
                lower_envelope = np.interp(np.arange(len(residual)), minima_indices, minima_values)
                
                # 计算均值
                mean_envelope = (upper_envelope + lower_envelope) / 2
                
                # 提取模态
                mode = residual - mean_envelope
                modes.append(mode)
                residual = mean_envelope
            else:
                modes.append(residual)
                break
        
        return modes
    
    def _wavelet_features(self, signal_data):
        """小波变换特征"""
        features = {}
        
        try:
            # 使用简化的离散小波变换
            # 实际应用中可以使用PyWavelets库
            
            # 简化的多尺度分析
            scales = [2, 4, 8, 16]
            
            for scale in scales:
                # 简化的尺度分析
                downsampled = signal_data[::scale]
                if len(downsampled) > 10:
                    features[f'wavelet_scale_{scale}_energy'] = np.sum(downsampled**2)
                    features[f'wavelet_scale_{scale}_std'] = np.std(downsampled)
                else:
                    features[f'wavelet_scale_{scale}_energy'] = 0
                    features[f'wavelet_scale_{scale}_std'] = 0
            
        except Exception as e:
            print(f"Wavelet feature extraction failed: {e}")
            for scale in [2, 4, 8, 16]:
                features[f'wavelet_scale_{scale}_energy'] = 0
                features[f'wavelet_scale_{scale}_std'] = 0
        
        return features
    
    def extract_all_features(self, signal_data):
        """提取所有特征"""
        features = {}
        
        # 时域特征
        features.update(self.extract_time_domain_features(signal_data))
        
        # 频域特征
        features.update(self.extract_frequency_domain_features(signal_data))
        
        # 时频域特征
        features.update(self.extract_time_frequency_features(signal_data))
        
        return features
    
    def build_comprehensive_dataset(self):
        """构建全面的数据集"""
        print("Building comprehensive dataset...")
        
        source_features, target_features = [], []
        source_labels = []
        
        # 处理源域数据
        for filename, data_info in self.source_data.items():
            segments = self.preprocess_signal(data_info['data'], data_info['sampling_rate'])
            for segment in segments[:3]:  # 限制分段数量
                features = self.extract_all_features(segment)
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
        
        print(f"Comprehensive dataset built: {len(source_df)} source samples, {len(target_df)} target samples")
        print(f"Total features: {len(source_df.columns) - 2}")  # 减去domain和label列
        print(f"Source labels: {dict(pd.Series(source_labels).value_counts())}")
        
        return self.dataset
    
    def plot_domain_feature_distributions(self, save_path=None):
        """绘制源域和目标域特征分布对比"""
        # 选择部分重要特征进行可视化
        feature_cols = [col for col in self.dataset.columns if col not in ['domain', 'label']]
        important_features = feature_cols[:12]  # 选择前12个特征
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, feature in enumerate(important_features):
            if i < len(axes):
                # 源域和目标域数据
                source_data = self.dataset[self.dataset['domain'] == 'Source'][feature]
                target_data = self.dataset[self.dataset['domain'] == 'Target'][feature]
                
                # 绘制KDE图
                axes[i].hist(source_data, bins=30, alpha=0.7, label='Source', density=True, color='blue')
                axes[i].hist(target_data, bins=30, alpha=0.7, label='Target', density=True, color='red')
                axes[i].set_title(f'{feature}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_robustness(self):
        """分析特征跨域鲁棒性"""
        print("\n=== Feature Robustness Analysis ===")
        
        feature_cols = [col for col in self.dataset.columns if col not in ['domain', 'label']]
        robustness_scores = {}
        
        for feature in feature_cols:
            source_data = self.dataset[self.dataset['domain'] == 'Source'][feature]
            target_data = self.dataset[self.dataset['domain'] == 'Target'][feature]
            
            # 计算Wasserstein距离
            from scipy.stats import wasserstein_distance
            wd = wasserstein_distance(source_data, target_data)
            
            # 计算重叠度
            min_val = min(source_data.min(), target_data.min())
            max_val = max(source_data.max(), target_data.max())
            
            # 创建直方图
            bins = np.linspace(min_val, max_val, 50)
            source_hist, _ = np.histogram(source_data, bins=bins, density=True)
            target_hist, _ = np.histogram(target_data, bins=bins, density=True)
            
            # 计算重叠面积
            overlap = np.minimum(source_hist, target_hist)
            overlap_score = np.sum(overlap) / np.sum(np.maximum(source_hist, target_hist))
            
            robustness_scores[feature] = {
                'wasserstein_distance': wd,
                'overlap_score': overlap_score,
                'robustness_score': 1 / (1 + wd) * overlap_score
            }
        
        # 按鲁棒性得分排序
        sorted_features = sorted(robustness_scores.items(), 
                               key=lambda x: x[1]['robustness_score'], reverse=True)
        
        print("Top 10 most robust features:")
        for i, (feature, scores) in enumerate(sorted_features[:10]):
            print(f"{i+1:2d}. {feature:25s} - Robustness: {scores['robustness_score']:.4f}")
        
        return robustness_scores

def main():
    """主函数"""
    data_path = "../数据集"
    
    # 创建特征提取器
    extractor = ComprehensiveFeatureExtractor(data_path)
    
    # 加载数据
    extractor.load_data()
    
    # 构建全面数据集
    dataset = extractor.build_comprehensive_dataset()
    
    # 保存数据集
    dataset.to_csv('../02_特征提取/comprehensive_features.csv', index=False)
    print("Comprehensive features saved to comprehensive_features.csv")
    
    # 绘制特征分布
    extractor.plot_domain_feature_distributions('../04_结果可视化/comprehensive_feature_distributions.png')
    
    # 分析特征鲁棒性
    robustness_scores = extractor.analyze_feature_robustness()
    
    return extractor, dataset

if __name__ == "__main__":
    extractor, dataset = main()
