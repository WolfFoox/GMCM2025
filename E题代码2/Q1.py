import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# 设置中文字体
def setup_chinese_font():
    """设置中文字体显示"""
    try:
        # 设置matplotlib参数
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # 尝试找到可用的中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'Arial Unicode MS']
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        found_font = None
        for font in chinese_fonts:
            if font in available_fonts:
                found_font = font
                break

        if found_font:
            plt.rcParams['font.sans-serif'] = [found_font, 'DejaVu Sans']
            return True
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            return False
    except Exception as e:
        print(f"字体设置警告: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        return False


USE_CHINESE = setup_chinese_font()


class SimplifiedBearingDataProcessor:
    """简化版轴承故障数据预处理类（带可视化）"""

    def __init__(self, data_root_path):
        """
        初始化数据处理器

        Args:
            data_root_path: 数据集根目录路径
        """
        self.data_root_path = Path(data_root_path)
        self.source_data_path = self.data_root_path / "源域数据集"
        self.target_data_path = self.data_root_path / "目标域数据集"

        # 轴承参数（根据文档中的表格）
        self.bearing_params = {
            'SKF6205': {'n_balls': 9, 'd': 0.3126, 'D': 1.537},  # 驱动端
            'SKF6203': {'n_balls': 9, 'd': 0.2656, 'D': 1.122}  # 风扇端
        }

        # 故障类型映射
        if USE_CHINESE:
            self.fault_types = {
                'B': '滚动体故障',
                'IR': '内圈故障',
                'OR': '外圈故障',
                'N': '正常状态'
            }
        else:
            self.fault_types = {
                'B': 'Ball Fault',
                'IR': 'Inner Race Fault',
                'OR': 'Outer Race Fault',
                'N': 'Normal'
            }

        # 故障尺寸
        self.fault_sizes = ['0007', '0014', '0021', '0028']

        # 载荷等级
        self.load_levels = ['0', '1', '2', '3']

        # OR故障的位置类型
        self.or_positions = ['Centered', 'Opposite', 'Orthogonal']

        # 存储样本信号用于可视化
        self.sample_signals = {}

    def load_mat_file(self, file_path):
        """
        加载MATLAB .mat文件

        Args:
            file_path: .mat文件路径

        Returns:
            dict: 包含振动数据的字典
        """
        try:
            mat_data = sio.loadmat(file_path)

            # 提取有用的数据，过滤掉MATLAB的元数据
            data = {}
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    data[key] = value

            return data
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return None

    def extract_signal_from_mat(self, mat_data):
        """
        从mat数据中提取振动信号

        Args:
            mat_data: 加载的mat数据

        Returns:
            dict: 包含不同位置振动信号的字典
        """
        signals = {}
        rpm = None

        for key, value in mat_data.items():
            if 'DE_time' in key:
                signals['DE'] = value.flatten()
            elif 'FE_time' in key:
                signals['FE'] = value.flatten()
            elif 'BA_time' in key:
                signals['BA'] = value.flatten()
            elif 'RPM' in key:
                rpm = value.flatten()[0] if len(value.flatten()) > 0 else None

        signals['RPM'] = rpm
        return signals

    def calculate_fault_frequencies(self, rpm, bearing_type='SKF6205'):
        """
        计算轴承故障特征频率

        Args:
            rpm: 转速 (转/分钟)
            bearing_type: 轴承型号

        Returns:
            dict: 故障特征频率
        """
        if rpm is None:
            return {}

        params = self.bearing_params[bearing_type]
        fr = rpm / 60  # 转频 Hz
        n = params['n_balls']
        d = params['d']
        D = params['D']

        # 根据公式计算故障特征频率
        bpfo = fr * n / 2 * (1 - d / D)  # 外圈故障特征频率
        bpfi = fr * n / 2 * (1 + d / D)  # 内圈故障特征频率
        bsf = fr * D / d * (1 - (d / D) ** 2)  # 滚动体故障特征频率
        ftf = fr / 2 * (1 - d / D)  # 滚动体公转频率

        return {
            'BPFO': bpfo,
            'BPFI': bpfi,
            'BSF': bsf,
            'FTF': ftf,
            'FR': fr
        }

    def extract_time_domain_features(self, signal_data):
        """
        提取核心时域特征（简化版）

        Args:
            signal_data: 振动信号

        Returns:
            dict: 时域特征
        """
        features = {}

        # 核心统计特征
        features['rms'] = np.sqrt(np.mean(signal_data ** 2))
        features['peak'] = np.max(np.abs(signal_data))
        features['std'] = np.std(signal_data)

        # 关键形状特征
        features['kurtosis'] = self._kurtosis(signal_data)  # 峭度是轴承故障诊断的关键指标
        features['skewness'] = self._skewness(signal_data)  # 偏度

        # 重要的无量纲特征
        if features['rms'] > 0:
            features['crest_factor'] = features['peak'] / features['rms']  # 峰值因子
        else:
            features['crest_factor'] = 0

        return features

    def _skewness(self, signal_data):
        """计算偏度"""
        if np.std(signal_data) == 0:
            return 0
        return np.mean(((signal_data - np.mean(signal_data)) / np.std(signal_data)) ** 3)

    def _kurtosis(self, signal_data):
        """计算峭度"""
        if np.std(signal_data) == 0:
            return 0
        return np.mean(((signal_data - np.mean(signal_data)) / np.std(signal_data)) ** 4)

    def extract_frequency_domain_features(self, signal_data, fs, fault_freqs=None):
        """
        提取核心频域特征（简化版）

        Args:
            signal_data: 振动信号
            fs: 采样频率
            fault_freqs: 故障特征频率字典

        Returns:
            dict: 频域特征
        """
        features = {}

        # FFT变换
        n = len(signal_data)
        fft_vals = fft(signal_data)
        freqs = fftfreq(n, 1 / fs)[:n // 2]
        magnitude = np.abs(fft_vals[:n // 2])

        # 基本频域特征
        features['freq_mean'] = np.mean(magnitude)
        features['freq_std'] = np.std(magnitude)

        # 频谱重心
        if np.sum(magnitude) > 0:
            features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            features['spectral_centroid'] = 0

        # 故障特征频率相关特征（这是轴承故障诊断的核心）
        if fault_freqs:
            for fault_name in ['BPFO', 'BPFI', 'BSF']:  # 只保留最重要的三个故障频率
                fault_freq = fault_freqs.get(fault_name, 0)
                if fault_freq > 0 and fault_freq < fs / 2:
                    # 找到最接近故障频率的索引
                    idx = np.argmin(np.abs(freqs - fault_freq))
                    features[f'{fault_name}_amplitude'] = magnitude[idx]

                    # 故障频率的前3个谐波能量（简化版）
                    harmonics_energy = 0
                    for h in range(1, 4):  # 前3个谐波
                        harmonic_freq = fault_freq * h
                        if harmonic_freq < fs / 2:
                            h_idx = np.argmin(np.abs(freqs - harmonic_freq))
                            harmonics_energy += magnitude[h_idx]
                    features[f'{fault_name}_harmonics_energy'] = harmonics_energy

        return features

    def extract_time_frequency_features(self, signal_data, fs):
        """
        提取简化的时频域特征

        Args:
            signal_data: 振动信号
            fs: 采样频率

        Returns:
            dict: 时频域特征
        """
        features = {}

        # 使用Welch方法计算功率谱密度
        from scipy.signal import welch

        try:
            freqs, psd = welch(signal_data, fs, nperseg=min(1024, len(signal_data) // 4))
        except:
            freqs, psd = welch(signal_data, fs, nperseg=min(256, len(signal_data) // 2))

        # 简化的分频带能量（只分两个主要频带）
        freq_bands = [
            (0, fs / 4, 'low'),  # 低频段
            (fs / 4, fs / 2, 'high')  # 高频段
        ]

        total_energy = np.sum(psd)
        for f_low, f_high, band_name in freq_bands:
            band_mask = (freqs >= f_low) & (freqs < f_high)
            band_energy = np.sum(psd[band_mask])
            features[f'{band_name}_band_ratio'] = band_energy / total_energy if total_energy > 0 else 0

        return features

    def preprocess_signal(self, signal_data, fs):
        """
        信号预处理

        Args:
            signal_data: 原始振动信号
            fs: 采样频率

        Returns:
            np.array: 预处理后的信号
        """
        # 去除直流分量
        signal_data = signal_data - np.mean(signal_data)

        # 带通滤波（去除低频噪声和高频噪声）
        try:
            nyquist = fs / 2
            low_cutoff = 10 / nyquist  # 10 Hz
            high_cutoff = min(5000 / nyquist, 0.95)  # 5000 Hz或接近奈奎斯特频率

            if low_cutoff < high_cutoff and low_cutoff > 0 and high_cutoff < 1:
                b, a = scipy_signal.butter(4, [low_cutoff, high_cutoff], btype='band')
                signal_data = scipy_signal.filtfilt(b, a, signal_data)
        except:
            # 如果滤波失败，返回原信号
            pass

        return signal_data

    def parse_file_path_info(self, file_path):
        """
        解析文件路径，提取故障信息

        Args:
            file_path: 文件路径

        Returns:
            dict: 包含故障信息的字典
        """
        path_parts = Path(file_path).parts
        filename = Path(file_path).stem

        info = {
            'fault_type': None,
            'fault_size': None,
            'load_level': None,
            'sensor_type': None,
            'or_position': None,
            'sampling_rate': None
        }

        # 确定采样率和传感器类型
        for part in path_parts:
            if '12kHz_DE' in part:
                info['sensor_type'] = 'DE'
                info['sampling_rate'] = 12000
            elif '12kHz_FE' in part:
                info['sensor_type'] = 'FE'
                info['sampling_rate'] = 12000
            elif '48kHz_DE' in part:
                info['sensor_type'] = 'DE'
                info['sampling_rate'] = 48000
            elif '48kHz_Normal' in part:
                info['sensor_type'] = 'Normal'
                info['sampling_rate'] = 48000
                info['fault_type'] = 'N'

        # 确定故障类型
        for part in path_parts:
            if part in ['B', 'IR', 'OR']:
                info['fault_type'] = part
                break

        # 确定故障尺寸
        for part in path_parts:
            if part in self.fault_sizes:
                info['fault_size'] = part
                break

        # 确定OR位置类型
        for part in path_parts:
            if part in self.or_positions:
                info['or_position'] = part
                break

        # 从文件名提取载荷信息
        for load in self.load_levels:
            if filename.endswith(f'_{load}'):
                info['load_level'] = load
                break

        return info

    def process_single_file(self, file_path, save_sample_signals=False, require_fault_type=True):
        """
        处理单个.mat文件

        Args:
            file_path: 文件路径
            save_sample_signals: 是否保存样本信号用于可视化

        Returns:
            dict: 提取的特征
        """
        # 解析文件路径获取标签信息
        info = self.parse_file_path_info(file_path)

        # 加载数据
        mat_data = self.load_mat_file(file_path)
        if mat_data is None:
            return None

        # 提取信号
        signals = self.extract_signal_from_mat(mat_data)

        result = {
            'file_path': str(file_path),
            'fault_type': info['fault_type'],
            'fault_size': info['fault_size'],
            'load_level': info['load_level'],
            'sensor_type': info['sensor_type'],
            'or_position': info['or_position'],
            'sampling_rate': info['sampling_rate'],
            'rpm': signals.get('RPM')
        }

        # 如果没有识别出故障类型
        if result['fault_type'] is None:
            if require_fault_type:
                print(f"无法识别故障类型: {file_path}")
                return None
            else:
                # 允许目标域无标签样本进入后续流程
                result['fault_type'] = 'Unknown'

        # 计算故障特征频率
        bearing_type = 'SKF6205' if info['sensor_type'] == 'DE' else 'SKF6203'
        fault_freqs = self.calculate_fault_frequencies(signals.get('RPM'), bearing_type)

        # 获取采样频率
        fs = info['sampling_rate'] if info['sampling_rate'] else 12000

        # 保存样本信号用于可视化（每种故障类型保存一个样本）
        if save_sample_signals and result['fault_type'] not in self.sample_signals:
            if 'DE' in signals and signals['DE'] is not None and len(signals['DE']) > 0:
                signal_data = signals['DE']
                processed_signal = self.preprocess_signal(signal_data, fs)
                if len(processed_signal) >= 1000:  # 确保信号足够长
                    self.sample_signals[result['fault_type']] = {
                        'raw_signal': signal_data[:10000],  # 保存前10000个点
                        'processed_signal': processed_signal[:10000],
                        'fs': fs,
                        'fault_freqs': fault_freqs,
                        'file_path': str(file_path)
                    }

        # 处理每个传感器位置的信号（优先处理DE数据）
        for pos in ['DE', 'FE', 'BA']:
            if pos in signals and signals[pos] is not None and len(signals[pos]) > 0:
                signal_data = signals[pos]

                # 信号预处理
                processed_signal = self.preprocess_signal(signal_data, fs)

                # 如果信号太短，跳过
                if len(processed_signal) < 100:
                    continue

                # 提取简化的特征
                time_features = self.extract_time_domain_features(processed_signal)
                freq_features = self.extract_frequency_domain_features(processed_signal, fs, fault_freqs)
                tf_features = self.extract_time_frequency_features(processed_signal, fs)

                # 添加位置前缀
                for feature_name, value in time_features.items():
                    result[f'{pos}_{feature_name}'] = value
                for feature_name, value in freq_features.items():
                    result[f'{pos}_{feature_name}'] = value
                for feature_name, value in tf_features.items():
                    result[f'{pos}_{feature_name}'] = value

        return result

    def visualize_signal_analysis(self):
        """可视化1：原始信号波形和频谱分析"""
        if not self.sample_signals:
            print("没有样本信号数据，请先运行数据处理并设置save_sample_signals=True")
            return

        fig = plt.figure(figsize=(20, 12))

        # 颜色映射
        colors = {'B': '#FF6B6B', 'IR': '#4ECDC4', 'OR': '#45B7D1', 'N': '#96CEB4'}

        # 时域信号对比
        ax1 = plt.subplot(2, 3, 1)
        for fault_type, signal_data in self.sample_signals.items():
            signal_array = signal_data['raw_signal'][:2000]  # 显示前2000个点
            fs = signal_data['fs']
            t = np.arange(len(signal_array)) / fs

            ax1.plot(t, signal_array, label=self.fault_types[fault_type],
                     color=colors[fault_type], alpha=0.8, linewidth=1)

        if USE_CHINESE:
            ax1.set_title('原始振动信号时域波形对比', fontsize=14, fontweight='bold')
            ax1.set_xlabel('时间 (s)')
            ax1.set_ylabel('振动幅值')
        else:
            ax1.set_title('Raw Vibration Signal Comparison', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 预处理后信号对比
        ax2 = plt.subplot(2, 3, 2)
        for fault_type, signal_data in self.sample_signals.items():
            signal_array = signal_data['processed_signal'][:2000]
            fs = signal_data['fs']
            t = np.arange(len(signal_array)) / fs

            ax2.plot(t, signal_array, label=self.fault_types[fault_type],
                     color=colors[fault_type], alpha=0.8, linewidth=1)

        if USE_CHINESE:
            ax2.set_title('预处理后信号对比', fontsize=14, fontweight='bold')
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('振动幅值')
        else:
            ax2.set_title('Preprocessed Signal Comparison', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 频谱对比
        ax3 = plt.subplot(2, 3, 3)
        for fault_type, signal_data in self.sample_signals.items():
            signal_array = signal_data['processed_signal']
            fs = signal_data['fs']

            # FFT分析
            n = len(signal_array)
            fft_vals = fft(signal_array)
            freqs = fftfreq(n, 1 / fs)[:n // 2]
            magnitude = np.abs(fft_vals[:n // 2])

            # 只显示0-1000Hz的频谱
            freq_mask = freqs <= 1000
            ax3.semilogy(freqs[freq_mask], magnitude[freq_mask],
                         label=self.fault_types[fault_type],
                         color=colors[fault_type], alpha=0.8, linewidth=1.5)

        if USE_CHINESE:
            ax3.set_title('频谱对比 (0-1000Hz)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('频率 (Hz)')
            ax3.set_ylabel('幅值 (对数尺度)')
        else:
            ax3.set_title('Frequency Spectrum Comparison (0-1000Hz)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Magnitude (Log Scale)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 故障特征频率标注（以B型故障为例）
        ax4 = plt.subplot(2, 3, 4)
        if 'B' in self.sample_signals:
            signal_data = self.sample_signals['B']
            signal_array = signal_data['processed_signal']
            fs = signal_data['fs']
            fault_freqs = signal_data['fault_freqs']

            # FFT分析
            n = len(signal_array)
            fft_vals = fft(signal_array)
            freqs = fftfreq(n, 1 / fs)[:n // 2]
            magnitude = np.abs(fft_vals[:n // 2])

            # 显示0-500Hz的频谱
            freq_mask = freqs <= 500
            ax4.plot(freqs[freq_mask], magnitude[freq_mask],
                     color=colors['B'], linewidth=1.5)

            # 标注故障特征频率
            for freq_name, freq_value in fault_freqs.items():
                if freq_name in ['BPFO', 'BPFI', 'BSF'] and freq_value <= 500:
                    ax4.axvline(x=freq_value, color='red', linestyle='--', alpha=0.7)
                    ax4.text(freq_value, np.max(magnitude[freq_mask]) * 0.8,
                             freq_name, rotation=90, ha='center', va='bottom')

            if USE_CHINESE:
                ax4.set_title('故障特征频率标注 (滚动体故障)', fontsize=14, fontweight='bold')
                ax4.set_xlabel('频率 (Hz)')
                ax4.set_ylabel('幅值')
            else:
                ax4.set_title('Fault Frequency Annotation (Ball Fault)', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Frequency (Hz)')
                ax4.set_ylabel('Magnitude')
            ax4.grid(True, alpha=0.3)

        # 时频分析（以IR故障为例）
        ax5 = plt.subplot(2, 3, 5)
        if 'IR' in self.sample_signals:
            signal_data = self.sample_signals['IR']
            signal_array = signal_data['processed_signal'][:4096]  # 取4096个点进行时频分析
            fs = signal_data['fs']

            from scipy.signal import spectrogram
            f, t, Sxx = spectrogram(signal_array, fs, nperseg=256, noverlap=128)

            # 只显示0-1000Hz
            freq_mask = f <= 1000
            im = ax5.pcolormesh(t, f[freq_mask], 10 * np.log10(Sxx[freq_mask, :] + 1e-12),
                                shading='gouraud', cmap='jet')

            if USE_CHINESE:
                ax5.set_title('时频谱图 (内圈故障)', fontsize=14, fontweight='bold')
                ax5.set_xlabel('时间 (s)')
                ax5.set_ylabel('频率 (Hz)')
            else:
                ax5.set_title('Spectrogram (Inner Race Fault)', fontsize=14, fontweight='bold')
                ax5.set_xlabel('Time (s)')
                ax5.set_ylabel('Frequency (Hz)')
            plt.colorbar(im, ax=ax5, label='Power (dB)')

        # 包络谱分析（以OR故障为例）
        ax6 = plt.subplot(2, 3, 6)
        if 'OR' in self.sample_signals:
            signal_data = self.sample_signals['OR']
            signal_array = signal_data['processed_signal']
            fs = signal_data['fs']

            # 希尔伯特变换求包络
            analytic_signal = scipy_signal.hilbert(signal_array)
            envelope = np.abs(analytic_signal)

            # 包络FFT
            n = len(envelope)
            fft_vals = fft(envelope)
            freqs = fftfreq(n, 1 / fs)[:n // 2]
            magnitude = np.abs(fft_vals[:n // 2])

            # 显示0-200Hz的包络谱
            freq_mask = freqs <= 200
            ax6.plot(freqs[freq_mask], magnitude[freq_mask],
                     color=colors['OR'], linewidth=1.5)

            if USE_CHINESE:
                ax6.set_title('包络谱分析 (外圈故障)', fontsize=14, fontweight='bold')
                ax6.set_xlabel('频率 (Hz)')
                ax6.set_ylabel('包络幅值')
            else:
                ax6.set_title('Envelope Spectrum (Outer Race Fault)', fontsize=14, fontweight='bold')
                ax6.set_xlabel('Frequency (Hz)')
                ax6.set_ylabel('Envelope Magnitude')
            ax6.grid(True, alpha=0.3)

        if USE_CHINESE:
            plt.suptitle('轴承振动信号分析', fontsize=16, fontweight='bold')
        else:
            plt.suptitle('Bearing Vibration Signal Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def visualize_feature_distribution(self, df):
        """可视化2：关键特征分布对比"""
        if df.empty or 'fault_type' not in df.columns:
            print("数据框为空或缺少故障类型列")
            return

        fig = plt.figure(figsize=(20, 12))

        # 颜色映射
        colors = {'B': '#FF6B6B', 'IR': '#4ECDC4', 'OR': '#45B7D1', 'N': '#96CEB4'}

        # 关键特征列表
        key_features = ['DE_rms', 'DE_kurtosis', 'DE_crest_factor', 'DE_peak']

        if USE_CHINESE:
            feature_names = ['RMS值', '峭度', '峰值因子', '峰值']
        else:
            feature_names = ['RMS', 'Kurtosis', 'Crest Factor', 'Peak']

        # 绘制每个特征的分布
        for i, (feature, name) in enumerate(zip(key_features, feature_names)):
            if feature not in df.columns:
                continue

            ax = plt.subplot(2, 3, i + 1)

            # 绘制各故障类型的直方图
            for fault_type in df['fault_type'].unique():
                fault_data = df[df['fault_type'] == fault_type][feature].dropna()
                if len(fault_data) > 0:
                    ax.hist(fault_data, alpha=0.6, bins=20,
                            label=self.fault_types.get(fault_type, fault_type),
                            color=colors.get(fault_type, '#999999'), density=True)

            ax.set_title(f'{name}分布对比' if USE_CHINESE else f'{name} Distribution',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel(name)
            ax.set_ylabel('密度' if USE_CHINESE else 'Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 故障特征频率能量对比
        ax5 = plt.subplot(2, 3, 5)
        freq_features = ['DE_BPFO_amplitude', 'DE_BPFI_amplitude', 'DE_BSF_amplitude']
        available_freq_features = [f for f in freq_features if f in df.columns]

        if available_freq_features:
            fault_types = df['fault_type'].unique()
            x_pos = np.arange(len(fault_types))
            width = 0.8 / len(available_freq_features)

            for i, feature in enumerate(available_freq_features):
                values = []
                for fault in fault_types:
                    fault_data = df[df['fault_type'] == fault][feature].dropna()
                    values.append(fault_data.mean() if len(fault_data) > 0 else 0)

                feature_name = feature.replace('DE_', '').replace('_amplitude', '')
                ax5.bar(x_pos + i * width, values, width,
                        label=feature_name, alpha=0.8)

            ax5.set_title('故障特征频率能量对比' if USE_CHINESE else 'Fault Frequency Energy Comparison',
                          fontsize=14, fontweight='bold')
            ax5.set_xlabel('故障类型' if USE_CHINESE else 'Fault Type')
            ax5.set_ylabel('平均幅值' if USE_CHINESE else 'Average Amplitude')
            ax5.set_xticks(x_pos + width * (len(available_freq_features) - 1) / 2)
            ax5.set_xticklabels([self.fault_types.get(f, f) for f in fault_types])
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 特征相关性分析
        ax6 = plt.subplot(2, 3, 6)
        feature_subset = [f for f in key_features if f in df.columns]
        if len(feature_subset) > 1:
            corr_data = df[feature_subset].corr()

            # 手动绘制相关性矩阵
            im = ax6.imshow(corr_data, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

            # 设置刻度和标签
            ax6.set_xticks(range(len(feature_subset)))
            ax6.set_yticks(range(len(feature_subset)))
            ax6.set_xticklabels([f.replace('DE_', '') for f in feature_subset], rotation=45)
            ax6.set_yticklabels([f.replace('DE_', '') for f in feature_subset])

            # 添加相关系数文本
            for i in range(len(feature_subset)):
                for j in range(len(feature_subset)):
                    text = ax6.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontweight='bold')

            ax6.set_title('特征相关性矩阵' if USE_CHINESE else 'Feature Correlation Matrix',
                          fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax6, shrink=0.8)

        if USE_CHINESE:
            plt.suptitle('关键特征分布与相关性分析', fontsize=16, fontweight='bold')
        else:
            plt.suptitle('Key Feature Distribution and Correlation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def visualize_data_overview(self, source_df, target_df):
        """可视化3：数据集概览和统计分析"""
        fig = plt.figure(figsize=(20, 10))

        # 1. 源域数据故障类型分布
        ax1 = plt.subplot(2, 4, 1)
        if not source_df.empty and 'fault_type' in source_df.columns:
            fault_counts = source_df['fault_type'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

            bars = ax1.bar(range(len(fault_counts)), fault_counts.values,
                           color=colors[:len(fault_counts)], alpha=0.8)

            ax1.set_xticks(range(len(fault_counts)))
            ax1.set_xticklabels([self.fault_types.get(f, f) for f in fault_counts.index], rotation=45)

            if USE_CHINESE:
                ax1.set_title('源域故障类型分布', fontsize=12, fontweight='bold')
                ax1.set_ylabel('样本数量')
            else:
                ax1.set_title('Source Domain Fault Distribution', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Sample Count')

            # 添加数值标签
            for bar, count in zip(bars, fault_counts.values):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         str(count), ha='center', va='bottom', fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # 2. 源域采样频率分布
        ax2 = plt.subplot(2, 4, 2)
        if not source_df.empty and 'sampling_rate' in source_df.columns:
            sampling_counts = source_df['sampling_rate'].value_counts()

            bars = ax2.bar(range(len(sampling_counts)), sampling_counts.values,
                           color='lightblue', alpha=0.8)

            ax2.set_xticks(range(len(sampling_counts)))
            ax2.set_xticklabels([f'{int(sr / 1000)}kHz' for sr in sampling_counts.index])

            if USE_CHINESE:
                ax2.set_title('源域采样频率分布', fontsize=12, fontweight='bold')
                ax2.set_ylabel('样本数量')
            else:
                ax2.set_title('Source Sampling Rate Distribution', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Sample Count')

            for bar, count in zip(bars, sampling_counts.values):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         str(count), ha='center', va='bottom', fontweight='bold')
            ax2.grid(True, alpha=0.3)

        # 3. 目标域数据概览
        ax3 = plt.subplot(2, 4, 3)
        if not target_df.empty:
            target_info = [
                ('文件数量' if USE_CHINESE else 'File Count', len(target_df)),
                ('采样频率' if USE_CHINESE else 'Sampling Rate', '32kHz'),
                ('数据时长' if USE_CHINESE else 'Duration', '8s'),
                ('转速' if USE_CHINESE else 'RPM', '~600')
            ]

            ax3.axis('off')
            y_pos = 0.8
            for info, value in target_info:
                ax3.text(0.1, y_pos, f'{info}: {value}', fontsize=12,
                         transform=ax3.transAxes, fontweight='bold')
                y_pos -= 0.2

            if USE_CHINESE:
                ax3.set_title('目标域数据概览', fontsize=12, fontweight='bold')
            else:
                ax3.set_title('Target Domain Overview', fontsize=12, fontweight='bold')

        # 4. 特征数量统计
        ax4 = plt.subplot(2, 4, 4)
        if not source_df.empty:
            feature_types = []
            feature_counts = []

            # 统计不同类型特征的数量
            for prefix in ['DE_', 'FE_', 'BA_']:
                prefix_features = [col for col in source_df.columns if col.startswith(prefix)]
                if prefix_features:
                    feature_types.append(prefix.replace('_', ''))
                    feature_counts.append(len(prefix_features))

            if feature_types:
                bars = ax4.bar(feature_types, feature_counts,
                               color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)

                if USE_CHINESE:
                    ax4.set_title('各传感器位置特征数量', fontsize=12, fontweight='bold')
                    ax4.set_ylabel('特征数量')
                else:
                    ax4.set_title('Feature Count by Sensor Position', fontsize=12, fontweight='bold')
                    ax4.set_ylabel('Feature Count')

                for bar, count in zip(bars, feature_counts):
                    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                             str(count), ha='center', va='bottom', fontweight='bold')
                ax4.grid(True, alpha=0.3)

        # 5-8. 关键特征的箱线图
        key_features = ['DE_rms', 'DE_kurtosis', 'DE_crest_factor', 'DE_peak']
        if USE_CHINESE:
            feature_titles = ['RMS值分布', '峭度分布', '峰值因子分布', '峰值分布']
        else:
            feature_titles = ['RMS Distribution', 'Kurtosis Distribution', 'Crest Factor Distribution',
                              'Peak Distribution']

        for i, (feature, title) in enumerate(zip(key_features, feature_titles)):
            if feature in source_df.columns:
                ax = plt.subplot(2, 4, 5 + i)

                # 准备箱线图数据
                fault_types = source_df['fault_type'].unique()
                data_for_boxplot = []
                labels = []

                for fault_type in fault_types:
                    fault_data = source_df[source_df['fault_type'] == fault_type][feature].dropna()
                    if len(fault_data) > 0:
                        data_for_boxplot.append(fault_data.values)
                        labels.append(self.fault_types.get(fault_type, fault_type))

                if data_for_boxplot:
                    box_plot = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)

                    # 设置箱线图颜色
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax.set_title(title, fontsize=12, fontweight='bold')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)

        if USE_CHINESE:
            plt.suptitle('数据集概览与特征统计分析', fontsize=16, fontweight='bold')
        else:
            plt.suptitle('Dataset Overview and Feature Statistics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # ========== 新增可视化函数 ==========

    def visualize_3d_feature_space(self, df):
        """可视化4：3D特征空间分布"""
        if df.empty or 'fault_type' not in df.columns:
            print("数据框为空或缺少故障类型列")
            return

        # 选择三个关键特征进行3D可视化
        key_features = ['DE_rms', 'DE_kurtosis', 'DE_crest_factor']
        available_features = [f for f in key_features if f in df.columns]

        if len(available_features) < 3:
            print("可用特征不足3个，无法进行3D可视化")
            return

        fig = plt.figure(figsize=(20, 12))

        # 颜色映射
        colors = {'B': '#FF6B6B', 'IR': '#4ECDC4', 'OR': '#45B7D1', 'N': '#96CEB4'}

        # 3D散点图
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')

        for fault_type in df['fault_type'].unique():
            if fault_type not in colors:
                continue
            fault_data = df[df['fault_type'] == fault_type]

            x = fault_data[available_features[0]].dropna()
            y = fault_data[available_features[1]].dropna()
            z = fault_data[available_features[2]].dropna()

            # 确保三个特征的数据长度一致
            min_len = min(len(x), len(y), len(z))
            if min_len > 0:
                ax1.scatter(x[:min_len], y[:min_len], z[:min_len],
                            c=colors[fault_type], alpha=0.6, s=30,
                            label=self.fault_types.get(fault_type, fault_type))

        ax1.set_xlabel(available_features[0].replace('DE_', ''))
        ax1.set_ylabel(available_features[1].replace('DE_', ''))
        ax1.set_zlabel(available_features[2].replace('DE_', ''))
        ax1.set_title('3D特征空间分布' if USE_CHINESE else '3D Feature Space Distribution',
                      fontsize=14, fontweight='bold')
        ax1.legend()

        # PCA降维可视化
        ax2 = plt.subplot(2, 2, 2)

        # 选择数值特征进行PCA
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_features if col.startswith('DE_') and
                        col not in ['DE_sampling_rate', 'DE_rpm']]

        if len(feature_cols) >= 2:
            # 准备数据
            X = df[feature_cols].fillna(0)
            y = df['fault_type'].fillna('Unknown')

            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # PCA降维
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # 绘制PCA结果
            for fault_type in df['fault_type'].unique():
                if fault_type not in colors:
                    continue
                mask = y == fault_type
                ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=colors[fault_type], alpha=0.6, s=30,
                            label=self.fault_types.get(fault_type, fault_type))

            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax2.set_title('PCA降维可视化' if USE_CHINESE else 'PCA Dimensionality Reduction',
                          fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # t-SNE降维可视化
        ax3 = plt.subplot(2, 2, 3)

        if len(feature_cols) >= 2 and len(df) > 30:  # t-SNE需要足够的样本
            try:
                # 随机采样以加快t-SNE计算
                if len(df) > 1000:
                    sample_df = df.sample(n=1000, random_state=42)
                else:
                    sample_df = df

                X_sample = sample_df[feature_cols].fillna(0)
                y_sample = sample_df['fault_type'].fillna('Unknown')

                # 标准化
                X_sample_scaled = scaler.fit_transform(X_sample)

                # t-SNE降维
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_sample) // 4))
                X_tsne = tsne.fit_transform(X_sample_scaled)

                # 绘制t-SNE结果
                for fault_type in sample_df['fault_type'].unique():
                    if fault_type not in colors:
                        continue
                    mask = y_sample == fault_type
                    ax3.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                                c=colors[fault_type], alpha=0.6, s=30,
                                label=self.fault_types.get(fault_type, fault_type))

                ax3.set_xlabel('t-SNE 1')
                ax3.set_ylabel('t-SNE 2')
                ax3.set_title('t-SNE降维可视化' if USE_CHINESE else 't-SNE Dimensionality Reduction',
                              fontsize=14, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            except Exception as e:
                ax3.text(0.5, 0.5, f't-SNE计算失败: {str(e)}',
                         transform=ax3.transAxes, ha='center', va='center')

        # 特征重要性雷达图
        ax4 = plt.subplot(2, 2, 4, projection='polar')

        if len(feature_cols) >= 3:
            # 计算每个故障类型的特征平均值
            feature_importance = {}
            selected_features = feature_cols[:8]  # 选择前8个特征

            for fault_type in df['fault_type'].unique():
                if fault_type not in colors:
                    continue
                fault_data = df[df['fault_type'] == fault_type][selected_features]
                # 标准化处理
                fault_mean = (fault_data.mean() - df[selected_features].mean()) / df[selected_features].std()
                feature_importance[fault_type] = fault_mean.fillna(0).values

            # 设置角度
            angles = np.linspace(0, 2 * np.pi, len(selected_features), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形

            # 绘制雷达图
            for fault_type, values in feature_importance.items():
                values = np.concatenate([values, [values[0]]])  # 闭合图形
                ax4.plot(angles, values, 'o-', linewidth=2,
                         label=self.fault_types.get(fault_type, fault_type),
                         color=colors[fault_type])
                ax4.fill(angles, values, alpha=0.25, color=colors[fault_type])

            # 设置标签
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels([f.replace('DE_', '') for f in selected_features])
            ax4.set_title('特征重要性雷达图' if USE_CHINESE else 'Feature Importance Radar Chart',
                          fontsize=14, fontweight='bold', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        if USE_CHINESE:
            plt.suptitle('高维特征空间可视化分析', fontsize=16, fontweight='bold')
        else:
            plt.suptitle('High-Dimensional Feature Space Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def visualize_fault_progression(self, df):
        """可视化5：故障严重程度进展分析"""
        if df.empty or 'fault_size' not in df.columns:
            print("数据框为空或缺少故障尺寸列")
            return

        fig = plt.figure(figsize=(20, 12))

        # 颜色映射
        size_colors = {'0007': '#90EE90', '0014': '#FFD700', '0021': '#FFA500', '0028': '#FF6347'}
        fault_colors = {'B': '#FF6B6B', 'IR': '#4ECDC4', 'OR': '#45B7D1'}

        # 1. 故障尺寸vs关键特征趋势图
        ax1 = plt.subplot(2, 3, 1)
        key_feature = 'DE_kurtosis'  # 峭度是故障严重程度的重要指标

        if key_feature in df.columns:
            for fault_type in ['B', 'IR', 'OR']:
                fault_data = df[df['fault_type'] == fault_type]
                if len(fault_data) == 0:
                    continue

                sizes = []
                means = []
                stds = []

                for size in self.fault_sizes:
                    size_data = fault_data[fault_data['fault_size'] == size][key_feature].dropna()
                    if len(size_data) > 0:
                        sizes.append(float(size))
                        means.append(size_data.mean())
                        stds.append(size_data.std())

                if sizes:
                    ax1.errorbar(sizes, means, yerr=stds,
                                 label=self.fault_types.get(fault_type, fault_type),
                                 color=fault_colors[fault_type], marker='o', linewidth=2)

            ax1.set_xlabel('故障尺寸 (inches)' if USE_CHINESE else 'Fault Size (inches)')
            ax1.set_ylabel('峭度均值' if USE_CHINESE else 'Mean Kurtosis')
            ax1.set_title('故障尺寸vs峭度趋势' if USE_CHINESE else 'Fault Size vs Kurtosis Trend',
                          fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. 故障尺寸热力图
        ax2 = plt.subplot(2, 3, 2)

        # 准备热力图数据
        heatmap_data = []
        features_for_heatmap = ['DE_rms', 'DE_kurtosis', 'DE_crest_factor', 'DE_peak']
        available_features = [f for f in features_for_heatmap if f in df.columns]

        if available_features:
            for feature in available_features:
                row_data = []
                for size in self.fault_sizes:
                    size_data = df[df['fault_size'] == size][feature].dropna()
                    row_data.append(size_data.mean() if len(size_data) > 0 else 0)
                heatmap_data.append(row_data)

            heatmap_data = np.array(heatmap_data)

            # 标准化数据
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            heatmap_data_norm = scaler.fit_transform(heatmap_data.T).T

            im = ax2.imshow(heatmap_data_norm, cmap='YlOrRd', aspect='auto')

            # 设置刻度和标签
            ax2.set_xticks(range(len(self.fault_sizes)))
            ax2.set_xticklabels(self.fault_sizes)
            ax2.set_yticks(range(len(available_features)))
            ax2.set_yticklabels([f.replace('DE_', '') for f in available_features])

            # 添加数值标签
            for i in range(len(available_features)):
                for j in range(len(self.fault_sizes)):
                    text = ax2.text(j, i, f'{heatmap_data_norm[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontweight='bold')

            ax2.set_title('故障尺寸特征热力图' if USE_CHINESE else 'Fault Size Feature Heatmap',
                          fontsize=14, fontweight='bold')
            ax2.set_xlabel('故障尺寸' if USE_CHINESE else 'Fault Size')
            plt.colorbar(im, ax=ax2, shrink=0.8)

        # 3. 载荷vs故障严重程度
        ax3 = plt.subplot(2, 3, 3)

        if 'load_level' in df.columns and 'DE_rms' in df.columns:
            load_data = {}
            for load in self.load_levels:
                load_samples = df[df['load_level'] == load]['DE_rms'].dropna()
                if len(load_samples) > 0:
                    load_data[load] = load_samples.values

            if load_data:
                ax3.boxplot(load_data.values(), labels=load_data.keys(), patch_artist=True)
                ax3.set_xlabel('载荷等级' if USE_CHINESE else 'Load Level')
                ax3.set_ylabel('RMS值' if USE_CHINESE else 'RMS Value')
                ax3.set_title('载荷等级vs振动强度' if USE_CHINESE else 'Load Level vs Vibration Intensity',
                              fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)

        # 4. 故障位置对比（仅针对OR故障）
        ax4 = plt.subplot(2, 3, 4)

        if 'or_position' in df.columns and 'DE_kurtosis' in df.columns:
            or_data = df[df['fault_type'] == 'OR']
            if len(or_data) > 0:
                position_data = []
                position_labels = []

                for position in self.or_positions:
                    pos_samples = or_data[or_data['or_position'] == position]['DE_kurtosis'].dropna()
                    if len(pos_samples) > 0:
                        position_data.append(pos_samples.values)
                        position_labels.append(position)

                if position_data:
                    box_plot = ax4.boxplot(position_data, labels=position_labels, patch_artist=True)

                    # 设置颜色
                    colors_pos = ['#FFB6C1', '#87CEEB', '#98FB98']
                    for patch, color in zip(box_plot['boxes'], colors_pos[:len(box_plot['boxes'])]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax4.set_xlabel('OR故障位置' if USE_CHINESE else 'OR Fault Position')
                    ax4.set_ylabel('峭度' if USE_CHINESE else 'Kurtosis')
                    ax4.set_title('OR故障位置对比' if USE_CHINESE else 'OR Fault Position Comparison',
                                  fontsize=14, fontweight='bold')
                    ax4.tick_params(axis='x', rotation=45)
                    ax4.grid(True, alpha=0.3)

        # 5. 瀑布图：不同故障尺寸的频谱对比
        ax5 = plt.subplot(2, 3, 5)

        if self.sample_signals and 'B' in self.sample_signals:
            # 模拟不同尺寸的频谱（这里使用示例数据）
            signal_data = self.sample_signals['B']
            signal_array = signal_data['processed_signal']
            fs = signal_data['fs']

            # FFT分析
            n = len(signal_array)
            fft_vals = fft(signal_array)
            freqs = fftfreq(n, 1 / fs)[:n // 2]
            magnitude = np.abs(fft_vals[:n // 2])

            # 模拟不同尺寸的影响（实际应用中应使用真实的不同尺寸数据）
            freq_mask = freqs <= 500
            freqs_plot = freqs[freq_mask]

            for i, size in enumerate(self.fault_sizes):
                # 模拟不同尺寸的幅值变化
                magnitude_scaled = magnitude[freq_mask] * (1 + i * 0.5)
                ax5.plot(freqs_plot, magnitude_scaled + i * np.max(magnitude_scaled) * 0.3,
                         label=f'{size} inches', color=list(size_colors.values())[i], linewidth=1.5)

            ax5.set_xlabel('频率 (Hz)' if USE_CHINESE else 'Frequency (Hz)')
            ax5.set_ylabel('幅值 (偏移显示)' if USE_CHINESE else 'Magnitude (Offset)')
            ax5.set_title('不同故障尺寸频谱瀑布图' if USE_CHINESE else 'Fault Size Frequency Waterfall',
                          fontsize=14, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. 故障演化趋势预测
        ax6 = plt.subplot(2, 3, 6)

        if 'DE_rms' in df.columns and 'fault_size' in df.columns:
            # 准备数据
            size_numeric = []
            rms_values = []

            for fault_type in ['B', 'IR', 'OR']:
                fault_data = df[df['fault_type'] == fault_type]
                for size in self.fault_sizes:
                    size_data = fault_data[fault_data['fault_size'] == size]['DE_rms'].dropna()
                    if len(size_data) > 0:
                        size_numeric.extend([float(size)] * len(size_data))
                        rms_values.extend(size_data.values)

            if size_numeric and rms_values:
                # 散点图
                ax6.scatter(size_numeric, rms_values, alpha=0.6, s=30, color='gray')

                # 拟合趋势线
                z = np.polyfit(size_numeric, rms_values, 2)  # 二次多项式拟合
                p = np.poly1d(z)
                x_trend = np.linspace(min(size_numeric), max(size_numeric), 100)
                ax6.plot(x_trend, p(x_trend), 'r--', linewidth=2, label='趋势线')

                ax6.set_xlabel('故障尺寸 (inches)' if USE_CHINESE else 'Fault Size (inches)')
                ax6.set_ylabel('RMS值' if USE_CHINESE else 'RMS Value')
                ax6.set_title('故障演化趋势' if USE_CHINESE else 'Fault Evolution Trend',
                              fontsize=14, fontweight='bold')
                ax6.legend()
                ax6.grid(True, alpha=0.3)

        if USE_CHINESE:
            plt.suptitle('故障严重程度进展分析', fontsize=16, fontweight='bold')
        else:
            plt.suptitle('Fault Severity Progression Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def visualize_frequency_analysis_comparison(self):
        """可视化6：深度频域分析对比"""
        if not self.sample_signals:
            print("没有样本信号数据，请先运行数据处理并设置save_sample_signals=True")
            return

        fig = plt.figure(figsize=(20, 16))

        # 颜色映射
        colors = {'B': '#FF6B6B', 'IR': '#4ECDC4', 'OR': '#45B7D1', 'N': '#96CEB4'}

        # 1. 功率谱密度对比
        ax1 = plt.subplot(3, 3, 1)

        for fault_type, signal_data in self.sample_signals.items():
            signal_array = signal_data['processed_signal']
            fs = signal_data['fs']

            # 使用Welch方法计算PSD
            from scipy.signal import welch
            freqs, psd = welch(signal_array, fs, nperseg=1024)

            # 只显示0-1000Hz
            freq_mask = freqs <= 1000
            ax1.semilogy(freqs[freq_mask], psd[freq_mask],
                         label=self.fault_types[fault_type],
                         color=colors[fault_type], linewidth=2)

        ax1.set_xlabel('频率 (Hz)' if USE_CHINESE else 'Frequency (Hz)')
        ax1.set_ylabel('功率谱密度' if USE_CHINESE else 'Power Spectral Density')
        ax1.set_title('功率谱密度对比' if USE_CHINESE else 'Power Spectral Density Comparison',
                      fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 包络谱对比
        ax2 = plt.subplot(3, 3, 2)

        for fault_type, signal_data in self.sample_signals.items():
            signal_array = signal_data['processed_signal']
            fs = signal_data['fs']

            # 希尔伯特变换求包络
            analytic_signal = scipy_signal.hilbert(signal_array)
            envelope = np.abs(analytic_signal)

            # 包络FFT
            n = len(envelope)
            fft_vals = fft(envelope)
            freqs = fftfreq(n, 1 / fs)[:n // 2]
            magnitude = np.abs(fft_vals[:n // 2])

            # 只显示0-300Hz的包络谱
            freq_mask = freqs <= 300
            ax2.plot(freqs[freq_mask], magnitude[freq_mask],
                     label=self.fault_types[fault_type],
                     color=colors[fault_type], linewidth=2)

        ax2.set_xlabel('频率 (Hz)' if USE_CHINESE else 'Frequency (Hz)')
        ax2.set_ylabel('包络幅值' if USE_CHINESE else 'Envelope Magnitude')
        ax2.set_title('包络谱对比' if USE_CHINESE else 'Envelope Spectrum Comparison',
                      fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 倒频谱分析
        ax3 = plt.subplot(3, 3, 3)

        for fault_type, signal_data in self.sample_signals.items():
            signal_array = signal_data['processed_signal']
            fs = signal_data['fs']

            # 计算倒频谱
            fft_vals = fft(signal_array)
            magnitude = np.abs(fft_vals)
            log_magnitude = np.log(magnitude + 1e-12)  # 避免log(0)
            cepstrum = np.real(np.fft.ifft(log_magnitude))

            # 倒频谱的时间轴（quefrency）
            quefrency = np.arange(len(cepstrum)) / fs

            # 只显示前半部分
            half_len = len(cepstrum) // 2
            ax3.plot(quefrency[:half_len], cepstrum[:half_len],
                     label=self.fault_types[fault_type],
                     color=colors[fault_type], linewidth=1.5)

        ax3.set_xlabel('Quefrency (s)' if USE_CHINESE else 'Quefrency (s)')
        ax3.set_ylabel('Cepstrum')
        ax3.set_title('倒频谱分析' if USE_CHINESE else 'Cepstrum Analysis',
                      fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4-6. 不同故障类型的时频图
        fault_types_to_plot = ['B', 'IR', 'OR']
        for i, fault_type in enumerate(fault_types_to_plot):
            if fault_type not in self.sample_signals:
                continue

            ax = plt.subplot(3, 3, 4 + i)
            signal_data = self.sample_signals[fault_type]
            signal_array = signal_data['processed_signal'][:4096]  # 取4096个点
            fs = signal_data['fs']

            from scipy.signal import spectrogram
            f, t, Sxx = spectrogram(signal_array, fs, nperseg=256, noverlap=192)

            # 只显示0-1500Hz
            freq_mask = f <= 1500
            im = ax.pcolormesh(t, f[freq_mask], 10 * np.log10(Sxx[freq_mask, :] + 1e-12),
                               shading='gouraud', cmap='jet')

            ax.set_xlabel('时间 (s)' if USE_CHINESE else 'Time (s)')
            ax.set_ylabel('频率 (Hz)' if USE_CHINESE else 'Frequency (Hz)')
            ax.set_title(f'{self.fault_types[fault_type]}时频图' if USE_CHINESE
                         else f'{self.fault_types[fault_type]} Spectrogram',
                         fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)

        # 7. 频率切片对比（特定频率范围的能量）
        ax7 = plt.subplot(3, 3, 7)

        # 定义频率段
        freq_bands = [(0, 100), (100, 300), (300, 600), (600, 1000)]
        band_names = ['0-100Hz', '100-300Hz', '300-600Hz', '600-1000Hz']

        band_energies = {fault_type: [] for fault_type in self.sample_signals.keys()}

        for fault_type, signal_data in self.sample_signals.items():
            signal_array = signal_data['processed_signal']
            fs = signal_data['fs']

            # FFT分析
            n = len(signal_array)
            fft_vals = fft(signal_array)
            freqs = fftfreq(n, 1 / fs)[:n // 2]
            magnitude = np.abs(fft_vals[:n // 2])

            # 计算各频段能量
            for f_low, f_high in freq_bands:
                band_mask = (freqs >= f_low) & (freqs < f_high)
                band_energy = np.sum(magnitude[band_mask])
                band_energies[fault_type].append(band_energy)

        # 绘制柱状图
        x = np.arange(len(band_names))
        width = 0.8 / len(self.sample_signals)

        for i, (fault_type, energies) in enumerate(band_energies.items()):
            ax7.bar(x + i * width, energies, width,
                    label=self.fault_types[fault_type],
                    color=colors[fault_type], alpha=0.8)

        ax7.set_xlabel('频率段' if USE_CHINESE else 'Frequency Band')
        ax7.set_ylabel('能量' if USE_CHINESE else 'Energy')
        ax7.set_title('频段能量分布' if USE_CHINESE else 'Frequency Band Energy Distribution',
                      fontsize=12, fontweight='bold')
        ax7.set_xticks(x + width * (len(self.sample_signals) - 1) / 2)
        ax7.set_xticklabels(band_names)
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. 相位谱对比
        ax8 = plt.subplot(3, 3, 8)

        for fault_type, signal_data in self.sample_signals.items():
            signal_array = signal_data['processed_signal']
            fs = signal_data['fs']

            # FFT分析
            n = len(signal_array)
            fft_vals = fft(signal_array)
            freqs = fftfreq(n, 1 / fs)[:n // 2]
            phase = np.angle(fft_vals[:n // 2])

            # 只显示0-500Hz的相位谱
            freq_mask = freqs <= 500
            ax8.plot(freqs[freq_mask], phase[freq_mask],
                     label=self.fault_types[fault_type],
                     color=colors[fault_type], linewidth=1, alpha=0.8)

        ax8.set_xlabel('频率 (Hz)' if USE_CHINESE else 'Frequency (Hz)')
        ax8.set_ylabel('相位 (rad)' if USE_CHINESE else 'Phase (rad)')
        ax8.set_title('相位谱对比' if USE_CHINESE else 'Phase Spectrum Comparison',
                      fontsize=12, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. 频谱峰值检测
        ax9 = plt.subplot(3, 3, 9)

        if 'B' in self.sample_signals:
            signal_data = self.sample_signals['B']
            signal_array = signal_data['processed_signal']
            fs = signal_data['fs']
            fault_freqs = signal_data['fault_freqs']

            # FFT分析
            n = len(signal_array)
            fft_vals = fft(signal_array)
            freqs = fftfreq(n, 1 / fs)[:n // 2]
            magnitude = np.abs(fft_vals[:n // 2])

            # 频谱峰值检测
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(magnitude, height=np.max(magnitude) * 0.1, distance=20)

            # 显示0-800Hz的频谱和峰值
            freq_mask = freqs <= 800
            ax9.plot(freqs[freq_mask], magnitude[freq_mask],
                     color=colors['B'], linewidth=1.5, label='频谱')

            # 标注检测到的峰值
            peak_mask = peaks < len(freqs[freq_mask])
            if np.any(peak_mask):
                valid_peaks = peaks[peak_mask]
                ax9.plot(freqs[valid_peaks], magnitude[valid_peaks],
                         'ro', markersize=8, label='检测峰值')

            # 标注理论故障频率
            for freq_name, freq_value in fault_freqs.items():
                if freq_name in ['BPFO', 'BPFI', 'BSF'] and freq_value <= 800:
                    ax9.axvline(x=freq_value, color='green', linestyle='--', alpha=0.7, linewidth=2)
                    ax9.text(freq_value, np.max(magnitude[freq_mask]) * 0.9,
                             freq_name, rotation=90, ha='center', va='bottom', color='green')

            ax9.set_xlabel('频率 (Hz)' if USE_CHINESE else 'Frequency (Hz)')
            ax9.set_ylabel('幅值' if USE_CHINESE else 'Magnitude')
            ax9.set_title('频谱峰值检测 (滚动体故障)' if USE_CHINESE
                          else 'Spectrum Peak Detection (Ball Fault)',
                          fontsize=12, fontweight='bold')
            ax9.legend()
            ax9.grid(True, alpha=0.3)

        if USE_CHINESE:
            plt.suptitle('深度频域分析对比', fontsize=16, fontweight='bold')
        else:
            plt.suptitle('Advanced Frequency Domain Analysis Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def visualize_comprehensive_dashboard(self, df):
        """可视化7：综合仪表盘"""
        if df.empty:
            print("数据框为空")
            return

        fig = plt.figure(figsize=(24, 16))

        # 创建网格布局
        gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.3)

        # 颜色映射
        colors = {'B': '#FF6B6B', 'IR': '#4ECDC4', 'OR': '#45B7D1', 'N': '#96CEB4'}

        # 1. 总体数据统计仪表
        ax1 = fig.add_subplot(gs[0, 0:2])

        # 创建仪表盘样式的统计信息
        stats = [
            ('总样本数', len(df)),
            ('故障类型', len(df['fault_type'].unique()) if 'fault_type' in df.columns else 0),
            ('特征维度', len([col for col in df.columns if col.startswith('DE_')])),
            ('采样频率', f"{df['sampling_rate'].mode()[0] / 1000:.0f}kHz" if 'sampling_rate' in df.columns else 'N/A')
        ]

        ax1.axis('off')
        for i, (label, value) in enumerate(stats):
            y_pos = 0.8 - i * 0.2
            ax1.text(0.1, y_pos, f'{label}:', fontsize=14, fontweight='bold',
                     transform=ax1.transAxes)
            ax1.text(0.6, y_pos, str(value), fontsize=14, color='blue',
                     fontweight='bold', transform=ax1.transAxes)

        ax1.set_title('数据统计概览' if USE_CHINESE else 'Data Statistics Overview',
                      fontsize=14, fontweight='bold')

        # 2. 故障类型饼图
        ax2 = fig.add_subplot(gs[0, 2:4])

        if 'fault_type' in df.columns:
            fault_counts = df['fault_type'].value_counts()
            labels = [self.fault_types.get(f, f) for f in fault_counts.index]
            colors_pie = [colors.get(f, '#999999') for f in fault_counts.index]

            wedges, texts, autotexts = ax2.pie(fault_counts.values, labels=labels,
                                               colors=colors_pie, autopct='%1.1f%%',
                                               startangle=90, textprops={'fontsize': 10})

            ax2.set_title('故障类型分布' if USE_CHINESE else 'Fault Type Distribution',
                          fontsize=14, fontweight='bold')

        # 3. 关键指标雷达图
        ax3 = fig.add_subplot(gs[0, 4:6], projection='polar')

        if 'fault_type' in df.columns:
            key_features = ['DE_rms', 'DE_kurtosis', 'DE_crest_factor', 'DE_peak', 'DE_std']
            available_features = [f for f in key_features if f in df.columns]

            if len(available_features) >= 3:
                # 计算每个故障类型的标准化特征值
                angles = np.linspace(0, 2 * np.pi, len(available_features), endpoint=False).tolist()
                angles += angles[:1]

                for fault_type in df['fault_type'].unique():
                    if fault_type not in colors:
                        continue

                    fault_data = df[df['fault_type'] == fault_type][available_features]
                    if len(fault_data) == 0:
                        continue

                    # 计算标准化均值
                    fault_means = fault_data.mean()
                    overall_means = df[available_features].mean()
                    overall_stds = df[available_features].std()

                    normalized_values = ((fault_means - overall_means) / overall_stds).fillna(0).values
                    normalized_values = np.concatenate([normalized_values, [normalized_values[0]]])

                    ax3.plot(angles, normalized_values, 'o-', linewidth=2,
                             label=self.fault_types.get(fault_type, fault_type),
                             color=colors[fault_type])
                    ax3.fill(angles, normalized_values, alpha=0.25, color=colors[fault_type])

                ax3.set_xticks(angles[:-1])
                ax3.set_xticklabels([f.replace('DE_', '') for f in available_features])
                ax3.set_title('特征对比雷达图' if USE_CHINESE else 'Feature Comparison Radar',
                              fontsize=14, fontweight='bold', pad=20)
                ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        # 4. 时域特征趋势图
        ax4 = fig.add_subplot(gs[1, 0:3])

        time_features = ['DE_rms', 'DE_peak', 'DE_std']
        available_time_features = [f for f in time_features if f in df.columns]

        if available_time_features and 'fault_type' in df.columns:
            x_pos = np.arange(len(available_time_features))
            width = 0.8 / len(df['fault_type'].unique())

            for i, fault_type in enumerate(df['fault_type'].unique()):
                if fault_type not in colors:
                    continue

                fault_data = df[df['fault_type'] == fault_type]
                means = [fault_data[feature].mean() if feature in fault_data.columns
                         else 0 for feature in available_time_features]

                ax4.bar(x_pos + i * width, means, width,
                        label=self.fault_types.get(fault_type, fault_type),
                        color=colors[fault_type], alpha=0.8)

            ax4.set_xlabel('时域特征' if USE_CHINESE else 'Time Domain Features')
            ax4.set_ylabel('平均值' if USE_CHINESE else 'Average Value')
            ax4.set_title('时域特征对比' if USE_CHINESE else 'Time Domain Feature Comparison',
                          fontsize=14, fontweight='bold')
            ax4.set_xticks(x_pos + width * (len(df['fault_type'].unique()) - 1) / 2)
            ax4.set_xticklabels([f.replace('DE_', '') for f in available_time_features])
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. 频域特征对比
        ax5 = fig.add_subplot(gs[1, 3:6])

        freq_features = ['DE_BPFO_amplitude', 'DE_BPFI_amplitude', 'DE_BSF_amplitude']
        available_freq_features = [f for f in freq_features if f in df.columns]

        if available_freq_features and 'fault_type' in df.columns:
            fault_types = list(df['fault_type'].unique())

            # 创建热力图数据
            heatmap_data = []
            for feature in available_freq_features:
                row = []
                for fault_type in fault_types:
                    fault_data = df[df['fault_type'] == fault_type][feature].dropna()
                    row.append(fault_data.mean() if len(fault_data) > 0 else 0)
                heatmap_data.append(row)

            heatmap_data = np.array(heatmap_data)

            # 标准化
            if heatmap_data.size > 0:
                heatmap_data_norm = (heatmap_data - heatmap_data.min()) / (
                            heatmap_data.max() - heatmap_data.min() + 1e-8)

                im = ax5.imshow(heatmap_data_norm, cmap='Reds', aspect='auto')

                ax5.set_xticks(range(len(fault_types)))
                ax5.set_xticklabels([self.fault_types.get(f, f) for f in fault_types])
                ax5.set_yticks(range(len(available_freq_features)))
                ax5.set_yticklabels([f.replace('DE_', '').replace('_amplitude', '')
                                     for f in available_freq_features])

                # 添加数值标签
                for i in range(len(available_freq_features)):
                    for j in range(len(fault_types)):
                        text = ax5.text(j, i, f'{heatmap_data_norm[i, j]:.2f}',
                                        ha="center", va="center", color="black", fontweight='bold')

                ax5.set_title('故障频率能量热力图' if USE_CHINESE else 'Fault Frequency Energy Heatmap',
                              fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=ax5, shrink=0.8)

        # 6. 样本分布散点图矩阵
        ax6 = fig.add_subplot(gs[2, 0:3])

        scatter_features = ['DE_rms', 'DE_kurtosis']
        if all(f in df.columns for f in scatter_features) and 'fault_type' in df.columns:
            for fault_type in df['fault_type'].unique():
                if fault_type not in colors:
                    continue

                fault_data = df[df['fault_type'] == fault_type]
                x_data = fault_data[scatter_features[0]].dropna()
                y_data = fault_data[scatter_features[1]].dropna()

                min_len = min(len(x_data), len(y_data))
                if min_len > 0:
                    ax6.scatter(x_data[:min_len], y_data[:min_len],
                                c=colors[fault_type], alpha=0.6, s=50,
                                label=self.fault_types.get(fault_type, fault_type))

            ax6.set_xlabel(scatter_features[0].replace('DE_', ''))
            ax6.set_ylabel(scatter_features[1].replace('DE_', ''))
            ax6.set_title('特征散点分布' if USE_CHINESE else 'Feature Scatter Distribution',
                          fontsize=14, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. 特征重要性排序
        ax7 = fig.add_subplot(gs[2, 3:6])

        if 'fault_type' in df.columns:
            feature_cols = [col for col in df.columns if col.startswith('DE_') and
                            col not in ['DE_sampling_rate', 'DE_rpm']]

            if len(feature_cols) > 0:
                # 计算特征的方差（作为重要性指标）
                feature_importance = {}
                for feature in feature_cols[:10]:  # 只取前10个特征
                    if feature in df.columns:
                        feature_var = df[feature].var()
                        feature_importance[feature.replace('DE_', '')] = feature_var

                # 排序
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

                if sorted_features:
                    features, importances = zip(*sorted_features)
                    y_pos = np.arange(len(features))

                    bars = ax7.barh(y_pos, importances, color='skyblue', alpha=0.8)
                    ax7.set_yticks(y_pos)
                    ax7.set_yticklabels(features)
                    ax7.set_xlabel('方差 (重要性指标)' if USE_CHINESE else 'Variance (Importance)')
                    ax7.set_title('特征重要性排序' if USE_CHINESE else 'Feature Importance Ranking',
                                  fontsize=14, fontweight='bold')
                    ax7.grid(True, alpha=0.3)

                    # 添加数值标签
                    for bar, importance in zip(bars, importances):
                        ax7.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height() / 2,
                                 f'{importance:.2e}', ha='left', va='center', fontsize=8)

        # 8. 数据质量指标
        ax8 = fig.add_subplot(gs[3, 0:2])

        quality_metrics = []
        if not df.empty:
            # 计算数据质量指标
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            completeness = (total_cells - missing_cells) / total_cells * 100

            # 特征一致性（相关性）
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                avg_correlation = corr_matrix.abs().mean().mean()
            else:
                avg_correlation = 0

            quality_metrics = [
                ('数据完整性', f'{completeness:.1f}%'),
                ('特征相关性', f'{avg_correlation:.3f}'),
                ('样本平衡度', f'{df["fault_type"].value_counts().std():.1f}' if 'fault_type' in df.columns else 'N/A'),
                ('特征维度', f'{len([col for col in df.columns if col.startswith("DE_")])}')
            ]

        ax8.axis('off')
        for i, (metric, value) in enumerate(quality_metrics):
            y_pos = 0.8 - i * 0.2
            ax8.text(0.1, y_pos, f'{metric}:', fontsize=12, fontweight='bold',
                     transform=ax8.transAxes)
            ax8.text(0.7, y_pos, value, fontsize=12, color='green',
                     fontweight='bold', transform=ax8.transAxes)

        ax8.set_title('数据质量评估' if USE_CHINESE else 'Data Quality Assessment',
                      fontsize=14, fontweight='bold')

        # 9. 实时监控仪表盘（模拟）- 使用极坐标轴
        ax9 = fig.add_subplot(gs[3, 2:4], projection='polar')

        # 创建模拟的实时监控数据
        if 'DE_rms' in df.columns and 'fault_type' in df.columns:
            # 计算正常状态的阈值
            normal_data = df[df['fault_type'] == 'N']['DE_rms'].dropna() if 'N' in df['fault_type'].values else df[
                'DE_rms'].dropna()
            if len(normal_data) > 0:
                threshold = normal_data.mean() + 2 * normal_data.std()

                # 模拟当前状态
                current_values = []
                status_colors = []

                for fault_type in ['N', 'B', 'IR', 'OR']:
                    if fault_type in df['fault_type'].values:
                        current_value = df[df['fault_type'] == fault_type]['DE_rms'].mean()
                        current_values.append(current_value)
                        status_colors.append('green' if current_value < threshold else 'red')

                # 绘制仪表盘
                angles = np.linspace(0, 2 * np.pi, len(current_values), endpoint=False)
                ax9.bar(angles, current_values, width=0.5, bottom=0.0,
                        color=status_colors, alpha=0.7)

                ax9.set_theta_zero_location('N')
                ax9.set_theta_direction(-1)
                ax9.set_rlabel_position(0)
                ax9.set_title('状态监控仪表盘' if USE_CHINESE else 'Status Monitoring Dashboard',
                              fontsize=14, fontweight='bold', pad=20)

                # 添加阈值线
                threshold_line = [threshold] * len(current_values)
                ax9.plot(angles, threshold_line, 'r--', linewidth=2, label='报警阈值')
                ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        # 10. 趋势预测图
        ax10 = fig.add_subplot(gs[3, 4:6])

        if 'DE_kurtosis' in df.columns and 'fault_type' in df.columns:
            # 模拟时间序列数据
            time_points = np.arange(1, 25)  # 24小时

            for fault_type in df['fault_type'].unique():
                if fault_type not in colors:
                    continue

                base_value = df[df['fault_type'] == fault_type]['DE_kurtosis'].mean()
                if not np.isnan(base_value):
                    # 添加一些随机变化模拟趋势
                    trend = base_value + np.random.normal(0, base_value * 0.1, len(time_points))
                    ax10.plot(time_points, trend, 'o-',
                              label=self.fault_types.get(fault_type, fault_type),
                              color=colors[fault_type], linewidth=2, markersize=4)

            ax10.set_xlabel('时间 (小时)' if USE_CHINESE else 'Time (Hours)')
            ax10.set_ylabel('峭度值' if USE_CHINESE else 'Kurtosis Value')
            ax10.set_title('24小时趋势预测' if USE_CHINESE else '24-Hour Trend Prediction',
                           fontsize=14, fontweight='bold')
            ax10.legend()
            ax10.grid(True, alpha=0.3)

        if USE_CHINESE:
            plt.suptitle('轴承故障诊断综合仪表盘', fontsize=18, fontweight='bold')
        else:
            plt.suptitle('Bearing Fault Diagnosis Comprehensive Dashboard', fontsize=18, fontweight='bold')
        plt.show()

    # ========== 原有函数保持不变 ==========

    def process_source_data(self, sensor_types=['12kHz_DE_data', '12kHz_FE_data'],
                            sample_limit=None, save_sample_signals=True):
        """
        处理源域数据（增强版，支持可视化）

        Args:
            sensor_types: 要处理的传感器类型列表
            sample_limit: 每个类别最大样本数量限制
            save_sample_signals: 是否保存样本信号用于可视化

        Returns:
            pd.DataFrame: 处理后的特征数据框
        """
        all_features = []

        for sensor_type in sensor_types:
            sensor_path = self.source_data_path / sensor_type

            if not sensor_path.exists():
                print(f"传感器路径不存在: {sensor_path}")
                continue

            print(f"处理传感器类型: {sensor_type}")

            # 遍历故障类型文件夹
            for fault_folder in sensor_path.iterdir():
                if not fault_folder.is_dir():
                    continue

                if fault_folder.name in ['B', 'IR']:
                    # B和IR类型直接包含尺寸文件夹
                    print(f"  处理故障类型: {fault_folder.name}")

                    for size_folder in fault_folder.iterdir():
                        if size_folder.is_dir() and size_folder.name in self.fault_sizes:
                            print(f"    处理故障尺寸: {size_folder.name}")

                            # 处理该尺寸下的所有.mat文件
                            mat_files = list(size_folder.glob('*.mat'))

                            # 如果设置了样本限制，随机采样
                            if sample_limit and len(mat_files) > sample_limit:
                                mat_files = np.random.choice(mat_files, sample_limit, replace=False)

                            for mat_file in mat_files:
                                features = self.process_single_file(mat_file, save_sample_signals)
                                if features:
                                    all_features.append(features)

                elif fault_folder.name == 'OR':
                    # OR类型包含位置子文件夹
                    print(f"  处理故障类型: {fault_folder.name}")

                    for position_folder in fault_folder.iterdir():
                        if position_folder.is_dir() and position_folder.name in self.or_positions:
                            print(f"    处理OR位置: {position_folder.name}")

                            for size_folder in position_folder.iterdir():
                                if size_folder.is_dir() and size_folder.name in self.fault_sizes:
                                    print(f"      处理故障尺寸: {size_folder.name}")

                                    # 处理该尺寸下的所有.mat文件
                                    mat_files = list(size_folder.glob('*.mat'))

                                    # 如果设置了样本限制，随机采样
                                    if sample_limit and len(mat_files) > sample_limit:
                                        mat_files = np.random.choice(mat_files, sample_limit, replace=False)

                                    for mat_file in mat_files:
                                        features = self.process_single_file(mat_file, save_sample_signals)
                                        if features:
                                            all_features.append(features)

        # 处理正常数据
        normal_path = self.source_data_path / '48kHz_Normal_data'
        if normal_path.exists():
            print("处理正常状态数据")
            mat_files = list(normal_path.glob('*.mat'))

            if sample_limit and len(mat_files) > sample_limit:
                mat_files = np.random.choice(mat_files, sample_limit, replace=False)

            for mat_file in mat_files:
                features = self.process_single_file(mat_file, save_sample_signals)
                if features:
                    all_features.append(features)

        # 转换为DataFrame
        if len(all_features) == 0:
            print("警告: 没有成功处理任何文件!")
            return pd.DataFrame()

        df = pd.DataFrame(all_features)

        # 编码标签
        if 'fault_type' in df.columns and not df['fault_type'].isna().all():
            le = LabelEncoder()
            df['fault_label'] = le.fit_transform(df['fault_type'].fillna('Unknown'))

        print(f"总共处理了 {len(df)} 个样本")
        if 'fault_type' in df.columns:
            print(f"故障类型分布:\n{df['fault_type'].value_counts()}")

        return df

    def process_target_data(self):
        """
        处理目标域数据

        Returns:
            pd.DataFrame: 处理后的目标域数据
        """
        all_features = []

        if not self.target_data_path.exists():
            print(f"目标域路径不存在: {self.target_data_path}")
            return pd.DataFrame()

        target_files = list(self.target_data_path.glob('*.mat'))

        print(f"处理目标域数据，共 {len(target_files)} 个文件")

        for mat_file in target_files:
            # 目标域可能无标签，放行到特征提取
            features = self.process_single_file(mat_file, save_sample_signals=False, require_fault_type=False)
            if features:
                features['file_id'] = mat_file.stem  # 使用文件名作为ID
                features['sampling_rate'] = 32000  # 目标域采样频率32kHz
                all_features.append(features)

        if len(all_features) == 0:
            print("警告: 没有成功处理任何目标域文件!")
            return pd.DataFrame()

        df = pd.DataFrame(all_features)
        print(f"目标域数据处理完成，共 {len(df)} 个样本")

        return df

    def save_processed_data(self, source_df, target_df, output_dir):
        """
        保存处理后的数据

        Args:
            source_df: 源域数据框
            target_df: 目标域数据框
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 保存源域数据
        if not source_df.empty:
            source_df.to_csv(output_path / 'source_domain_features_simplified.csv', index=False)
            print(f"源域数据已保存，形状: {source_df.shape}")
        else:
            print("警告: 源域数据为空，未保存")

        # 保存目标域数据
        if not target_df.empty:
            target_df.to_csv(output_path / 'target_domain_features_simplified.csv', index=False)
            print(f"目标域数据已保存，形状: {target_df.shape}")
        else:
            print("警告: 目标域数据为空，未保存")

        # 保存特征名称
        if not source_df.empty:
            feature_columns = [col for col in source_df.columns
                               if col not in ['file_path', 'fault_type', 'fault_size',
                                              'load_level', 'sensor_type', 'rpm', 'fault_label',
                                              'or_position', 'sampling_rate', 'file_id']]

            with open(output_path / 'feature_names_simplified.txt', 'w') as f:
                for feature in feature_columns:
                    f.write(f"{feature}\n")

            print(f"简化后特征维度: {len(feature_columns)}")

        print(f"数据已保存到: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 初始化增强版数据处理器
    data_root = r".."
    processor = SimplifiedBearingDataProcessor(data_root)

    # 处理源域数据（启用样本信号保存）
    print("开始处理源域数据...")
    source_data = processor.process_source_data(
        sensor_types=['12kHz_DE_data', '12kHz_FE_data', '48kHz_DE_data'],
        sample_limit=None,  # 处理所有可用数据
        save_sample_signals=True  # 保存样本信号用于可视化
    )

    # 处理目标域数据
    print("\n开始处理目标域数据...")
    target_data = processor.process_target_data()

    # 全面的可视化分析
    print("\n生成全面的可视化分析...")

    # 原有可视化
    processor.visualize_signal_analysis()  # 可视化1：信号分析
    processor.visualize_feature_distribution(source_data)  # 可视化2：特征分布
    processor.visualize_data_overview(source_data, target_data)  # 可视化3：数据概览

    # 新增可视化
    processor.visualize_3d_feature_space(source_data)  # 可视化4：3D特征空间
    processor.visualize_fault_progression(source_data)  # 可视化5：故障进展分析
    processor.visualize_frequency_analysis_comparison()  # 可视化6：深度频域分析
    processor.visualize_comprehensive_dashboard(source_data)  # 可视化7：综合仪表盘

    # 保存处理后的数据
    print("\n保存处理后的数据...")
    processor.save_processed_data(source_data, target_data, "processed_data")

    # 显示数据统计信息
    print("\n=== 数据统计信息 ===")
    if not source_data.empty:
        print(f"源域数据形状: {source_data.shape}")
        if 'fault_type' in source_data.columns:
            print(f"源域故障类型分布:")
            print(source_data['fault_type'].value_counts())

    if not target_data.empty:
        print(f"目标域数据形状: {target_data.shape}")

    # 检查特征列
    if not source_data.empty:
        feature_cols = [col for col in source_data.columns
                        if col not in ['file_path', 'fault_type', 'fault_size',
                                       'load_level', 'sensor_type', 'rpm', 'fault_label',
                                       'or_position', 'sampling_rate', 'file_id']]
        print(f"\n提取的特征数量: {len(feature_cols)}")
        print("主要特征:", feature_cols[:10] if len(feature_cols) > 10 else feature_cols)

    print("\n=== 可视化功能说明 ===")
    print("1. 信号分析：原始信号、预处理信号、频谱对比、故障频率标注等")
    print("2. 特征分布：关键特征分布、故障频率能量、特征相关性分析")
    print("3. 数据概览：数据集统计、采样频率分布、特征数量统计、箱线图")
    print("4. 3D特征空间：3D散点图、PCA降维、t-SNE降维、特征重要性雷达图")
    print("5. 故障进展分析：故障尺寸趋势、载荷影响、位置对比、演化预测")
    print("6. 深度频域分析：功率谱密度、包络谱、倒频谱、时频图、峰值检测")
    print("7. 综合仪表盘：全方位数据监控、质量评估、实时状态、趋势预测")