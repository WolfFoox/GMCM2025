import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


class TargetDomainProcessor:
    """目标域数据处理类"""

    def __init__(self, target_data_path):
        """
        初始化目标域数据处理器

        Args:
            target_data_path: 目标域数据文件夹路径
        """
        self.target_data_path = Path(target_data_path)
        self.target_features = None
        self.file_info = {}

        # 目标域参数（根据文档）
        self.sampling_rate = 32000  # 32kHz
        self.rpm = 600  # 约600 rpm (90km/h)
        self.duration = 8  # 8秒采集时间

        # 轴承参数（假设为类似规格）
        self.bearing_params = {
            'n_balls': 9,
            'd': 0.3126,
            'D': 1.537
        }

    def load_mat_file(self, file_path):
        """加载单个.mat文件"""
        try:
            mat_data = sio.loadmat(file_path)

            # 提取有用数据
            data = {}
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    data[key] = value

            return data
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return None

    def extract_signal_from_mat(self, mat_data):
        """从mat数据中提取振动信号"""
        signals = {}

        # 寻找振动信号数据
        for key, value in mat_data.items():
            if isinstance(value, np.ndarray) and len(value.shape) >= 1:
                # 展平信号数据
                signal_data = value.flatten()
                if len(signal_data) > 1000:  # 确保是有效的振动信号
                    signals[key] = signal_data
                    break

        return signals

    def calculate_fault_frequencies(self, rpm):
        """计算故障特征频率"""
        fr = rpm / 60  # 转频
        n = self.bearing_params['n_balls']
        d = self.bearing_params['d']
        D = self.bearing_params['D']

        # 计算故障特征频率
        bpfo = fr * n / 2 * (1 - d / D)  # 外圈故障
        bpfi = fr * n / 2 * (1 + d / D)  # 内圈故障
        bsf = fr * D / d * (1 - (d / D) ** 2)  # 滚动体故障

        return {
            'BPFO': bpfo,
            'BPFI': bpfi,
            'BSF': bsf,
            'FR': fr
        }

    def extract_time_domain_features(self, signal):
        """提取时域特征"""
        features = {}

        # 基本统计特征
        features['rms'] = np.sqrt(np.mean(signal ** 2))
        features['peak'] = np.max(np.abs(signal))
        features['std'] = np.std(signal)
        features['mean'] = np.mean(signal)

        # 重要形状特征
        features['kurtosis'] = self._kurtosis(signal)
        features['skewness'] = self._skewness(signal)

        # 无量纲特征
        if features['rms'] > 0:
            features['crest_factor'] = features['peak'] / features['rms']
        else:
            features['crest_factor'] = 0

        return features

    def _skewness(self, signal):
        """计算偏度"""
        if np.std(signal) == 0:
            return 0
        return np.mean(((signal - np.mean(signal)) / np.std(signal)) ** 3)

    def _kurtosis(self, signal):
        """计算峭度"""
        if np.std(signal) == 0:
            return 0
        return np.mean(((signal - np.mean(signal)) / np.std(signal)) ** 4)

    def extract_frequency_domain_features(self, signal, fs, fault_freqs):
        """提取频域特征"""
        features = {}

        # FFT变换
        n = len(signal)
        fft_vals = fft(signal)
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

        # 故障特征频率相关特征
        for fault_name in ['BPFO', 'BPFI', 'BSF']:
            fault_freq = fault_freqs.get(fault_name, 0)
            if fault_freq > 0 and fault_freq < fs / 2:
                # 找到最接近故障频率的索引
                idx = np.argmin(np.abs(freqs - fault_freq))
                features[f'{fault_name}_amplitude'] = magnitude[idx]

                # 故障频率的前3个谐波能量
                harmonics_energy = 0
                for h in range(1, 4):
                    harmonic_freq = fault_freq * h
                    if harmonic_freq < fs / 2:
                        h_idx = np.argmin(np.abs(freqs - harmonic_freq))
                        harmonics_energy += magnitude[h_idx]
                features[f'{fault_name}_harmonics_energy'] = harmonics_energy

        return features

    def extract_time_frequency_features(self, signal, fs):
        """提取时频域特征"""
        features = {}

        from scipy.signal import welch

        try:
            freqs, psd = welch(signal, fs, nperseg=min(1024, len(signal) // 4))
        except:
            freqs, psd = welch(signal, fs, nperseg=min(256, len(signal) // 2))

        # 频带能量比例
        total_energy = np.sum(psd)

        # 低频段 (0 - fs/4)
        low_mask = freqs < fs / 4
        low_energy = np.sum(psd[low_mask])
        features['low_band_ratio'] = low_energy / total_energy if total_energy > 0 else 0

        # 高频段 (fs/4 - fs/2)
        high_mask = freqs >= fs / 4
        high_energy = np.sum(psd[high_mask])
        features['high_band_ratio'] = high_energy / total_energy if total_energy > 0 else 0

        return features

    def preprocess_signal(self, signal, fs):
        """信号预处理"""
        # 去除直流分量
        signal = signal - np.mean(signal)

        # 带通滤波
        try:
            nyquist = fs / 2
            low_cutoff = 10 / nyquist
            high_cutoff = min(5000 / nyquist, 0.95)

            if low_cutoff < high_cutoff and low_cutoff > 0 and high_cutoff < 1:
                b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
                signal = signal.filtfilt(b, a, signal)
        except:
            pass

        return signal

    def process_single_file(self, file_path):
        """处理单个目标域文件"""
        file_id = file_path.stem

        # 加载数据
        mat_data = self.load_mat_file(file_path)
        if mat_data is None:
            return None

        # 提取信号
        signals = self.extract_signal_from_mat(mat_data)
        if not signals:
            print(f"警告: 文件 {file_id} 中未找到有效振动信号")
            return None

        # 获取主要振动信号
        signal_key = list(signals.keys())[0]
        signal_data = signals[signal_key]

        result = {
            'file_id': file_id,
            'signal_length': len(signal_data),
            'sampling_rate': self.sampling_rate,
            'rpm': self.rpm,
            'signal_key': signal_key
        }

        # 信号预处理
        processed_signal = self.preprocess_signal(signal_data, self.sampling_rate)

        # 计算故障特征频率
        fault_freqs = self.calculate_fault_frequencies(self.rpm)

        # 提取特征
        time_features = self.extract_time_domain_features(processed_signal)
        freq_features = self.extract_frequency_domain_features(processed_signal, self.sampling_rate, fault_freqs)
        tf_features = self.extract_time_frequency_features(processed_signal, self.sampling_rate)

        # 合并所有特征
        result.update(time_features)
        result.update(freq_features)
        result.update(tf_features)

        return result

    def process_all_files(self):
        """处理所有目标域文件"""
        print("开始处理目标域数据...")
        print("=" * 50)

        if not self.target_data_path.exists():
            print(f"错误: 目标域路径不存在 {self.target_data_path}")
            return None

        # 获取所有.mat文件
        mat_files = list(self.target_data_path.glob('*.mat'))

        if not mat_files:
            print("错误: 未找到.mat文件")
            return None

        print(f"找到 {len(mat_files)} 个目标域文件")

        all_features = []

        for mat_file in sorted(mat_files):
            print(f"处理文件: {mat_file.name}")
            features = self.process_single_file(mat_file)
            if features:
                all_features.append(features)

        if not all_features:
            print("错误: 没有成功处理任何文件")
            return None

        # 转换为DataFrame
        self.target_features = pd.DataFrame(all_features)

        print(f"\n成功处理 {len(self.target_features)} 个文件")
        print(
            f"提取的特征数量: {len([col for col in self.target_features.columns if col not in ['file_id', 'signal_length', 'sampling_rate', 'rpm', 'signal_key']])}")

        return self.target_features

    def visualize_target_data(self):
        """可视化目标域数据分析"""
        if self.target_features is None:
            print("错误: 请先运行 process_all_files()")
            return

        print("\n目标域数据可视化分析")
        print("=" * 50)

        # 获取特征列
        feature_cols = [col for col in self.target_features.columns
                        if col not in ['file_id', 'signal_length', 'sampling_rate', 'rpm', 'signal_key']]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('目标域数据特征分析', fontsize=16, fontweight='bold')

        # 1. 文件ID分布
        file_ids = self.target_features['file_id'].tolist()
        axes[0, 0].bar(range(len(file_ids)), [1] * len(file_ids), color='skyblue')
        axes[0, 0].set_title('目标域文件分布 (A-P)', fontweight='bold')
        axes[0, 0].set_xlabel('文件')
        axes[0, 0].set_ylabel('文件数')
        axes[0, 0].set_xticks(range(len(file_ids)))
        axes[0, 0].set_xticklabels(file_ids, rotation=45)

        # 2. 信号长度分布
        signal_lengths = self.target_features['signal_length']
        axes[0, 1].hist(signal_lengths, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('信号长度分布', fontweight='bold')
        axes[0, 1].set_xlabel('信号长度 (采样点)')
        axes[0, 1].set_ylabel('频次')

        # 3. RMS特征分布
        if 'rms' in self.target_features.columns:
            rms_values = self.target_features['rms']
            axes[0, 2].hist(rms_values, bins=15, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('RMS值分布', fontweight='bold')
            axes[0, 2].set_xlabel('RMS值')
            axes[0, 2].set_ylabel('频次')

        # 4. 峭度特征分布
        if 'kurtosis' in self.target_features.columns:
            kurtosis_values = self.target_features['kurtosis']
            axes[1, 0].hist(kurtosis_values, bins=15, color='gold', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('峭度分布', fontweight='bold')
            axes[1, 0].set_xlabel('峭度值')
            axes[1, 0].set_ylabel('频次')

        # 5. 峰值因子分布
        if 'crest_factor' in self.target_features.columns:
            crest_values = self.target_features['crest_factor']
            axes[1, 1].hist(crest_values, bins=15, color='plum', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('峰值因子分布', fontweight='bold')
            axes[1, 1].set_xlabel('峰值因子')
            axes[1, 1].set_ylabel('频次')

        # 6. 多个关键特征的文件对比
        key_features = ['rms', 'kurtosis', 'crest_factor', 'spectral_centroid']
        available_features = [f for f in key_features if f in self.target_features.columns]

        if available_features:
            x_pos = np.arange(len(file_ids))
            width = 0.8 / len(available_features)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

            for i, feature in enumerate(available_features):
                # 标准化特征值用于显示
                feature_values = self.target_features[feature]
                normalized_values = (feature_values - feature_values.min()) / (
                            feature_values.max() - feature_values.min())

                axes[1, 2].bar(x_pos + i * width, normalized_values, width=width,
                               label=feature, color=colors[i % len(colors)], alpha=0.7)

            axes[1, 2].set_title('关键特征对比 (标准化)', fontweight='bold')
            axes[1, 2].set_xlabel('文件ID')
            axes[1, 2].set_ylabel('标准化值')
            axes[1, 2].set_xticks(x_pos + width * (len(available_features) - 1) / 2)
            axes[1, 2].set_xticklabels(file_ids, rotation=45)
            axes[1, 2].legend()

        plt.tight_layout()
        plt.show()

    def analyze_fault_indicators(self):
        """分析故障指示特征"""
        if self.target_features is None:
            print("错误: 请先运行 process_all_files()")
            return

        print("\n故障指示特征分析")
        print("=" * 50)

        # 关键故障指示特征
        fault_indicators = {
            'kurtosis': {'threshold': 3.5, 'desc': '峭度 (>3.5可能表示冲击性故障)'},
            'crest_factor': {'threshold': 4.0, 'desc': '峰值因子 (>4.0可能表示故障)'},
            'rms': {'desc': 'RMS值 (相对水平)'}
        }

        # 分析每个文件的故障指示
        results = []

        for idx, row in self.target_features.iterrows():
            file_id = row['file_id']
            analysis = {'file_id': file_id, 'indicators': []}

            # 检查峭度
            if 'kurtosis' in row and pd.notna(row['kurtosis']):
                kurtosis_val = row['kurtosis']
                if kurtosis_val > fault_indicators['kurtosis']['threshold']:
                    analysis['indicators'].append(f"高峭度({kurtosis_val:.2f})")

            # 检查峰值因子
            if 'crest_factor' in row and pd.notna(row['crest_factor']):
                crest_val = row['crest_factor']
                if crest_val > fault_indicators['crest_factor']['threshold']:
                    analysis['indicators'].append(f"高峰值因子({crest_val:.2f})")

            # 检查故障特征频率能量
            fault_freq_features = [col for col in row.index if 'amplitude' in col or 'harmonics_energy' in col]
            high_fault_energy = []

            for feature in fault_freq_features:
                if pd.notna(row[feature]) and row[feature] > np.percentile(self.target_features[feature].dropna(), 75):
                    high_fault_energy.append(feature)

            if high_fault_energy:
                analysis['indicators'].append(f"故障频率能量异常({len(high_fault_energy)}个)")

            analysis['total_indicators'] = len(analysis['indicators'])
            results.append(analysis)

        # 打印分析结果
        print("\n各文件故障指示分析:")
        print("-" * 60)

        for result in sorted(results, key=lambda x: x['total_indicators'], reverse=True):
            status = "疑似故障" if result['total_indicators'] > 0 else "可能正常"
            indicators_str = ", ".join(result['indicators']) if result['indicators'] else "无明显异常"
            print(f"文件 {result['file_id']:<3}: {status:<8} - {indicators_str}")

        # 统计汇总
        suspect_files = [r for r in results if r['total_indicators'] > 0]
        print(f"\n汇总:")
        print(f"  疑似故障文件: {len(suspect_files)}/{len(results)}")
        print(f"  疑似正常文件: {len(results) - len(suspect_files)}/{len(results)}")

        return results

    def save_processed_data(self, output_path="processed_data"):
        """保存处理后的目标域数据"""
        if self.target_features is None:
            print("错误: 请先运行 process_all_files()")
            return

        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        # 保存特征数据
        target_file = output_dir / 'target_domain_features.csv'
        self.target_features.to_csv(target_file, index=False)
        print(f"目标域特征数据已保存: {target_file}")

        # 保存特征名称
        feature_columns = [col for col in self.target_features.columns
                           if col not in ['file_id', 'signal_length', 'sampling_rate', 'rpm', 'signal_key']]

        feature_file = output_dir / 'target_domain_feature_names.txt'
        with open(feature_file, 'w') as f:
            for feature in feature_columns:
                f.write(f"{feature}\n")
        print(f"特征名称已保存: {feature_file}")

        print(f"\n目标域数据处理完成:")
        print(f"  文件数量: {len(self.target_features)}")
        print(f"  特征数量: {len(feature_columns)}")
        print(f"  保存路径: {output_dir}")

    def run_full_analysis(self):
        """运行完整的目标域数据分析"""
        print("开始目标域数据完整分析")
        print("=" * 60)

        # 1. 处理所有文件
        self.process_all_files()

        # 2. 可视化分析
        self.visualize_target_data()

        # 3. 故障指示分析
        fault_analysis = self.analyze_fault_indicators()

        # 4. 保存处理结果
        self.save_processed_data()

        print("\n目标域数据分析完成!")
        return self.target_features, fault_analysis


# 使用示例
if __name__ == "__main__":
    # 初始化目标域处理器
    target_processor = TargetDomainProcessor('../目标域数据集')  # 修改为你的实际路径

    # 运行完整分析
    target_features, fault_analysis = target_processor.run_full_analysis()

    # 显示基本统计
    if target_features is not None:
        print(f"\n目标域数据概览:")
        print(f"文件列表: {sorted(target_features['file_id'].tolist())}")
        print(f"数据形状: {target_features.shape}")