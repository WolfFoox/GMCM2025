#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
源域和目标域特征分布分析
绘制核密度估计图展示域间特征分布差异
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
from scipy.signal import resample, welch, find_peaks
import os
import glob
from scipy.io import loadmat
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import pywt
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

class DomainDistributionAnalyzer:
    """域分布分析器"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.source_data = {}
        self.target_data = {}
        self.target_sampling_rate = 32000
        self.segment_length = 2048
        self.overlap_ratio = 0.5
        
    def load_data(self):
        """加载源域和目标域数据"""
        print("Loading data for distribution analysis...")
        
        self._load_source_data()
        self._load_target_data()
        
        print(f"Source domain data loaded: {len(self.source_data)} files")
        print(f"Target domain data loaded: {len(self.target_data)} files")
        
    def _load_source_data(self):
        """加载源域数据"""
        source_path = os.path.join(self.data_path, "源域数据集")
        
        # 故障类型映射
        fault_types = {
            'B': 'Ball_Fault',
            'IR': 'Inner_Race_Fault',
            'OR': 'Outer_Race_Fault',
            'N': 'Normal'
        }
        
        # 加载12kHz DE数据
        de_12k_path = os.path.join(source_path, "12kHz_DE_data")
        if os.path.exists(de_12k_path):
            for fault_type in ['B', 'IR', 'OR']:
                fault_path = os.path.join(de_12k_path, fault_type)
                if os.path.exists(fault_path):
                    for load_dir in os.listdir(fault_path):
                        load_path = os.path.join(fault_path, load_dir)
                        if os.path.isdir(load_path):
                            for mat_file in glob.glob(os.path.join(load_path, "*.mat")):
                                try:
                                    data = loadmat(mat_file)
                                    signal_data = self._extract_signal_data(data)
                                    if signal_data is not None:
                                        filename = os.path.basename(mat_file)
                                        self.source_data[f"12kHz_DE_{fault_type}_{load_dir}_{filename}"] = {
                                            'data': signal_data,
                                            'fault_type': fault_types[fault_type],
                                            'sampling_rate': 12000,
                                            'location': 'DE'
                                        }
                                except Exception as e:
                                    print(f"Failed to load: {mat_file}, Error: {e}")
        
        # 加载48kHz数据
        de_48k_path = os.path.join(source_path, "48kHz_DE_data")
        if os.path.exists(de_48k_path):
            for fault_type in ['B', 'IR', 'OR']:
                fault_path = os.path.join(de_48k_path, fault_type)
                if os.path.exists(fault_path):
                    for load_dir in os.listdir(fault_path):
                        load_path = os.path.join(fault_path, load_dir)
                        if os.path.isdir(load_path):
                            for mat_file in glob.glob(os.path.join(load_path, "*.mat")):
                                try:
                                    data = loadmat(mat_file)
                                    signal_data = self._extract_signal_data(data)
                                    if signal_data is not None:
                                        filename = os.path.basename(mat_file)
                                        self.source_data[f"48kHz_DE_{fault_type}_{load_dir}_{filename}"] = {
                                            'data': signal_data,
                                            'fault_type': fault_types[fault_type],
                                            'sampling_rate': 48000,
                                            'location': 'DE'
                                        }
                                except Exception as e:
                                    print(f"Failed to load: {mat_file}, Error: {e}")
        
        # 加载正常数据
        normal_path = os.path.join(source_path, "48kHz_Normal_data")
        if os.path.exists(normal_path):
            for mat_file in glob.glob(os.path.join(normal_path, "*.mat")):
                try:
                    data = loadmat(mat_file)
                    signal_data = self._extract_signal_data(data)
                    if signal_data is not None:
                        filename = os.path.basename(mat_file)
                        self.source_data[f"48kHz_Normal_{filename}"] = {
                            'data': signal_data,
                            'fault_type': 'Normal',
                            'sampling_rate': 48000,
                            'location': 'DE'
                        }
                except Exception as e:
                    print(f"Failed to load: {mat_file}, Error: {e}")
    
    def _load_target_data(self):
        """加载目标域数据"""
        target_path = os.path.join(self.data_path, "目标域数据集")
        
        for mat_file in glob.glob(os.path.join(target_path, "*.mat")):
            try:
                data = loadmat(mat_file)
                signal_data = self._extract_signal_data(data)
                if signal_data is not None:
                    filename = os.path.basename(mat_file)
                    self.target_data[filename] = {
                        'data': signal_data,
                        'sampling_rate': 32000,
                        'location': 'DE',
                        'filename': filename
                    }
            except Exception as e:
                print(f"Failed to load target file: {mat_file}, Error: {e}")
    
    def _extract_signal_data(self, mat_data):
        """从mat文件中提取振动信号数据"""
        possible_keys = ['X', 'x', 'data', 'signal', 'vibration', 'DE', 'FE', 'BA']
        
        for key in possible_keys:
            if key in mat_data:
                data = mat_data[key]
                if isinstance(data, np.ndarray):
                    if data.ndim > 1:
                        data = data.flatten()
                    return data
        
        for key in mat_data.keys():
            if not key.startswith('__'):
                data = mat_data[key]
                if isinstance(data, np.ndarray) and data.size > 1000:
                    if data.ndim > 1:
                        data = data.flatten()
                    return data
        
        return None
    
    def extract_features(self, signal_data, sampling_rate):
        """提取特征"""
        features = {}
        
        # 时域特征
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['max'] = np.max(signal_data)
        features['min'] = np.min(signal_data)
        features['peak_to_peak'] = features['max'] - features['min']
        features['skewness'] = skew(signal_data)
        features['kurtosis'] = kurtosis(signal_data)
        features['energy'] = np.sum(signal_data**2)
        features['power'] = np.mean(signal_data**2)
        features['impulse_factor'] = features['max'] / np.mean(np.abs(signal_data))
        features['crest_factor'] = features['max'] / features['rms']
        features['shape_factor'] = features['rms'] / np.mean(np.abs(signal_data))
        
        # 冲击能量比
        signal_abs = np.abs(signal_data)
        threshold = np.mean(signal_abs) + 3 * np.std(signal_abs)
        impact_indices = signal_abs > threshold
        features['impact_energy_ratio'] = np.sum(signal_data[impact_indices]**2) / features['energy']
        
        # 包络分析
        envelope = np.abs(signal.hilbert(signal_data))
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_std'] = np.std(envelope)
        features['envelope_kurtosis'] = kurtosis(envelope)
        
        return features
    
    def preprocess_signal(self, signal_data, original_sr, target_sr=None):
        """信号预处理"""
        if target_sr is None:
            target_sr = self.target_sampling_rate
            
        if original_sr != target_sr:
            num_samples = int(len(signal_data) * target_sr / original_sr)
            signal_data = resample(signal_data, num_samples)
        
        # 去噪处理
        nyquist = target_sr / 2
        cutoff = nyquist * 0.8
        b, a = signal.butter(4, cutoff / nyquist, btype='low')
        signal_data = signal.filtfilt(b, a, signal_data)
        
        segments = self._segment_signal(signal_data, target_sr)
        return segments, target_sr
    
    def _segment_signal(self, signal_data, sampling_rate):
        """信号分段处理"""
        segment_length = self.segment_length
        overlap_length = int(segment_length * self.overlap_ratio)
        step_length = segment_length - overlap_length
        
        segments = []
        for i in range(0, len(signal_data) - segment_length + 1, step_length):
            segment = signal_data[i:i + segment_length]
            if len(segment) == segment_length:
                segments.append(segment)
        
        return segments
    
    def build_feature_dataset(self):
        """构建特征数据集"""
        print("Building feature dataset for distribution analysis...")
        
        source_features = []
        target_features = []
        source_labels = []
        
        # 处理源域数据
        for filename, data_info in self.source_data.items():
            signal_data = data_info['data']
            sampling_rate = data_info['sampling_rate']
            
            segments, new_sr = self.preprocess_signal(signal_data, sampling_rate)
            
            for segment in segments[:3]:  # 限制分段数量
                features = self.extract_features(segment, new_sr)
                source_features.append(features)
                source_labels.append(data_info['fault_type'])
        
        # 处理目标域数据
        for filename, data_info in self.target_data.items():
            signal_data = data_info['data']
            sampling_rate = data_info['sampling_rate']
            
            segments, new_sr = self.preprocess_signal(signal_data, sampling_rate)
            
            for segment in segments[:3]:  # 限制分段数量
                features = self.extract_features(segment, new_sr)
                target_features.append(features)
        
        # 转换为DataFrame
        source_df = pd.DataFrame(source_features)
        source_df['domain'] = 'Source'
        source_df['label'] = source_labels
        
        target_df = pd.DataFrame(target_features)
        target_df['domain'] = 'Target'
        
        self.combined_df = pd.concat([source_df, target_df], ignore_index=True)
        
        print(f"Feature dataset built:")
        print(f"- Source samples: {len(source_df)}")
        print(f"- Target samples: {len(target_df)}")
        
        return self.combined_df
    
    def plot_domain_distributions(self, save_path='../04_结果可视化/'):
        """绘制源域和目标域特征分布对比图"""
        if not hasattr(self, 'combined_df'):
            self.build_feature_dataset()
        
        # 选择关键特征进行可视化
        key_features = [
            'rms', 'kurtosis', 'energy', 'envelope_kurtosis',
            'impact_energy_ratio', 'crest_factor', 'shape_factor'
        ]
        
        # 创建图形
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Source Domain vs Target Domain Feature Distributions', 
                     fontsize=16, fontweight='bold')
        
        # 分离源域和目标域数据
        source_df = self.combined_df[self.combined_df['domain'] == 'Source']
        target_df = self.combined_df[self.combined_df['domain'] == 'Target']
        
        # 为每个特征绘制分布图
        for i, feature in enumerate(key_features):
            if i >= 7:  # 只显示7个特征
                break
                
            row = i // 4
            col = i % 4
            
            # 获取源域和目标域的特征数据
            source_data = source_df[feature].dropna()
            target_data = target_df[feature].dropna()
            
            # 标准化数据到0-100范围（模拟您看到的图）
            def normalize_to_100(data):
                if len(data) == 0:
                    return data
                min_val, max_val = np.min(data), np.max(data)
                if max_val == min_val:
                    return np.full_like(data, 50)  # 如果所有值相同，设为50
                return (data - min_val) / (max_val - min_val) * 100
            
            source_norm = normalize_to_100(source_data)
            target_norm = normalize_to_100(target_data)
            
            # 绘制核密度估计图
            if len(source_norm) > 0 and len(target_norm) > 0:
                # 使用seaborn的kdeplot
                sns.kdeplot(data=source_norm, ax=axes[row, col], 
                           fill=True, alpha=0.6, color='blue', label='Source Domain')
                sns.kdeplot(data=target_norm, ax=axes[row, col], 
                           fill=True, alpha=0.6, color='pink', label='Target Domain')
                
                # 设置标题和标签
                axes[row, col].set_title(f'({chr(97+i)}) {feature}', fontweight='bold')
                axes[row, col].set_xlabel('Normalized Value')
                axes[row, col].set_ylabel('Density')
                axes[row, col].set_xlim(0, 100)
                axes[row, col].grid(True, alpha=0.3)
                
                # 添加图例（只在第一个子图添加）
                if i == 0:
                    axes[row, col].legend()
        
        # 隐藏最后一个空的子图
        if len(key_features) < 8:
            axes[1, 3].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        save_file = os.path.join(save_path, 'domain_feature_distributions.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Domain distribution plot saved to: {save_file}")
        
        plt.show()
    
    def plot_detailed_distributions(self, save_path='../04_结果可视化/'):
        """绘制详细的域分布对比图"""
        if not hasattr(self, 'combined_df'):
            self.build_feature_dataset()
        
        # 选择更多特征
        all_features = [col for col in self.combined_df.columns 
                       if col not in ['domain', 'label']]
        
        # 选择前12个特征
        selected_features = all_features[:12]
        
        # 创建图形
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Detailed Source vs Target Domain Feature Distributions', 
                     fontsize=16, fontweight='bold')
        
        # 分离源域和目标域数据
        source_df = self.combined_df[self.combined_df['domain'] == 'Source']
        target_df = self.combined_df[self.combined_df['domain'] == 'Target']
        
        # 为每个特征绘制分布图
        for i, feature in enumerate(selected_features):
            row = i // 4
            col = i % 4
            
            # 获取源域和目标域的特征数据
            source_data = source_df[feature].dropna()
            target_data = target_df[feature].dropna()
            
            if len(source_data) > 0 and len(target_data) > 0:
                # 绘制直方图和KDE
                axes[row, col].hist(source_data, bins=20, alpha=0.6, 
                                   color='blue', label='Source', density=True)
                axes[row, col].hist(target_data, bins=20, alpha=0.6, 
                                   color='pink', label='Target', density=True)
                
                # 添加KDE曲线
                try:
                    from scipy.stats import gaussian_kde
                    source_kde = gaussian_kde(source_data)
                    target_kde = gaussian_kde(target_data)
                    
                    x_range = np.linspace(min(min(source_data), min(target_data)),
                                        max(max(source_data), max(target_data)), 100)
                    axes[row, col].plot(x_range, source_kde(x_range), 
                                       color='darkblue', linewidth=2, label='Source KDE')
                    axes[row, col].plot(x_range, target_kde(x_range), 
                                       color='darkred', linewidth=2, label='Target KDE')
                except:
                    pass
                
                # 设置标题和标签
                axes[row, col].set_title(f'{feature}', fontweight='bold')
                axes[row, col].set_xlabel('Value')
                axes[row, col].set_ylabel('Density')
                axes[row, col].grid(True, alpha=0.3)
                
                # 添加图例（只在第一个子图添加）
                if i == 0:
                    axes[row, col].legend()
        
        plt.tight_layout()
        
        # 保存图片
        save_file = os.path.join(save_path, 'detailed_domain_distributions.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Detailed distribution plot saved to: {save_file}")
        
        plt.show()
    
    def calculate_domain_distance(self):
        """计算域间距离"""
        if not hasattr(self, 'combined_df'):
            self.build_feature_dataset()
        
        source_df = self.combined_df[self.combined_df['domain'] == 'Source']
        target_df = self.combined_df[self.combined_df['domain'] == 'Target']
        
        numeric_cols = [col for col in self.combined_df.columns 
                       if col not in ['domain', 'label']]
        
        distances = {}
        for feature in numeric_cols:
            source_data = source_df[feature].dropna()
            target_data = target_df[feature].dropna()
            
            if len(source_data) > 0 and len(target_data) > 0:
                # 计算Wasserstein距离（简化版）
                from scipy.stats import wasserstein_distance
                try:
                    distance = wasserstein_distance(source_data, target_data)
                    distances[feature] = distance
                except:
                    # 如果计算失败，使用均值差异
                    distance = abs(np.mean(source_data) - np.mean(target_data))
                    distances[feature] = distance
        
        # 按距离排序
        sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)
        
        print("Domain distances (larger = more different):")
        for feature, distance in sorted_distances:
            print(f"{feature}: {distance:.4f}")
        
        return sorted_distances

def main():
    """主函数"""
    data_path = "../数据集"
    
    # 创建域分布分析器
    analyzer = DomainDistributionAnalyzer(data_path)
    
    # 加载数据
    analyzer.load_data()
    
    # 构建特征数据集
    analyzer.build_feature_dataset()
    
    # 绘制域分布对比图
    analyzer.plot_domain_distributions()
    
    # 绘制详细分布图
    analyzer.plot_detailed_distributions()
    
    # 计算域间距离
    analyzer.calculate_domain_distance()
    
    print("Domain distribution analysis completed successfully!")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
