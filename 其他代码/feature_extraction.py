#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高速列车轴承智能故障诊断 - 特征提取（简化版）
任务1：从源域数据中筛选数据，进行特征分析，平衡数据集
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BearingFeatureExtractor:
    """轴承特征提取器"""
    
    def __init__(self, source_data_path, target_data_path):
        self.source_data_path = source_data_path
        self.target_data_path = target_data_path
        self.source_data = {}
        self.target_data = {}
        
    def load_source_data(self):
        """加载源域数据"""
        print("正在加载源域数据...")
        
        # 定义故障类型映射
        fault_types = {
            'B': 'Ball_Fault',      # 滚珠故障
            'IR': 'Inner_Race_Fault',  # 内圈故障
            'OR': 'Outer_Race_Fault',  # 外圈故障
            'N': 'Normal'           # 正常
        }
        
        # 加载12kHz DE数据（主要数据）
        data_path = os.path.join(self.source_data_path, "12kHz_DE_data")
        if os.path.exists(data_path):
            self.source_data = {}
            
            # 加载故障数据
            for fault_type in ['B', 'IR', 'OR']:
                fault_path = os.path.join(data_path, fault_type)
                if os.path.exists(fault_path):
                    self.source_data[fault_type] = []
                    
                    for fault_size in os.listdir(fault_path):
                        fault_size_path = os.path.join(fault_path, fault_size)
                        if os.path.isdir(fault_size_path):
                            for file in os.listdir(fault_size_path):
                                if file.endswith('.mat'):
                                    file_path = os.path.join(fault_size_path, file)
                                    try:
                                        mat_data = loadmat(file_path)
                                        # 获取振动信号数据
                                        for key in mat_data.keys():
                                            if not key.startswith('__'):
                                                signal_data = mat_data[key].flatten()
                                                self.source_data[fault_type].append({
                                                    'file': file,
                                                    'data': signal_data,
                                                    'fault_type': fault_types[fault_type],
                                                    'fault_size': fault_size,
                                                    'sampling_rate': '12kHz',
                                                    'sensor': 'DE'
                                                })
                                    except Exception as e:
                                        print(f"加载文件 {file_path} 时出错: {e}")
            
            # 加载正常数据
            normal_path = os.path.join(self.source_data_path, "48kHz_Normal_data")
            if os.path.exists(normal_path):
                self.source_data['N'] = []
                for file in os.listdir(normal_path):
                    if file.endswith('.mat'):
                        file_path = os.path.join(normal_path, file)
                        try:
                            mat_data = loadmat(file_path)
                            for key in mat_data.keys():
                                if not key.startswith('__'):
                                    signal_data = mat_data[key].flatten()
                                    self.source_data['N'].append({
                                        'file': file,
                                        'data': signal_data,
                                        'fault_type': 'Normal',
                                        'fault_size': '0',
                                        'sampling_rate': '48kHz',
                                        'sensor': 'DE'
                                    })
                        except Exception as e:
                            print(f"加载文件 {file_path} 时出错: {e}")
        
        print("源域数据加载完成")
        self._print_data_summary()
    
    def load_target_data(self):
        """加载目标域数据"""
        print("正在加载目标域数据...")
        
        for file in os.listdir(self.target_data_path):
            if file.endswith('.mat'):
                file_path = os.path.join(self.target_data_path, file)
                try:
                    mat_data = loadmat(file_path)
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            signal_data = mat_data[key].flatten()
                            self.target_data[file.replace('.mat', '')] = {
                                'data': signal_data,
                                'file': file
                            }
                except Exception as e:
                    print(f"加载文件 {file_path} 时出错: {e}")
        
        print(f"目标域数据加载完成，共{len(self.target_data)}个文件")
    
    def _print_data_summary(self):
        """打印数据摘要"""
        print("\n=== 源域数据摘要 ===")
        for fault_type in self.source_data:
            count = len(self.source_data[fault_type])
            print(f"{fault_type}: {count}个样本")
    
    def extract_time_domain_features(self, signal_data):
        """提取时域特征"""
        features = {}
        
        if len(signal_data) == 0:
            return {key: 0 for key in ['mean', 'std', 'var', 'rms', 'max', 'min', 'peak_to_peak', 
                                      'skewness', 'kurtosis', 'energy', 'power', 'crest_factor', 
                                      'shape_factor', 'impulse_factor']}
        
        # 基本统计特征
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['max'] = np.max(signal_data)
        features['min'] = np.min(signal_data)
        features['peak_to_peak'] = features['max'] - features['min']
        
        # 形状特征
        features['skewness'] = skew(signal_data)
        features['kurtosis'] = kurtosis(signal_data)
        
        # 能量特征
        features['energy'] = np.sum(signal_data**2)
        features['power'] = features['energy'] / len(signal_data)
        
        # 其他时域特征
        features['crest_factor'] = features['max'] / features['rms'] if features['rms'] != 0 else 0
        features['shape_factor'] = features['rms'] / np.mean(np.abs(signal_data)) if np.mean(np.abs(signal_data)) != 0 else 0
        features['impulse_factor'] = features['max'] / np.mean(np.abs(signal_data)) if np.mean(np.abs(signal_data)) != 0 else 0
        
        return features
    
    def extract_frequency_domain_features(self, signal_data, fs=12000):
        """提取频域特征"""
        features = {}
        
        if len(signal_data) == 0:
            return {key: 0 for key in ['freq_mean', 'freq_std', 'freq_max', 'freq_energy', 
                                      'main_freq', 'freq_peak_count', 'spectral_centroid', 'spectral_bandwidth']}
        
        # 计算FFT
        fft_data = np.fft.fft(signal_data)
        fft_magnitude = np.abs(fft_data)
        fft_power = fft_magnitude**2
        
        # 频率轴
        freqs = np.fft.fftfreq(len(signal_data), 1/fs)
        freqs = freqs[:len(freqs)//2]
        fft_magnitude = fft_magnitude[:len(fft_magnitude)//2]
        fft_power = fft_power[:len(fft_power)//2]
        
        if len(fft_magnitude) == 0:
            return {key: 0 for key in ['freq_mean', 'freq_std', 'freq_max', 'freq_energy', 
                                      'main_freq', 'freq_peak_count', 'spectral_centroid', 'spectral_bandwidth']}
        
        # 频域统计特征
        features['freq_mean'] = np.mean(fft_magnitude)
        features['freq_std'] = np.std(fft_magnitude)
        features['freq_max'] = np.max(fft_magnitude)
        features['freq_energy'] = np.sum(fft_power)
        
        # 主频率特征
        try:
            peak_indices = signal.find_peaks(fft_magnitude, height=np.max(fft_magnitude)*0.1)[0]
            if len(peak_indices) > 0:
                features['main_freq'] = freqs[peak_indices[np.argmax(fft_magnitude[peak_indices])]]
                features['freq_peak_count'] = len(peak_indices)
            else:
                features['main_freq'] = 0
                features['freq_peak_count'] = 0
        except:
            features['main_freq'] = 0
            features['freq_peak_count'] = 0
        
        # 频谱重心
        features['spectral_centroid'] = np.sum(freqs * fft_power) / np.sum(fft_power) if np.sum(fft_power) != 0 else 0
        
        # 频谱带宽
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * fft_power) / np.sum(fft_power)) if np.sum(fft_power) != 0 else 0
        
        return features
    
    def extract_all_features(self, signal_data, fs=12000):
        """提取所有特征"""
        features = {}
        
        # 时域特征
        time_features = self.extract_time_domain_features(signal_data)
        features.update(time_features)
        
        # 频域特征
        freq_features = self.extract_frequency_domain_features(signal_data, fs)
        features.update(freq_features)
        
        return features
    
    def create_balanced_dataset(self, max_samples_per_class=100):
        """创建平衡的数据集"""
        print("正在创建平衡数据集...")
        
        # 收集所有数据
        all_data = []
        
        for fault_type in self.source_data:
            for sample in self.source_data[fault_type]:
                all_data.append(sample)
        
        # 按故障类型分组
        fault_groups = {}
        for sample in all_data:
            fault_type = sample['fault_type']
            if fault_type not in fault_groups:
                fault_groups[fault_type] = []
            fault_groups[fault_type].append(sample)
        
        # 平衡数据集
        balanced_data = []
        for fault_type, samples in fault_groups.items():
            if len(samples) > max_samples_per_class:
                # 随机选择样本
                selected_samples = np.random.choice(samples, max_samples_per_class, replace=False)
            else:
                selected_samples = samples
            balanced_data.extend(selected_samples)
            print(f"{fault_type}: {len(selected_samples)}个样本")
        
        print(f"平衡后总样本数: {len(balanced_data)}")
        return balanced_data
    
    def analyze_features(self, data_samples, fs=12000):
        """分析特征"""
        print("正在提取特征...")
        
        feature_list = []
        labels = []
        
        for i, sample in enumerate(data_samples):
            if i % 50 == 0:
                print(f"处理进度: {i}/{len(data_samples)}")
            
            # 提取特征
            features = self.extract_all_features(sample['data'], fs)
            feature_list.append(features)
            labels.append(sample['fault_type'])
        
        # 转换为DataFrame
        feature_df = pd.DataFrame(feature_list)
        feature_df['label'] = labels
        
        print(f"特征提取完成，共{len(feature_df.columns)-1}个特征")
        return feature_df
    
    def save_features(self, feature_df, save_path='extracted_features.csv'):
        """保存特征数据"""
        feature_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"特征数据已保存到 {save_path}")
        
        # 保存特征统计信息
        stats_path = save_path.replace('.csv', '_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("=== 特征提取统计信息 ===\n\n")
            f.write(f"总样本数: {len(feature_df)}\n")
            f.write(f"特征数量: {len(feature_df.columns)-1}\n\n")
            
            f.write("各故障类型样本数:\n")
            for label, count in feature_df['label'].value_counts().items():
                f.write(f"  {label}: {count}\n")
            
            f.write("\n特征统计信息:\n")
            feature_cols = [col for col in feature_df.columns if col != 'label']
            f.write(feature_df[feature_cols].describe().to_string())
        
        print(f"统计信息已保存到 {stats_path}")
    
    def create_feature_visualization(self, feature_df, save_path='feature_analysis'):
        """创建特征可视化"""
        print("正在生成特征分析可视化...")
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 处理NaN值
        feature_cols = [col for col in feature_df.columns if col != 'label']
        feature_df_clean = feature_df[feature_cols].fillna(0)
        
        # 1. 特征分布图
        plt.figure(figsize=(15, 10))
        
        # 选择前12个特征进行可视化
        selected_features = feature_cols[:12]
        
        for i, feature in enumerate(selected_features):
            plt.subplot(3, 4, i+1)
            for label in feature_df['label'].unique():
                data = feature_df[feature_df['label'] == label][feature].fillna(0)
                plt.hist(data, alpha=0.7, label=label, bins=20)
            plt.title(f'{feature}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 特征相关性热力图
        plt.figure(figsize=(12, 10))
        correlation_matrix = feature_df_clean.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. PCA降维可视化
        plt.figure(figsize=(12, 5))
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df_clean)
        
        # PCA降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.subplot(1, 2, 1)
        for label in feature_df['label'].unique():
            mask = feature_df['label'] == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title('PCA降维可视化')
        plt.legend()
        
        # 特征重要性分析
        plt.subplot(1, 2, 2)
        feature_importance = feature_df_clean.std().sort_values(ascending=False)
        feature_importance.head(15).plot(kind='bar')
        plt.title('特征重要性（标准差）')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化结果已保存到 {save_path} 目录")

def main():
    """主函数"""
    print("=== 高速列车轴承智能故障诊断 - 特征提取 ===\n")
    
    # 初始化特征提取器
    extractor = BearingFeatureExtractor('源域数据集', '目标域数据集')
    
    # 加载数据
    extractor.load_source_data()
    extractor.load_target_data()
    
    # 创建平衡数据集
    balanced_data = extractor.create_balanced_dataset(max_samples_per_class=100)
    
    # 分析特征
    feature_df = extractor.analyze_features(balanced_data, fs=12000)
    
    # 保存特征
    extractor.save_features(feature_df)
    
    # 创建可视化
    extractor.create_feature_visualization(feature_df)
    
    print("\n=== 任务1完成 ===")
    print("已成功完成数据分析与故障特征提取任务")
    print("生成的文件:")
    print("- extracted_features.csv: 特征数据")
    print("- extracted_features_stats.txt: 统计信息")
    print("- feature_analysis/: 可视化结果")

if __name__ == "__main__":
    main()
