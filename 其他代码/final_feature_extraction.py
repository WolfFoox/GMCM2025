#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终优化的特征提取方案
基于12kHz数据优先策略，包含所有故障类型
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FinalFeatureExtractor:
    """最终特征提取器"""
    
    def __init__(self, source_data_path):
        self.source_data_path = source_data_path
        self.all_data = []
        self.feature_df = None
        
    def load_12kHz_data(self):
        """加载12kHz数据（优先数据源）"""
        print("=== 加载12kHz数据（优先数据源）===")
        
        # 定义故障类型映射
        fault_types = {
            'B': 'Ball_Fault',      # 滚珠故障
            'IR': 'Inner_Race_Fault',  # 内圈故障
            'OR': 'Outer_Race_Fault',  # 外圈故障
            'N': 'Normal'           # 正常
        }
        
        total_loaded = 0
        valid_samples = 0
        invalid_samples = 0
        
        # 加载12kHz DE数据（主要数据源）
        print("\n--- 12kHz DE 传感器数据 ---")
        de_path = os.path.join(self.source_data_path, "12kHz_DE_data")
        if os.path.exists(de_path):
            # 加载B和IR数据
            for fault_type in ['B', 'IR']:
                fault_path = os.path.join(de_path, fault_type)
                if os.path.exists(fault_path):
                    count = self._load_fault_data(fault_path, fault_types[fault_type], 
                                                '12kHz', 'DE')
                    valid_samples += count
                    print(f"  {fault_type}: {count}个样本")
            
            # 加载OR数据
            or_path = os.path.join(de_path, 'OR')
            if os.path.exists(or_path):
                or_count = self._load_or_data(or_path, 'Outer_Race_Fault', '12kHz', 'DE')
                valid_samples += or_count
                print(f"  OR: {or_count}个样本")
        
        # 加载12kHz FE数据（补充数据源）
        print("\n--- 12kHz FE 传感器数据 ---")
        fe_path = os.path.join(self.source_data_path, "12kHz_FE_data")
        if os.path.exists(fe_path):
            # 加载B和IR数据
            for fault_type in ['B', 'IR']:
                fault_path = os.path.join(fe_path, fault_type)
                if os.path.exists(fault_path):
                    count = self._load_fault_data(fault_path, fault_types[fault_type], 
                                                '12kHz', 'FE')
                    valid_samples += count
                    print(f"  {fault_type}: {count}个样本")
            
            # 加载OR数据
            or_path = os.path.join(fe_path, 'OR')
            if os.path.exists(or_path):
                or_count = self._load_or_data(or_path, 'Outer_Race_Fault', '12kHz', 'FE')
                valid_samples += or_count
                print(f"  OR: {or_count}个样本")
        
        # 加载正常数据（48kHz）
        print("\n--- 正常数据（48kHz）---")
        normal_path = os.path.join(self.source_data_path, "48kHz_Normal_data")
        if os.path.exists(normal_path):
            normal_count = self._load_fault_data(normal_path, 'Normal', '48kHz', 'DE')
            valid_samples += normal_count
            print(f"  Normal: {normal_count}个样本")
        
        print(f"\n数据加载统计:")
        print(f"  有效样本: {valid_samples}")
        print(f"  无效样本: {invalid_samples}")
        print(f"  数据有效率: {valid_samples/(valid_samples+invalid_samples)*100:.2f}%")
        
        return valid_samples, invalid_samples
    
    def _load_fault_data(self, fault_path, fault_type, sampling_rate, sensor_type):
        """加载故障数据"""
        count = 0
        for fault_size in os.listdir(fault_path):
            fault_size_path = os.path.join(fault_path, fault_size)
            if os.path.isdir(fault_size_path):
                for file in os.listdir(fault_size_path):
                    if file.endswith('.mat'):
                        file_path = os.path.join(fault_size_path, file)
                        try:
                            mat_data = loadmat(file_path)
                            for key in mat_data.keys():
                                if not key.startswith('__'):
                                    signal_data = mat_data[key].flatten()
                                    if self._validate_signal(signal_data):
                                        self.all_data.append({
                                            'file': file,
                                            'data': signal_data,
                                            'fault_type': fault_type,
                                            'fault_size': fault_size,
                                            'sampling_rate': sampling_rate,
                                            'sensor': sensor_type,
                                            'file_path': file_path
                                        })
                                        count += 1
                        except Exception as e:
                            print(f"加载失败: {file_path} - {e}")
        return count
    
    def _load_or_data(self, or_path, fault_type, sampling_rate, sensor_type):
        """加载OR数据"""
        count = 0
        for or_type in ['Centered', 'Opposite', 'Orthogonal']:
            or_type_path = os.path.join(or_path, or_type)
            if os.path.exists(or_type_path):
                for fault_size in os.listdir(or_type_path):
                    fault_size_path = os.path.join(or_type_path, fault_size)
                    if os.path.isdir(fault_size_path):
                        for file in os.listdir(fault_size_path):
                            if file.endswith('.mat'):
                                file_path = os.path.join(fault_size_path, file)
                                try:
                                    mat_data = loadmat(file_path)
                                    for key in mat_data.keys():
                                        if not key.startswith('__'):
                                            signal_data = mat_data[key].flatten()
                                            if self._validate_signal(signal_data):
                                                self.all_data.append({
                                                    'file': file,
                                                    'data': signal_data,
                                                    'fault_type': fault_type,
                                                    'fault_size': fault_size,
                                                    'sampling_rate': sampling_rate,
                                                    'sensor': sensor_type,
                                                    'or_type': or_type,
                                                    'file_path': file_path
                                                })
                                                count += 1
                                except Exception as e:
                                    print(f"加载失败: {file_path} - {e}")
        return count
    
    def _validate_signal(self, signal_data):
        """验证信号数据质量"""
        if len(signal_data) == 0:
            return False
        if np.all(signal_data == 0):
            return False
        if np.any(np.isnan(signal_data)):
            return False
        if np.any(np.isinf(signal_data)):
            return False
        if len(signal_data) < 100:  # 信号太短
            return False
        return True
    
    def analyze_data_distribution(self):
        """分析数据分布"""
        print("\n=== 数据分布分析 ===")
        
        # 按故障类型统计
        fault_counts = {}
        for sample in self.all_data:
            fault_type = sample['fault_type']
            if fault_type not in fault_counts:
                fault_counts[fault_type] = 0
            fault_counts[fault_type] += 1
        
        print("各故障类型样本数:")
        for fault_type, count in fault_counts.items():
            print(f"  {fault_type}: {count}个样本")
        
        # 按传感器类型统计
        sensor_counts = {}
        for sample in self.all_data:
            sensor = sample['sensor']
            if sensor not in sensor_counts:
                sensor_counts[sensor] = 0
            sensor_counts[sensor] += 1
        
        print("\n各传感器样本数:")
        for sensor, count in sensor_counts.items():
            print(f"  {sensor}: {count}个样本")
        
        return fault_counts, sensor_counts
    
    def create_balanced_dataset(self, max_samples_per_class=100):
        """创建平衡数据集"""
        print(f"\n=== 创建平衡数据集 ===")
        
        # 按故障类型分组
        fault_groups = {}
        for sample in self.all_data:
            fault_type = sample['fault_type']
            if fault_type not in fault_groups:
                fault_groups[fault_type] = []
            fault_groups[fault_type].append(sample)
        
        # 创建平衡数据集
        balanced_data = []
        for fault_type, samples in fault_groups.items():
            if len(samples) > max_samples_per_class:
                # 随机选择样本
                selected_samples = np.random.choice(samples, max_samples_per_class, replace=False)
            else:
                selected_samples = samples
            balanced_data.extend(selected_samples)
            print(f"  {fault_type}: {len(selected_samples)}个样本")
        
        print(f"平衡后总样本数: {len(balanced_data)}")
        return balanced_data
    
    def extract_comprehensive_features(self, signal_data, fs=12000):
        """提取综合特征"""
        features = {}
        
        if len(signal_data) == 0:
            return self._get_zero_features()
        
        # 1. 时域特征
        features.update(self._extract_time_domain_features(signal_data))
        
        # 2. 频域特征
        features.update(self._extract_frequency_domain_features(signal_data, fs))
        
        # 3. 时频域特征
        features.update(self._extract_time_frequency_features(signal_data))
        
        return features
    
    def _get_zero_features(self):
        """返回零特征"""
        return {key: 0 for key in [
            'mean', 'std', 'var', 'rms', 'max', 'min', 'peak_to_peak',
            'skewness', 'kurtosis', 'energy', 'power', 'crest_factor',
            'shape_factor', 'impulse_factor', 'freq_mean', 'freq_std',
            'freq_max', 'freq_energy', 'main_freq', 'freq_peak_count',
            'spectral_centroid', 'spectral_bandwidth', 'zero_crossing_rate',
            'mean_abs_diff', 'variance_abs_diff', 'max_abs_diff'
        ]}
    
    def _extract_time_domain_features(self, signal_data):
        """提取时域特征"""
        features = {}
        
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
    
    def _extract_frequency_domain_features(self, signal_data, fs):
        """提取频域特征"""
        features = {}
        
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
    
    def _extract_time_frequency_features(self, signal_data):
        """提取时频域特征"""
        features = {}
        
        # 零交叉率
        zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
        
        # 差分特征
        diff_signal = np.diff(signal_data)
        features['mean_abs_diff'] = np.mean(np.abs(diff_signal))
        features['variance_abs_diff'] = np.var(np.abs(diff_signal))
        features['max_abs_diff'] = np.max(np.abs(diff_signal))
        
        return features
    
    def extract_features_from_dataset(self, data_samples, fs=12000):
        """从数据集中提取特征"""
        print("正在提取特征...")
        
        feature_list = []
        labels = []
        metadata = []
        
        for i, sample in enumerate(data_samples):
            if i % 50 == 0:
                print(f"处理进度: {i}/{len(data_samples)}")
            
            # 提取特征
            features = self.extract_comprehensive_features(sample['data'], fs)
            feature_list.append(features)
            labels.append(sample['fault_type'])
            metadata.append({
                'file': sample['file'],
                'fault_size': sample['fault_size'],
                'sampling_rate': sample['sampling_rate'],
                'sensor': sample['sensor']
            })
        
        # 转换为DataFrame
        feature_df = pd.DataFrame(feature_list)
        feature_df['label'] = labels
        
        # 添加元数据
        for i, meta in enumerate(metadata):
            for key, value in meta.items():
                feature_df.loc[i, key] = value
        
        print(f"特征提取完成，共{len(feature_df.columns)-1}个特征")
        return feature_df
    
    def analyze_feature_importance(self, feature_df):
        """分析特征重要性"""
        print("\n=== 特征重要性分析 ===")
        
        feature_cols = [col for col in feature_df.columns if col not in ['label', 'file', 'fault_size', 'sampling_rate', 'sensor']]
        
        # 计算特征与标签的相关性
        correlations = {}
        for feature in feature_cols:
            if feature_df[feature].dtype in ['float64', 'int64']:
                corr = feature_df[feature].corr(feature_df['label'].astype('category').cat.codes)
                correlations[feature] = abs(corr)
        
        # 按相关性排序
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print("前10个重要特征:")
        for i, (feature, corr) in enumerate(sorted_features[:10]):
            print(f"  {i+1}. {feature}: {corr:.4f}")
        
        return sorted_features
    
    def create_visualization(self, fault_counts, balanced_data, feature_df, save_path='final_analysis'):
        """创建可视化"""
        print("正在生成可视化...")
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 数据分布对比
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        fault_types = list(fault_counts.keys())
        raw_counts = list(fault_counts.values())
        
        plt.bar(fault_types, raw_counts)
        plt.title('原始数据分布')
        plt.xticks(rotation=45)
        plt.ylabel('样本数量')
        
        # 添加数值标签
        for i, v in enumerate(raw_counts):
            plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        # 2. 平衡后数据分布
        plt.subplot(2, 3, 2)
        balanced_counts = {}
        for sample in balanced_data:
            fault_type = sample['fault_type']
            if fault_type not in balanced_counts:
                balanced_counts[fault_type] = 0
            balanced_counts[fault_type] += 1
        
        balanced_fault_types = list(balanced_counts.keys())
        balanced_fault_counts = list(balanced_counts.values())
        
        plt.bar(balanced_fault_types, balanced_fault_counts)
        plt.title('平衡后数据分布')
        plt.xticks(rotation=45)
        plt.ylabel('样本数量')
        
        # 添加数值标签
        for i, v in enumerate(balanced_fault_counts):
            plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        # 3. 特征相关性热力图
        plt.subplot(2, 3, 3)
        feature_cols = [col for col in feature_df.columns if col not in ['label', 'file', 'fault_size', 'sampling_rate', 'sensor']]
        correlation_matrix = feature_df[feature_cols[:10]].corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('特征相关性')
        
        # 4. PCA降维可视化
        plt.subplot(2, 3, 4)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df[feature_cols].fillna(0))
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        for label in feature_df['label'].unique():
            mask = feature_df['label'] == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title('PCA降维可视化')
        plt.legend()
        
        # 5. 特征重要性
        plt.subplot(2, 3, 5)
        sorted_features = self.analyze_feature_importance(feature_df)
        top_features = sorted_features[:10]
        feature_names = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]
        
        plt.barh(feature_names, importance_values)
        plt.title('特征重要性')
        plt.xlabel('相关性')
        
        # 6. 传感器分布
        plt.subplot(2, 3, 6)
        sensor_counts = feature_df['sensor'].value_counts()
        plt.pie(sensor_counts.values, labels=sensor_counts.index, autopct='%1.1f%%')
        plt.title('传感器分布')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/final_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"可视化结果已保存到 {save_path} 目录")
    
    def save_results(self, feature_df, save_path='final_results'):
        """保存结果"""
        print("正在保存结果...")
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 保存特征数据
        feature_df.to_csv(f'{save_path}/final_features.csv', index=False, encoding='utf-8-sig')
        
        # 保存统计信息
        with open(f'{save_path}/final_stats.txt', 'w', encoding='utf-8') as f:
            f.write("=== 最终特征提取统计信息 ===\n\n")
            f.write(f"总样本数: {len(feature_df)}\n")
            f.write(f"特征数量: {len(feature_df.columns)-1}\n\n")
            
            f.write("各故障类型样本数:\n")
            for label, count in feature_df['label'].value_counts().items():
                f.write(f"  {label}: {count}个样本\n")
            
            f.write("\n各传感器样本数:\n")
            for sensor, count in feature_df['sensor'].value_counts().items():
                f.write(f"  {sensor}: {count}个样本\n")
            
            f.write("\n各采样率样本数:\n")
            for sampling_rate, count in feature_df['sampling_rate'].value_counts().items():
                f.write(f"  {sampling_rate}: {count}个样本\n")
        
        print(f"结果已保存到 {save_path} 目录")

def main():
    """主函数"""
    print("=== 最终特征提取方案 ===")
    
    extractor = FinalFeatureExtractor('源域数据集')
    
    # 1. 加载12kHz数据（优先数据源）
    valid_samples, invalid_samples = extractor.load_12kHz_data()
    
    # 2. 分析数据分布
    fault_counts, sensor_counts = extractor.analyze_data_distribution()
    
    # 3. 创建平衡数据集
    balanced_data = extractor.create_balanced_dataset(max_samples_per_class=100)
    
    # 4. 提取特征
    feature_df = extractor.extract_features_from_dataset(balanced_data)
    
    # 5. 创建可视化
    extractor.create_visualization(fault_counts, balanced_data, feature_df)
    
    # 6. 保存结果
    extractor.save_results(feature_df)
    
    print("\n=== 任务1完成 ===")
    print(f"原始有效数据: {valid_samples}个样本")
    print(f"处理后数据: {len(balanced_data)}个样本")
    print(f"数据利用率: {len(balanced_data)/valid_samples*100:.2f}%")

if __name__ == "__main__":
    main()
