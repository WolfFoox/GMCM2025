#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面的数据加载策略
包含所有采样频率和传感器类型
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveDataLoader:
    """全面数据加载器"""
    
    def __init__(self, source_data_path):
        self.source_data_path = source_data_path
        self.all_data = []
        self.data_stats = {}
        
    def load_all_data_sources(self):
        """加载所有数据源"""
        print("=== 全面数据加载策略 ===")
        
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
        
        # 1. 加载12kHz数据（DE和FE传感器）
        print("\n--- 12kHz数据加载 ---")
        for sensor_type in ['DE', 'FE']:
            print(f"\n  12kHz {sensor_type} 传感器:")
            data_path = os.path.join(self.source_data_path, f"12kHz_{sensor_type}_data")
            if os.path.exists(data_path):
                # 加载B和IR数据
                for fault_type in ['B', 'IR']:
                    fault_path = os.path.join(data_path, fault_type)
                    if os.path.exists(fault_path):
                        count = self._load_fault_data(fault_path, fault_types[fault_type], 
                                                    '12kHz', sensor_type, 'DE')
                        valid_samples += count
                        print(f"    {fault_type}: {count}个样本")
                
                # 加载OR数据
                or_path = os.path.join(data_path, 'OR')
                if os.path.exists(or_path):
                    or_count = self._load_or_data(or_path, 'Outer_Race_Fault', 
                                                '12kHz', sensor_type)
                    valid_samples += or_count
                    print(f"    OR: {or_count}个样本")
        
        # 2. 加载48kHz数据（DE和FE传感器）
        print("\n--- 48kHz数据加载 ---")
        for sensor_type in ['DE', 'FE']:
            print(f"\n  48kHz {sensor_type} 传感器:")
            data_path = os.path.join(self.source_data_path, f"48kHz_{sensor_type}_data")
            if os.path.exists(data_path):
                # 加载B和IR数据
                for fault_type in ['B', 'IR']:
                    fault_path = os.path.join(data_path, fault_type)
                    if os.path.exists(fault_path):
                        count = self._load_fault_data(fault_path, fault_types[fault_type], 
                                                    '48kHz', sensor_type, 'DE')
                        valid_samples += count
                        print(f"    {fault_type}: {count}个样本")
                
                # 加载OR数据
                or_path = os.path.join(data_path, 'OR')
                if os.path.exists(or_path):
                    or_count = self._load_or_data(or_path, 'Outer_Race_Fault', 
                                                '48kHz', sensor_type)
                    valid_samples += or_count
                    print(f"    OR: {or_count}个样本")
        
        # 3. 加载正常数据
        print("\n--- 正常数据加载 ---")
        normal_path = os.path.join(self.source_data_path, "48kHz_Normal_data")
        if os.path.exists(normal_path):
            normal_count = self._load_fault_data(normal_path, 'Normal', 
                                               '48kHz', 'DE', 'DE')
            valid_samples += normal_count
            print(f"  Normal: {normal_count}个样本")
        
        print(f"\n数据加载统计:")
        print(f"  总文件数: {total_loaded}")
        print(f"  有效样本: {valid_samples}")
        print(f"  无效样本: {invalid_samples}")
        if total_loaded > 0:
            print(f"  数据有效率: {valid_samples/total_loaded*100:.2f}%")
        else:
            print(f"  数据有效率: 100.00%")
        
        return valid_samples, invalid_samples
    
    def _load_fault_data(self, fault_path, fault_type, sampling_rate, sensor_type, sensor_name):
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
                                            'sensor': sensor_name,
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
        
        # 按采样频率统计
        sampling_counts = {}
        for sample in self.all_data:
            sampling_rate = sample['sampling_rate']
            if sampling_rate not in sampling_counts:
                sampling_counts[sampling_rate] = 0
            sampling_counts[sampling_rate] += 1
        
        print("\n各采样频率样本数:")
        for sampling_rate, count in sampling_counts.items():
            print(f"  {sampling_rate}: {count}个样本")
        
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
        
        return fault_counts, sampling_counts, sensor_counts
    
    def create_balanced_dataset(self, strategy='intelligent'):
        """创建智能平衡数据集"""
        print(f"\n=== 创建智能平衡数据集 (策略: {strategy}) ===")
        
        # 按故障类型分组
        fault_groups = {}
        for sample in self.all_data:
            fault_type = sample['fault_type']
            if fault_type not in fault_groups:
                fault_groups[fault_type] = []
            fault_groups[fault_type].append(sample)
        
        # 智能平衡策略
        if strategy == 'intelligent':
            # 优先使用12kHz DE数据，补充其他数据源
            balanced_data = []
            
            for fault_type, samples in fault_groups.items():
                # 按优先级排序：12kHz DE > 12kHz FE > 48kHz DE > 48kHz FE
                priority_samples = []
                for sample in samples:
                    priority = 0
                    if sample['sampling_rate'] == '12kHz' and sample['sensor'] == 'DE':
                        priority = 4
                    elif sample['sampling_rate'] == '12kHz' and sample['sensor'] == 'FE':
                        priority = 3
                    elif sample['sampling_rate'] == '48kHz' and sample['sensor'] == 'DE':
                        priority = 2
                    else:
                        priority = 1
                    priority_samples.append((priority, sample))
                
                # 按优先级排序
                priority_samples.sort(key=lambda x: x[0], reverse=True)
                
                # 选择样本
                if fault_type == 'Normal':
                    # 正常数据保持原样
                    selected_samples = [s[1] for s in priority_samples]
                else:
                    # 其他故障类型选择前50个
                    selected_samples = [s[1] for s in priority_samples[:50]]
                
                balanced_data.extend(selected_samples)
                print(f"  {fault_type}: {len(selected_samples)}个样本")
        
        print(f"平衡后总样本数: {len(balanced_data)}")
        return balanced_data
    
    def create_visualization(self, fault_counts, balanced_data, save_path='comprehensive_analysis'):
        """创建可视化"""
        print("正在生成可视化...")
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 原始数据分布
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
        
        # 3. 采样频率分布
        plt.subplot(2, 3, 3)
        sampling_counts = {}
        for sample in balanced_data:
            sampling_rate = sample['sampling_rate']
            if sampling_rate not in sampling_counts:
                sampling_counts[sampling_rate] = 0
            sampling_counts[sampling_rate] += 1
        
        plt.pie(sampling_counts.values(), labels=sampling_counts.keys(), autopct='%1.1f%%')
        plt.title('采样频率分布')
        
        # 4. 传感器分布
        plt.subplot(2, 3, 4)
        sensor_counts = {}
        for sample in balanced_data:
            sensor = sample['sensor']
            if sensor not in sensor_counts:
                sensor_counts[sensor] = 0
            sensor_counts[sensor] += 1
        
        plt.pie(sensor_counts.values(), labels=sensor_counts.keys(), autopct='%1.1f%%')
        plt.title('传感器分布')
        
        # 5. 数据利用率
        plt.subplot(2, 3, 5)
        utilization_rates = []
        for fault_type in fault_types:
            raw_count = fault_counts.get(fault_type, 0)
            balanced_count = balanced_counts.get(fault_type, 0)
            if raw_count > 0:
                utilization_rate = balanced_count / raw_count * 100
            else:
                utilization_rate = 0
            utilization_rates.append(utilization_rate)
        
        plt.bar(fault_types, utilization_rates)
        plt.title('数据利用率 (%)')
        plt.xticks(rotation=45)
        plt.ylabel('利用率 (%)')
        
        # 添加数值标签
        for i, v in enumerate(utilization_rates):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # 6. 故障尺寸分布
        plt.subplot(2, 3, 6)
        fault_size_counts = {}
        for sample in balanced_data:
            fault_size = sample['fault_size']
            if fault_size not in fault_size_counts:
                fault_size_counts[fault_size] = 0
            fault_size_counts[fault_size] += 1
        
        plt.bar(fault_size_counts.keys(), fault_size_counts.values())
        plt.title('故障尺寸分布')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"可视化结果已保存到 {save_path} 目录")

def main():
    """主函数"""
    print("=== 全面数据加载策略 ===")
    
    loader = ComprehensiveDataLoader('源域数据集')
    
    # 1. 加载所有数据源
    valid_samples, invalid_samples = loader.load_all_data_sources()
    
    # 2. 分析数据分布
    fault_counts, sampling_counts, sensor_counts = loader.analyze_data_distribution()
    
    # 3. 创建智能平衡数据集
    balanced_data = loader.create_balanced_dataset(strategy='intelligent')
    
    # 4. 创建可视化
    loader.create_visualization(fault_counts, balanced_data)
    
    print("\n=== 分析完成 ===")
    print(f"原始有效数据: {valid_samples}个样本")
    print(f"处理后数据: {len(balanced_data)}个样本")
    print(f"数据利用率: {len(balanced_data)/valid_samples*100:.2f}%")

if __name__ == "__main__":
    main()
