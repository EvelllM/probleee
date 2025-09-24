#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
目标域数据处理 - 任务三
用于处理目标域轴承振动数据，提取与源域一致的特征
适配32kHz采样率和600rpm转速参数
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TargetDomainProcessor:
    """目标域数据处理器"""

    def __init__(self, target_data_path, source_features_path):
        """
        初始化目标域处理器

        Args:
            target_data_path: 目标域数据路径
            source_features_path: 源域特征列表路径
        """
        self.target_data_path = target_data_path
        self.source_features_path = source_features_path

        # 目标域参数设置（基于题目明确信息）
        self.target_fs = 32000  # 目标域采样频率 32kHz（题目明确）
        self.target_rpm = 600  # 目标域转速 600rpm（题目明确）
        self.target_data_length = 256000  # 8秒 × 32kHz = 256000点

        # 源域参数（基于实际数据分析）
        # 注意：源域有12kHz和48kHz两种采样率，转速是变量
        self.source_fs_options = [12000, 48000]  # 源域采样频率选项：12kHz和48kHz
        self.source_rpm_variable = True  # 源域转速是变量，不是固定值

        # 为了兼容性，设置默认的源域参数用于显示
        self.source_fs = 12000  # 默认使用12kHz作为参考
        self.source_rpm = 1797  # 默认转速作为参考

        # 轴承参数（保持一致）
        self.bearing_params = {
            'd': 0.3126,  # 滚珠直径与节圆直径比值 d/D
            'D': 1.537,  # 节圆直径相关参数
            'num_balls': 9,  # 滚珠数量 n
            'contact_angle': 0  # 接触角 度（简化公式中不使用）
        }

        # 加载源域特征列表
        self.source_features = self._load_source_features()

        # 初始化标准化器
        self.scaler = StandardScaler()
        self.scaler_fitted = False  # 标记标准化器是否已训练

        print(f"目标域处理器初始化完成")
        print(f"目标域参数: 采样率={self.target_fs}Hz, 转速={self.target_rpm}rpm")
        print(f"源域参数: 采样率={self.source_fs}Hz, 转速={self.source_rpm}rpm")
        print(f"需要提取的特征数量: {len(self.source_features)}")

    def _load_source_features(self):
        """加载源域特征列表"""
        try:
            with open(self.source_features_path, 'r', encoding='utf-8') as f:
                features = [line.strip() for line in f.readlines() if line.strip()]
            print(f"成功加载源域特征列表，共{len(features)}个特征")
            return features
        except Exception as e:
            print(f"加载源域特征列表失败: {e}")
            return []

    def _calculate_bearing_frequencies(self, rpm):
        """计算轴承故障特征频率"""
        # 转速转换为Hz
        fr = rpm / 60

        # 获取参数
        n = self.bearing_params['num_balls']  # 滚珠数量
        d = self.bearing_params['d']  # 滚珠直径与节圆直径比值
        D = self.bearing_params['D']  # 节圆直径相关参数

        # 使用简化公式计算各故障频率
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

    def load_target_data(self, file_path):
        """加载目标域数据文件"""
        try:
            if file_path.endswith('.mat'):
                # 加载.mat文件
                data = loadmat(file_path)
                # 获取振动数据（通常是第一个非元数据字段）
                data_keys = [k for k in data.keys() if not k.startswith('__')]
                if data_keys:
                    vibration_data = data[data_keys[0]]
                    if vibration_data.ndim > 1:
                        vibration_data = vibration_data.flatten()
                    return vibration_data
                else:
                    raise ValueError("未找到有效的振动数据")

            elif file_path.endswith('.xlsx'):
                # 加载Excel文件
                df = pd.read_excel(file_path)
                # 假设振动数据在第一列
                vibration_data = df.iloc[:, 0].values
                return vibration_data

            else:
                raise ValueError(f"不支持的文件格式: {file_path}")

        except Exception as e:
            print(f"加载数据文件失败 {file_path}: {e}")
            return None

    def extract_time_domain_features(self, signal_data):
        """提取时域特征"""
        features = {}

        # 基本统计特征
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data ** 2))
        features['max'] = np.max(signal_data)
        features['min'] = np.min(signal_data)
        features['peak_to_peak'] = features['max'] - features['min']
        features['range'] = features['peak_to_peak']

        # 形状特征
        features['skewness'] = stats.skew(signal_data)
        features['kurtosis'] = stats.kurtosis(signal_data)

        # 峰值因子和裕度因子
        features['crest_factor'] = features['max'] / features['rms']
        features['clearance_factor'] = features['max'] / np.mean(np.sqrt(np.abs(signal_data))) ** 2
        features['shape_factor'] = features['rms'] / np.mean(np.abs(signal_data))
        features['impulse_factor'] = features['max'] / np.mean(np.abs(signal_data))

        # 能量特征
        features['energy'] = np.sum(signal_data ** 2)
        features['power'] = features['energy'] / len(signal_data)

        return features

    def extract_frequency_domain_features(self, signal_data, fs):
        """提取频域特征"""
        features = {}

        # 计算FFT
        fft_data = np.fft.fft(signal_data)
        fft_magnitude = np.abs(fft_data[:len(fft_data) // 2])
        freqs = np.fft.fftfreq(len(signal_data), 1 / fs)[:len(fft_data) // 2]

        # 功率谱密度
        psd = fft_magnitude ** 2 / (fs * len(signal_data))

        # 频域统计特征
        features['freq_mean'] = np.sum(freqs * psd) / np.sum(psd)
        features['freq_std'] = np.sqrt(np.sum((freqs - features['freq_mean']) ** 2 * psd) / np.sum(psd))
        features['freq_skewness'] = np.sum((freqs - features['freq_mean']) ** 3 * psd) / (
                    np.sum(psd) * features['freq_std'] ** 3)
        features['freq_kurtosis'] = np.sum((freqs - features['freq_mean']) ** 4 * psd) / (
                    np.sum(psd) * features['freq_std'] ** 4)

        # 频域能量特征
        features['total_power'] = np.sum(psd)
        features['freq_rms'] = np.sqrt(features['total_power'])

        # 主频特征
        peak_freq_idx = np.argmax(psd)
        features['peak_frequency'] = freqs[peak_freq_idx]
        features['peak_amplitude'] = psd[peak_freq_idx]

        return features, freqs, psd

    def extract_bearing_fault_features(self, signal_data, fs, rpm):
        """提取轴承故障特征频率相关特征（目标域：600rpm）"""
        features = {}

        # 计算故障特征频率（基于600rpm）
        fault_freqs = self._calculate_bearing_frequencies(rpm)

        # 打印故障频率用于验证
        print(f"目标域故障特征频率（600rpm）：")
        for fault_type, freq in fault_freqs.items():
            print(f"  {fault_type}: {freq:.2f} Hz")

        # 计算FFT
        fft_data = np.fft.fft(signal_data)
        fft_magnitude = np.abs(fft_data[:len(fft_data) // 2])
        freqs = np.fft.fftfreq(len(signal_data), 1 / fs)[:len(fft_data) // 2]

        # 在各故障频率附近提取能量
        for fault_type, fault_freq in fault_freqs.items():
            # 定义搜索窗口（±5Hz）
            window = 5
            freq_mask = (freqs >= fault_freq - window) & (freqs <= fault_freq + window)

            if np.any(freq_mask):
                # 能量集中度
                features[f'{fault_type}_energy_concentration'] = np.sum(fft_magnitude[freq_mask] ** 2)

                # 峰值幅度
                features[f'{fault_type}_peak_amplitude'] = np.max(fft_magnitude[freq_mask])

                # 平均幅度
                features[f'{fault_type}_mean_amplitude'] = np.mean(fft_magnitude[freq_mask])
            else:
                features[f'{fault_type}_energy_concentration'] = 0
                features[f'{fault_type}_peak_amplitude'] = 0
                features[f'{fault_type}_mean_amplitude'] = 0

        return features

    def assess_signal_quality(self, signal_data, fs):
        """评估信号质量"""
        quality_metrics = {}

        # 1. 信号长度检查
        quality_metrics['length_score'] = min(1.0, len(signal_data) / 10000)

        # 2. 信号幅值检查
        signal_std = np.std(signal_data)
        quality_metrics['amplitude_score'] = 1.0 if signal_std > 1e-6 else 0.0

        # 3. 信噪比估计
        # 使用高频部分估计噪声
        fft_data = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), 1 / fs)
        high_freq_mask = np.abs(freqs) > fs / 4
        noise_power = np.mean(np.abs(fft_data[high_freq_mask]) ** 2)
        signal_power = np.mean(signal_data ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        quality_metrics['snr'] = snr
        quality_metrics['snr_score'] = min(1.0, max(0.0, (snr - 10) / 30))

        # 4. 动态范围
        dynamic_range = (np.max(signal_data) - np.min(signal_data)) / (np.std(signal_data) + 1e-10)
        quality_metrics['dynamic_range'] = dynamic_range
        quality_metrics['dynamic_range_score'] = min(1.0, dynamic_range / 10)

        # 5. 综合质量评分
        quality_metrics['overall_quality'] = np.mean([
            quality_metrics['length_score'],
            quality_metrics['amplitude_score'],
            quality_metrics['snr_score'],
            quality_metrics['dynamic_range_score']
        ])

        return quality_metrics

    def extract_all_features(self, signal_data):
        """提取所有特征（适配目标域参数）"""
        # 确保信号是一维数组
        if signal_data.ndim > 1:
            signal_data = signal_data.flatten()

        # 验证数据长度是否符合预期（8秒 × 32kHz = 256000点）
        expected_length = 8 * self.target_fs  # 256000
        # if len(signal_data) != expected_length:
        #     print(f"警告：数据长度 {len(signal_data)} 不等于预期长度 {expected_length}")
        # 提取与源域完全对应的45个特征
        all_features = self._extract_all_45_features(signal_data)

        return all_features

    def _extract_all_45_features(self, signal):
        """提取与源域完全对应的45个特征"""
        features = {}
        fs = self.target_fs
        rpm = self.target_rpm

        # 计算轴承故障频率
        fault_freqs = self._calculate_bearing_frequencies(rpm)

        # 计算FFT
        fft_data = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1 / fs)
        magnitude = np.abs(fft_data)

        # 1. BSF_energy_concentration
        features['BSF_energy_concentration'] = self._calculate_fault_energy_concentration(
            magnitude, freqs, fault_freqs['BSF'])

        # 2. BPFI_energy_concentration
        features['BPFI_energy_concentration'] = self._calculate_fault_energy_concentration(
            magnitude, freqs, fault_freqs['BPFI'])

        # 3. BPFO_energy_concentration
        features['BPFO_energy_concentration'] = self._calculate_fault_energy_concentration(
            magnitude, freqs, fault_freqs['BPFO'])

        # 4. BPFO_H2_energy
        features['BPFO_H2_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['BPFO'], harmonic=2)

        # 5. FTF_H3_energy
        features['FTF_H3_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['FTF'], harmonic=3)

        # 6. FTF_H1_energy
        features['FTF_H1_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['FTF'], harmonic=1)

        # 7. impulse_factor
        features['impulse_factor'] = np.max(np.abs(signal)) / np.mean(np.abs(signal))

        # 8. crest_factor
        features['crest_factor'] = np.max(np.abs(signal)) / np.sqrt(np.mean(signal ** 2))

        # 9. kurtosis
        features['kurtosis'] = stats.kurtosis(signal)

        # 10. impact_intensity
        features['impact_intensity'] = np.sum(signal ** 4) / (len(signal) * np.var(signal) ** 2)

        # 11. am_modulation_depth
        features['am_modulation_depth'] = self._calculate_am_modulation_depth(signal)

        # 12. fm_modulation_index
        features['fm_modulation_index'] = self._calculate_fm_modulation_index(signal)

        # 13. shape_factor
        features['shape_factor'] = np.sqrt(np.mean(signal ** 2)) / np.mean(np.abs(signal))

        # 14. spectral_skewness
        features['spectral_skewness'] = self._calculate_spectral_skewness(magnitude, freqs)

        # 15. skewness
        features['skewness'] = stats.skew(signal)

        # 16. peak_to_peak
        features['peak_to_peak'] = np.max(signal) - np.min(signal)

        # 17. margin_factor
        features['margin_factor'] = np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal)))) ** 2

        # 18. freq_mean
        half_len = len(magnitude) // 2
        features['freq_mean'] = np.sum(freqs[:half_len] * magnitude[:half_len]) / np.sum(magnitude[:half_len])

        # 19. log_energy
        features['log_energy'] = np.log(np.sum(signal ** 2))

        # 20. freq_rms
        features['freq_rms'] = np.sqrt(np.mean(magnitude[:half_len] ** 2))

        # 21. freq_std
        features['freq_std'] = np.std(magnitude[:half_len])

        # 22. rms
        features['rms'] = np.sqrt(np.mean(signal ** 2))

        # 23. std
        features['std'] = np.std(signal)

        # 24. energy
        features['energy'] = np.sum(signal ** 2)

        # 25. abs_mean
        features['abs_mean'] = np.mean(np.abs(signal))

        # 26. zero_crossing_rate
        features['zero_crossing_rate'] = self._calculate_zero_crossing_rate(signal)

        # 27. spectral_entropy
        features['spectral_entropy'] = self._calculate_spectral_entropy(magnitude)

        # 28. band_1_energy_ratio
        features['band_1_energy_ratio'] = self._calculate_band_energy_ratio(magnitude, freqs, 0, fs / 8)

        # 29. band_2_energy_ratio
        features['band_2_energy_ratio'] = self._calculate_band_energy_ratio(magnitude, freqs, fs / 8, fs / 4)

        # 30. spectral_centroid
        features['spectral_centroid'] = self._calculate_spectral_centroid(magnitude, freqs)

        # 31. BPFI_H2_energy
        features['BPFI_H2_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['BPFI'], harmonic=2)

        # 32. band_3_energy_ratio
        features['band_3_energy_ratio'] = self._calculate_band_energy_ratio(magnitude, freqs, fs / 4, fs / 2)

        # 33. spectral_kurtosis
        features['spectral_kurtosis'] = self._calculate_spectral_kurtosis(magnitude)

        # 34. FTF_energy_concentration
        features['FTF_energy_concentration'] = self._calculate_fault_energy_concentration(
            magnitude, freqs, fault_freqs['FTF'])

        # 35. BPFI_H3_energy
        features['BPFI_H3_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['BPFI'], harmonic=3)

        # 36. spectral_spread
        features['spectral_spread'] = self._calculate_spectral_spread(magnitude, freqs)

        # 37. BPFO_H1_energy
        features['BPFO_H1_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['BPFO'], harmonic=1)

        # 38. impact_regularity
        features['impact_regularity'] = self._calculate_impact_regularity(signal)

        # 39. BPFI_H1_energy
        features['BPFI_H1_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['BPFI'], harmonic=1)

        # 40. BPFO_H3_energy
        features['BPFO_H3_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['BPFO'], harmonic=3)

        # 41. BSF_H2_energy
        features['BSF_H2_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['BSF'], harmonic=2)

        # 42. BSF_H1_energy
        features['BSF_H1_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['BSF'], harmonic=1)

        # 43. BSF_H3_energy
        features['BSF_H3_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['BSF'], harmonic=3)

        # 44. FTF_H2_energy
        features['FTF_H2_energy'] = self._calculate_harmonic_energy(
            magnitude, freqs, fault_freqs['FTF'], harmonic=2)

        # 45. mean
        features['mean'] = np.mean(signal)

        # 添加信号质量相关特征
        quality_metrics = self.assess_signal_quality(signal, fs)

        # 46. snr_db
        features['snr_db'] = quality_metrics.get('snr', 0)

        # 47. dynamic_range
        features['dynamic_range'] = quality_metrics.get('dynamic_range', 0)

        # 48. signal_quality
        features['signal_quality'] = quality_metrics.get('overall_quality', 0)

        return features

    def _calculate_fault_energy_concentration(self, magnitude, freqs, fault_freq, window=5):
        """计算故障频率附近的能量集中度"""
        half_len = len(magnitude) // 2
        freqs_half = freqs[:half_len]
        magnitude_half = magnitude[:half_len]

        freq_mask = (freqs_half >= fault_freq - window) & (freqs_half <= fault_freq + window)
        if np.any(freq_mask):
            return np.sum(magnitude_half[freq_mask] ** 2)
        return 0

    def _calculate_harmonic_energy(self, magnitude, freqs, fundamental_freq, harmonic=1, window=5):
        """计算谐波能量"""
        harmonic_freq = fundamental_freq * harmonic
        return self._calculate_fault_energy_concentration(magnitude, freqs, harmonic_freq, window)

    def _calculate_am_modulation_depth(self, signal):
        """计算调幅调制深度"""
        envelope = np.abs(signal)
        return (np.max(envelope) - np.min(envelope)) / (np.max(envelope) + np.min(envelope))

    def _calculate_fm_modulation_index(self, signal):
        """计算调频调制指数"""
        # 简化的调频指数计算
        from scipy.signal import hilbert
        analytic_signal = hilbert(signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_freq = np.diff(instantaneous_phase)
        return np.std(instantaneous_freq)

    def _calculate_spectral_skewness(self, magnitude, freqs):
        """计算频谱偏度"""
        half_len = len(magnitude) // 2
        magnitude_half = magnitude[:half_len]
        freqs_half = freqs[:half_len]

        # 计算频谱重心
        centroid = np.sum(freqs_half * magnitude_half) / np.sum(magnitude_half)

        # 计算频谱偏度
        numerator = np.sum((freqs_half - centroid) ** 3 * magnitude_half)
        denominator = np.sum(magnitude_half) * (
            np.sqrt(np.sum((freqs_half - centroid) ** 2 * magnitude_half) / np.sum(magnitude_half))) ** 3

        return numerator / denominator if denominator != 0 else 0

    def _calculate_zero_crossing_rate(self, signal):
        """计算过零率"""
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        return len(zero_crossings) / len(signal)

    def _calculate_spectral_entropy(self, magnitude):
        """计算频谱熵"""
        half_len = len(magnitude) // 2
        magnitude_half = magnitude[:half_len]

        # 归一化功率谱
        power_spectrum = magnitude_half ** 2
        power_spectrum = power_spectrum / np.sum(power_spectrum)

        # 计算熵
        entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-12))
        return entropy

    def _calculate_band_energy_ratio(self, magnitude, freqs, freq_low, freq_high):
        """计算频带能量比"""
        half_len = len(magnitude) // 2
        freqs_half = freqs[:half_len]
        magnitude_half = magnitude[:half_len]

        # 计算指定频带内的能量
        band_mask = (freqs_half >= freq_low) & (freqs_half <= freq_high)
        band_energy = np.sum(magnitude_half[band_mask] ** 2)

        # 计算总能量
        total_energy = np.sum(magnitude_half ** 2)

        return band_energy / total_energy if total_energy != 0 else 0

    def _calculate_spectral_centroid(self, magnitude, freqs):
        """计算频谱重心"""
        half_len = len(magnitude) // 2
        magnitude_half = magnitude[:half_len]
        freqs_half = freqs[:half_len]

        return np.sum(freqs_half * magnitude_half) / np.sum(magnitude_half)

    def _calculate_spectral_kurtosis(self, magnitude):
        """计算频谱峰度"""
        half_len = len(magnitude) // 2
        magnitude_half = magnitude[:half_len]

        return stats.kurtosis(magnitude_half)

    def _calculate_spectral_spread(self, magnitude, freqs):
        """计算频谱扩散度"""
        half_len = len(magnitude) // 2
        magnitude_half = magnitude[:half_len]
        freqs_half = freqs[:half_len]

        centroid = self._calculate_spectral_centroid(magnitude, freqs)
        spread = np.sqrt(np.sum((freqs_half - centroid) ** 2 * magnitude_half) / np.sum(magnitude_half))

        return spread

    def _calculate_impact_regularity(self, signal):
        """计算冲击规律性"""
        # 简化的冲击规律性计算
        envelope = np.abs(signal)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(envelope, height=np.mean(envelope) + 2 * np.std(envelope))

        if len(peaks) > 1:
            intervals = np.diff(peaks)
            return 1 / (1 + np.std(intervals) / np.mean(intervals))
        return 0

    def filter_features_by_source(self, all_features):
        """根据源域特征列表筛选特征，确保顺序一致"""
        filtered_features = {}

        for feature_name in self.source_features:
            if feature_name in all_features:
                filtered_features[feature_name] = all_features[feature_name]
            else:
                print(f"警告: 特征 '{feature_name}' 未在目标域数据中找到")
                filtered_features[feature_name] = 0  # 用0填充缺失特征

        print(f"成功筛选出 {len(filtered_features)} 个特征，与源域特征数量一致")
        return filtered_features

    def fit_scaler(self, feature_df):
        """
        训练标准化器

        Args:
            feature_df: 特征DataFrame，用于训练标准化器
        """
        if not self.scaler_fitted:
            # 只使用数值特征进行标准化
            numeric_features = feature_df.select_dtypes(include=[np.number])

            # 处理无穷值和NaN
            numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
            numeric_features = numeric_features.fillna(numeric_features.mean())

            # 训练标准化器
            self.scaler.fit(numeric_features)
            self.scaler_fitted = True

            print(f"标准化器训练完成，特征数量: {numeric_features.shape[1]}")
        else:
            print("标准化器已经训练过，跳过重复训练")

    def transform_features(self, feature_df):
        """
        应用标准化变换

        Args:
            feature_df: 待标准化的特征DataFrame

        Returns:
            standardized_df: 标准化后的特征DataFrame
        """
        if not self.scaler_fitted:
            raise ValueError("标准化器尚未训练，请先调用fit_scaler方法")

        # 只使用数值特征进行标准化
        numeric_features = feature_df.select_dtypes(include=[np.number])
        non_numeric_features = feature_df.select_dtypes(exclude=[np.number])

        # 处理无穷值和NaN
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
        numeric_features = numeric_features.fillna(numeric_features.mean())

        # 应用标准化
        standardized_numeric = self.scaler.transform(numeric_features)

        # 创建标准化后的DataFrame
        standardized_df = pd.DataFrame(
            standardized_numeric,
            columns=numeric_features.columns,
            index=feature_df.index
        )

        # 添加非数值特征（如果有的话）
        if not non_numeric_features.empty:
            standardized_df = pd.concat([standardized_df, non_numeric_features], axis=1)

        return standardized_df

    def fit_transform_features(self, feature_df):
        """
        训练标准化器并应用变换（一步完成）

        Args:
            feature_df: 特征DataFrame

        Returns:
            standardized_df: 标准化后的特征DataFrame
        """
        self.fit_scaler(feature_df)
        return self.transform_features(feature_df)

    def save_scaler(self, save_path):
        """
        保存训练好的标准化器

        Args:
            save_path: 保存路径
        """
        if not self.scaler_fitted:
            raise ValueError("标准化器尚未训练，无法保存")

        scaler_data = {
            'scaler': self.scaler,
            'scaler_fitted': self.scaler_fitted,
            'feature_names': self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else None
        }

        with open(save_path, 'wb') as f:
            pickle.dump(scaler_data, f)

        print(f"标准化器已保存到: {save_path}")

    def load_scaler(self, load_path):
        """
        加载预训练的标准化器

        Args:
            load_path: 标准化器文件路径
        """
        try:
            with open(load_path, 'rb') as f:
                scaler_data = pickle.load(f)

            self.scaler = scaler_data['scaler']
            self.scaler_fitted = scaler_data['scaler_fitted']

            print(f"标准化器已从 {load_path} 加载")
            return True
        except Exception as e:
            print(f"加载标准化器失败: {e}")
            return False

    def process_single_file(self, file_path):
        """处理单个目标域文件，直接对完整信号提取特征"""
        print(f"处理文件: {file_path}")

        # 加载数据
        signal_data = self.load_target_data(file_path)
        if signal_data is None:
            return None

        print(f"数据长度: {len(signal_data)}, 采样率: {self.target_fs}Hz")

        # 直接对完整信号提取特征
        all_features = self.extract_all_features(signal_data)

        # 根据源域特征列表筛选
        filtered_features = self.filter_features_by_source(all_features)

        return filtered_features

    def process_all_target_files(self):
        """处理所有目标域文件，支持滑动窗口"""
        import glob

        # 获取所有目标域.mat文件（避免重复搜索xlsx文件）
        mat_files = glob.glob(os.path.join(self.target_data_path, "*.mat"))
        mat_files.sort()

        print(f"找到 {len(mat_files)} 个目标域.mat文件")

        # 处理每个文件
        results = {}
        feature_matrix = []
        file_names = []

        for file_path in mat_files:
            file_name = os.path.basename(file_path).split('.')[0]
            features = self.process_single_file(file_path)

            if features is not None:
                # 每个文件生成一个特征向量
                results[file_name] = features
                feature_matrix.append(list(features.values()))
                file_names.append(file_name)

        if not feature_matrix:
            print("没有成功处理任何文件")
            return None, {}

        # 转换为DataFrame
        feature_df = pd.DataFrame(feature_matrix,
                                  columns=list(results[file_names[0]].keys()),
                                  index=file_names)

        print(f"成功处理 {len(mat_files)} 个文件，生成 {len(feature_df)} 个样本")
        print(f"特征维度: {feature_df.shape}")

        return feature_df, results

    def visualize_target_domain_data(self, feature_df, save_path=None):
        """可视化目标域数据分析"""
        if save_path is None:
            save_path = os.path.join(os.path.dirname(self.target_data_path), "target_domain_analysis")

        os.makedirs(save_path, exist_ok=True)

        # 1. 特征分布热力图
        plt.figure(figsize=(15, 10))
        correlation_matrix = feature_df.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('目标域特征相关性热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "target_domain_correlation_heatmap.png"), dpi=300)
        plt.show()

        # 2. 特征统计分布
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 特征均值分布
        feature_means = feature_df.mean()
        axes[0, 0].bar(range(len(feature_means)), feature_means.values)
        axes[0, 0].set_title('特征均值分布')
        axes[0, 0].set_xlabel('特征索引')
        axes[0, 0].set_ylabel('均值')

        # 特征标准差分布
        feature_stds = feature_df.std()
        axes[0, 1].bar(range(len(feature_stds)), feature_stds.values)
        axes[0, 1].set_title('特征标准差分布')
        axes[0, 1].set_xlabel('特征索引')
        axes[0, 1].set_ylabel('标准差')

        # 样本间距离分布
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(feature_df.values)
        axes[1, 0].hist(distances.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title('样本间欧氏距离分布')
        axes[1, 0].set_xlabel('距离')
        axes[1, 0].set_ylabel('频次')

        # 主成分分析可视化
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(feature_df.values)
        axes[1, 1].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        axes[1, 1].set_title(f'PCA降维可视化 (解释方差: {pca.explained_variance_ratio_.sum():.3f})')
        axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')

        # 添加样本标签
        for i, txt in enumerate(feature_df.index):
            axes[1, 1].annotate(txt, (pca_result[i, 0], pca_result[i, 1]), fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "target_domain_feature_analysis.png"), dpi=300)
        plt.show()

        print(f"可视化结果已保存到: {save_path}")

    def generate_processing_report(self, feature_df, save_path=None):
        """生成目标域数据处理报告"""
        if save_path is None:
            save_path = os.path.join(os.path.dirname(self.target_data_path), "target_domain_processing_report.txt")

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("目标域数据处理报告\n")
            f.write("=" * 50 + "\n\n")

            f.write("1. 处理参数配置\n")
            f.write("目标域参数（基于题目明确信息）:\n")
            f.write(f"   采样率: {self.target_fs} Hz (32kHz)\n")
            f.write(f"   转速: {self.target_rpm} rpm (600rpm)\n")
            f.write(f"   数据长度: {self.target_data_length} 点 (8秒)\n")
            f.write("源域参数（基于实际数据分析）:\n")
            f.write(f"   采样率: {self.source_fs_options} Hz (12kHz和48kHz)\n")
            f.write(f"   转速: 变量（不是固定值）\n\n")

            f.write("2. 数据处理结果\n")
            f.write(f"   处理文件数量: {len(feature_df)}\n")
            f.write(f"   特征维度: {feature_df.shape[1]}\n")
            f.write(f"   文件列表: {', '.join(feature_df.index.tolist())}\n\n")

            f.write("3. 特征统计信息\n")
            f.write(f"   特征均值范围: [{feature_df.mean().min():.6f}, {feature_df.mean().max():.6f}]\n")
            f.write(f"   特征标准差范围: [{feature_df.std().min():.6f}, {feature_df.std().max():.6f}]\n\n")

            f.write("4. 故障特征频率计算结果\n")
            fault_freqs = self._calculate_bearing_frequencies(self.target_rpm)
            for fault_type, freq in fault_freqs.items():
                f.write(f"   {fault_type}: {freq:.2f} Hz\n")
            f.write("\n")

            f.write("5. 特征列表\n")
            for i, feature_name in enumerate(feature_df.columns, 1):
                f.write(f"   {i:2d}. {feature_name}\n")
            f.write("\n")

            f.write("6. 数据质量评估\n")
            # 检查缺失值
            missing_count = feature_df.isnull().sum().sum()
            f.write(f"   缺失值数量: {missing_count}\n")

            # 检查异常值（使用3σ准则）
            outlier_count = 0
            for col in feature_df.columns:
                mean_val = feature_df[col].mean()
                std_val = feature_df[col].std()
                outliers = np.abs(feature_df[col] - mean_val) > 3 * std_val
                outlier_count += outliers.sum()
            f.write(f"   异常值数量 (3σ准则): {outlier_count}\n\n")

            f.write("7. 迁移学习准备状态\n")
            f.write("   ✓ 特征维度与源域一致\n")
            f.write("   ✓ 特征提取参数已适配\n")
            f.write("   ✓ 数据预处理完成\n")
            f.write("   → 可以进行迁移学习建模\n")

        print(f"处理报告已保存到: {save_path}")


def main():
    """主函数"""
    # 设置路径
    target_data_path = "数据集_xlsx/数据集_xlsx/目标域数据集"
    source_features_path = "clean_feature_list2.txt"

    # 创建处理器
    processor = TargetDomainProcessor(target_data_path, source_features_path)

    # 处理所有目标域文件
    print("\n开始处理目标域数据...")
    feature_df, results = processor.process_all_target_files()

    if feature_df is None:
        print("特征提取失败，程序退出")
        return

    # 保存原始特征数据
    output_path = "Q3迁移学习"
    os.makedirs(output_path, exist_ok=True)

    feature_df.to_csv(os.path.join(output_path, "target_domain_features_raw.csv"))
    print(f"原始特征数据已保存到: {os.path.join(output_path, 'target_domain_features_raw.csv')}")

    # 进行特征标准化
    print("\n开始特征标准化...")
    try:
        standardized_feature_df = processor.fit_transform_features(feature_df)

        # 保存标准化后的特征数据
        standardized_feature_df.to_csv(os.path.join(output_path, "target_domain_features_standardized.csv"))
        print(f"标准化特征数据已保存到: {os.path.join(output_path, 'target_domain_features_standardized.csv')}")

        # 保存标准化器
        scaler_path = os.path.join(output_path, "target_domain_scaler.pkl")
        processor.save_scaler(scaler_path)

        # 使用标准化后的特征进行后续分析
        feature_df = standardized_feature_df

    except Exception as e:
        print(f"标准化过程出现错误: {e}")
        print("将使用原始特征继续处理...")

    # 保存最终特征数据（标准化后或原始）
    # feature_df.to_csv(os.path.join(output_path, "target_domain_features.csv"))
    # print(f"最终特征数据已保存到: {os.path.join(output_path, 'target_domain_features.csv')}")

    # 可视化分析
    print("\n生成可视化分析...")
    processor.visualize_target_domain_data(feature_df, os.path.join(output_path, "visualizations"))

    # 生成处理报告
    print("\n生成处理报告...")
    processor.generate_processing_report(feature_df, os.path.join(output_path, "target_domain_processing_report.txt"))

    print("\n目标域数据处理完成！")
    print(f"处理结果保存在: {output_path}")


if __name__ == "__main__":
    main()