import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.io import loadmat
from scipy.stats import kurtosis, skew, entropy
from scipy.fftpack import fft, fftfreq
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.metrics import silhouette_score
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class IntelligentBearingAnalyzer:
    """智能轴承故障诊断分析器 - 改进版"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.features_df = None
        self.raw_signals = {}
        self.selected_features = None
        self.feature_importance_scores = {}
        
        # 轴承参数
        self.bearing_params = {
            'SKF6205': {'n': 9, 'd': 0.3126, 'D': 1.537},  # 驱动端
            'SKF6203': {'n': 9, 'd': 0.2656, 'D': 1.122}   # 风扇端
        }
        
        # 滑动窗口参数 - 基于源域数据分析优化
        # 源域数据规模：161文件，2110万数据点，平均13.1万点/文件
        # 类别分布：Ball(40文件), InnerRace(40文件), OuterRace(77文件), Normal(4文件)
        self.sliding_window_params = {
            'window_size': 2048,      # 窗口大小（约0.17秒@12kHz, 0.043秒@48kHz）
            'overlap_ratio': 0.75,    # 重叠比例75%，增加样本密度
            'min_windows_per_signal': 50,  # 每个信号最少提取的窗口数
            'max_windows_per_signal': 150  # 每个信号最多提取的窗口数（约60%利用率）
        }
        
        # 数据利用策略说明：
        # - 目标利用率：约60%（平衡样本数量与泛化能力）
        # - 预期样本数：Ball(6000), InnerRace(6000), OuterRace(9750), Normal(600)
        # - 依据：基于源域数据分析，确保每类≥5000样本且避免过拟合
        # - 参数设置：max_files_per_category=65, max_windows_per_signal=150
        
        # 创建结果文件夹
        self.results_dir = "enhanced_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"创建结果文件夹: {self.results_dir}")
    
    def calculate_fault_frequencies(self, rpm: float, bearing_type: str = 'SKF6205') -> Dict[str, float]:
        """计算轴承故障特征频率"""
        fr = rpm / 60  # 转频
        params = self.bearing_params[bearing_type]
        n, d, D = params['n'], params['d'], params['D']
        
        # 故障特征频率计算
        bpfo = fr * n / 2 * (1 - d / D)  # 外圈故障
        bpfi = fr * n / 2 * (1 + d / D)  # 内圈故障
        bsf = fr * D / d * (1 - (d / D) ** 2)  # 滚动体故障
        ftf = fr / 2 * (1 - d / D)  # 保持架故障
        
        return {'BPFO': bpfo, 'BPFI': bpfi, 'BSF': bsf, 'FTF': ftf, 'FR': fr}
    
    def assess_signal_quality(self, signal_data: np.ndarray, fs: float) -> Dict[str, float]:
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
        freqs = np.fft.fftfreq(len(signal_data), 1/fs)
        high_freq_mask = np.abs(freqs) > fs/4
        noise_power = np.mean(np.abs(fft_data[high_freq_mask])**2)
        signal_power = np.mean(signal_data**2)
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
    
    def adaptive_signal_preprocessing(self, signal_data: np.ndarray, fs: float) -> np.ndarray:
        """自适应信号预处理"""
        # 1. 去除直流分量
        signal_data = signal_data - np.mean(signal_data)
        
        # 2. 评估噪声水平
        noise_level = np.std(signal_data[-1000:])  # 使用信号末尾估计噪声
        
        # 3. 自适应滤波
        if noise_level > 0.1 * np.std(signal_data):
            # 高噪声情况：使用低通滤波
            nyquist = fs / 2
            cutoff = min(fs/4, 2000)  # 截止频率
            sos = signal.butter(4, cutoff/nyquist, btype='low', output='sos')
            signal_data = signal.sosfilt(sos, signal_data)
        
        # 4. 异常值处理
        threshold = 5 * np.std(signal_data)
        signal_data = np.clip(signal_data, -threshold, threshold)
        
        return signal_data
    
    def extract_sliding_windows(self, signal_data: np.ndarray, fs: float) -> List[np.ndarray]:
        """使用滑动窗口技术从单个信号中提取多个样本"""
        window_size = self.sliding_window_params['window_size']
        overlap_ratio = self.sliding_window_params['overlap_ratio']
        min_windows = self.sliding_window_params['min_windows_per_signal']
        max_windows = self.sliding_window_params['max_windows_per_signal']
        
        # 计算步长
        step_size = int(window_size * (1 - overlap_ratio))
        
        # 确保信号长度足够
        if len(signal_data) < window_size:
            # 如果信号太短，进行零填充
            padded_signal = np.pad(signal_data, (0, window_size - len(signal_data)), 'constant')
            return [padded_signal]
        
        # 提取所有可能的窗口
        windows = []
        start_idx = 0
        
        while start_idx + window_size <= len(signal_data):
            window = signal_data[start_idx:start_idx + window_size]
            windows.append(window)
            start_idx += step_size
            
            # 限制最大窗口数
            if len(windows) >= max_windows:
                break
        
        # 确保最少窗口数
        if len(windows) < min_windows and len(signal_data) >= window_size:
            # 如果窗口数不够，调整步长
            total_length = len(signal_data) - window_size
            if total_length > 0:
                adjusted_step = total_length // (min_windows - 1) if min_windows > 1 else step_size
                adjusted_step = max(adjusted_step, window_size // 4)  # 最小步长为窗口大小的1/4
                
                windows = []
                start_idx = 0
                for i in range(min_windows):
                    if start_idx + window_size <= len(signal_data):
                        window = signal_data[start_idx:start_idx + window_size]
                        windows.append(window)
                        start_idx += adjusted_step
                    else:
                        # 最后一个窗口从末尾开始
                        window = signal_data[-window_size:]
                        if len(windows) == 0 or not np.array_equal(window, windows[-1]):
                            windows.append(window)
                        break
        
        # 质量检查：移除异常窗口
        valid_windows = []
        for window in windows:
            # 检查窗口是否包含有效信号
            if np.std(window) > 1e-6 and not np.any(np.isnan(window)) and not np.any(np.isinf(window)):
                valid_windows.append(window)
        
        return valid_windows if valid_windows else [windows[0]] if windows else []
    
    def extract_physics_informed_features(self, signal_data: np.ndarray, fs: float, 
                                        rpm: float, bearing_type: str = 'SKF6205') -> Dict[str, float]:
        """提取物理机理驱动的特征"""
        features = {}
        
        # 计算故障特征频率
        fault_freqs = self.calculate_fault_frequencies(rpm, bearing_type)
        
        # FFT分析
        fft_data = np.fft.fft(signal_data)
        fft_magnitude = np.abs(fft_data)
        freqs = np.fft.fftfreq(len(signal_data), 1/fs)
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
        
        # 1. 故障特征频率能量集中度
        for fault_type, fault_freq in fault_freqs.items():
            if fault_type == 'FR':  # 跳过转频
                continue
                
            # 基频及谐波能量
            total_fault_energy = 0
            for harmonic in range(1, 4):  # 前3个谐波
                target_freq = fault_freq * harmonic
                freq_tolerance = 2  # Hz
                
                freq_mask = (positive_freqs >= target_freq - freq_tolerance) & \
                           (positive_freqs <= target_freq + freq_tolerance)
                
                if np.any(freq_mask):
                    harmonic_energy = np.sum(positive_magnitude[freq_mask]**2)
                    total_fault_energy += harmonic_energy
                    features[f'{fault_type}_H{harmonic}_energy'] = harmonic_energy
                else:
                    features[f'{fault_type}_H{harmonic}_energy'] = 0
            
            # 故障频率能量集中度
            total_energy = np.sum(positive_magnitude**2)
            features[f'{fault_type}_energy_concentration'] = total_fault_energy / (total_energy + 1e-10)
        
        # 2. 冲击响应特征（基于包络分析）
        analytic_signal = signal.hilbert(signal_data)
        envelope = np.abs(analytic_signal)
        
        # 冲击强度
        envelope_peaks, _ = signal.find_peaks(envelope, height=np.mean(envelope) + 2*np.std(envelope))
        features['impact_intensity'] = len(envelope_peaks) / len(envelope) * fs
        
        if len(envelope_peaks) > 1:
            # 冲击间隔一致性
            peak_intervals = np.diff(envelope_peaks) / fs
            features['impact_regularity'] = 1 / (np.std(peak_intervals) + 1e-10)
        else:
            features['impact_regularity'] = 0
        
        # 3. 调制特征
        # AM调制深度
        envelope_ac = envelope - np.mean(envelope)
        am_depth = np.std(envelope_ac) / (np.mean(envelope) + 1e-10)
        features['am_modulation_depth'] = am_depth
        
        # FM调制特征
        instantaneous_freq = np.diff(np.unwrap(np.angle(analytic_signal))) * fs / (2*np.pi)
        features['fm_modulation_index'] = np.std(instantaneous_freq) / (np.mean(np.abs(instantaneous_freq)) + 1e-10)
        
        return features
    
    def extract_enhanced_time_domain_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """提取增强的时域特征"""
        features = {}
        
        # 基本统计特征
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['peak_to_peak'] = np.max(signal_data) - np.min(signal_data)
        features['abs_mean'] = np.mean(np.abs(signal_data))
        
        # 形状因子
        if features['abs_mean'] > 1e-10:
            features['shape_factor'] = features['rms'] / features['abs_mean']
            features['crest_factor'] = np.max(np.abs(signal_data)) / features['rms']
        else:
            features['shape_factor'] = 0
            features['crest_factor'] = 0
        
        # 高阶统计特征
        features['skewness'] = skew(signal_data)
        features['kurtosis'] = kurtosis(signal_data)
        
        # 能量特征
        features['energy'] = np.sum(signal_data**2)
        features['log_energy'] = np.log(features['energy'] + 1e-10)
        
        # 零交叉率
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
        
        # 波形指标
        features['impulse_factor'] = np.max(np.abs(signal_data)) / features['abs_mean'] if features['abs_mean'] > 1e-10 else 0
        features['margin_factor'] = np.max(np.abs(signal_data)) / (np.mean(np.sqrt(np.abs(signal_data)))**2 + 1e-10)
        
        return features
    
    def extract_enhanced_frequency_features(self, signal_data: np.ndarray, fs: float) -> Dict[str, float]:
        """提取增强的频域特征"""
        features = {}
        
        # FFT计算
        fft_data = np.fft.fft(signal_data)
        fft_magnitude = np.abs(fft_data)
        freqs = np.fft.fftfreq(len(signal_data), 1/fs)
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
        
        # 基本频域统计
        features['freq_mean'] = np.mean(positive_magnitude)
        features['freq_std'] = np.std(positive_magnitude)
        features['freq_rms'] = np.sqrt(np.mean(positive_magnitude**2))
        
        # 频谱形状特征
        total_power = np.sum(positive_magnitude)
        if total_power > 1e-10:
            features['spectral_centroid'] = np.sum(positive_freqs * positive_magnitude) / total_power
            
            # 频谱扩散度
            spectral_spread = np.sqrt(np.sum(((positive_freqs - features['spectral_centroid'])**2) * positive_magnitude) / total_power)
            features['spectral_spread'] = spectral_spread
            
            # 频谱偏度和峰度
            if spectral_spread > 1e-10:
                features['spectral_skewness'] = np.sum(((positive_freqs - features['spectral_centroid'])**3) * positive_magnitude) / (total_power * spectral_spread**3)
                features['spectral_kurtosis'] = np.sum(((positive_freqs - features['spectral_centroid'])**4) * positive_magnitude) / (total_power * spectral_spread**4)
            else:
                features['spectral_skewness'] = 0
                features['spectral_kurtosis'] = 0
        else:
            features['spectral_centroid'] = 0
            features['spectral_spread'] = 0
            features['spectral_skewness'] = 0
            features['spectral_kurtosis'] = 0
        
        # 频谱熵
        power_spectrum = positive_magnitude**2
        power_spectrum_normalized = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        features['spectral_entropy'] = entropy(power_spectrum_normalized + 1e-10)
        
        # 频带能量分布（优化为5个频带）
        freq_bands = [(0, fs/8), (fs/8, fs/4), (fs/4, fs/2), (fs/2, 3*fs/4), (3*fs/4, fs/2)]
        total_energy = np.sum(positive_magnitude**2)
        
        for i, (f_low, f_high) in enumerate(freq_bands):
            band_mask = (positive_freqs >= f_low) & (positive_freqs < f_high)
            band_energy = np.sum(positive_magnitude[band_mask]**2)
            features[f'band_{i+1}_energy_ratio'] = band_energy / (total_energy + 1e-10)
        
        return features
    
    def extract_comprehensive_features(self, signal_data: np.ndarray, fs: float, 
                                     rpm: float, bearing_type: str = 'SKF6205') -> Dict[str, float]:
        """提取综合特征集"""
        features = {}
        
        # 1. 增强时域特征
        time_features = self.extract_enhanced_time_domain_features(signal_data)
        features.update(time_features)
        
        # 2. 增强频域特征
        freq_features = self.extract_enhanced_frequency_features(signal_data, fs)
        features.update(freq_features)
        
        # 3. 物理机理驱动特征
        physics_features = self.extract_physics_informed_features(signal_data, fs, rpm, bearing_type)
        features.update(physics_features)
        
        return features
    
    def intelligent_feature_selection(self, features_df: pd.DataFrame, target_features: int = 50) -> List[str]:
        """基于故障机理的智能特征选择"""
        print(f"开始基于故障机理的智能特征选择，目标特征数: {target_features}")
        
        # 准备数据 - 排除所有元数据字段
        metadata_columns = [
            'label', 'filename', 'rpm', 'bearing_type', 'signal_quality',
            'sampling_frequency', 'sampling_rate_category', 'sensor_position',
            'data_source_path', 'data_source_directory', 'fault_severity',
            'load_condition', 'signal_length', 'signal_duration_seconds',
            'snr_db', 'dynamic_range', 'amplitude_score', 'length_score',
            # 滑动窗口技术添加的新字段
            'original_filename', 'window_index', 'total_windows', 
            'original_signal_length', 'original_signal_duration'
        ]
        feature_columns = [col for col in features_df.columns if col not in metadata_columns]
        
        # 调试信息：检查数据类型
        print(f"数据框总列数: {len(features_df.columns)}")
        print(f"元数据列数: {len(metadata_columns)}")
        print(f"特征列数: {len(feature_columns)}")
        
        # 检查是否有非数值列混入特征列
        non_numeric_cols = []
        for col in feature_columns:
            if features_df[col].dtype == 'object' or features_df[col].dtype.name.startswith('str'):
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            print(f"警告：发现非数值特征列: {non_numeric_cols}")
            # 从特征列中移除非数值列
            feature_columns = [col for col in feature_columns if col not in non_numeric_cols]
            print(f"移除非数值列后的特征列数: {len(feature_columns)}")
        
        # 确保特征列都是数值类型
        numeric_features_df = features_df[feature_columns].select_dtypes(include=[np.number])
        if len(numeric_features_df.columns) != len(feature_columns):
            print(f"进一步过滤后的数值特征列数: {len(numeric_features_df.columns)}")
            feature_columns = list(numeric_features_df.columns)
        
        X = features_df[feature_columns].values
        y = features_df['label'].values
        
        # 处理缺失值和无穷值
        X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. 基于故障机理的特征分类
        fault_mechanism_features = {
            # 故障特征频率相关（最重要）
            'fault_frequency': [f for f in feature_columns if any(x in f for x in ['BPFO', 'BPFI', 'BSF', 'FTF', 'energy_concentration'])],
            
            # 冲击响应特征（故障产生冲击）
            'impact_response': [f for f in feature_columns if any(x in f for x in ['impact', 'impulse', 'crest', 'kurtosis'])],
            
            # 调制特征（故障引起调制现象）
            'modulation': [f for f in feature_columns if any(x in f for x in ['modulation', 'am_', 'fm_'])],
            
            # 包络分析特征（解调分析）
            'envelope': [f for f in feature_columns if any(x in f for x in ['envelope', 'hilbert'])],
            
            # 经典故障诊断指标
            'classic_indicators': [f for f in feature_columns if any(x in f for x in ['shape_factor', 'clearance_factor', 'skewness'])],
            
            # 频域特征
            'frequency_domain': [f for f in feature_columns if any(x in f for x in ['freq', 'spectral', 'centroid', 'bandwidth'])],
            
            # 时域统计特征
            'time_domain': [f for f in feature_columns if any(x in f for x in ['mean', 'std', 'rms', 'peak', 'abs_mean'])]
        }
        
        print("故障机理特征分类结果:")
        for category, features in fault_mechanism_features.items():
            print(f"  {category}: {len(features)} 个特征")
        
        # 2. 基于方差的预筛选
        variances = np.var(X_scaled, axis=0)
        high_variance_mask = variances > 0.01
        
        # 3. 统计评分计算
        mi_scores = mutual_info_classif(X_scaled[:, high_variance_mask], y, random_state=42)
        f_scores, _ = f_classif(X_scaled[:, high_variance_mask], y)
        f_scores = np.nan_to_num(f_scores, nan=0)
        
        # 归一化统计评分
        mi_scores_norm = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-10)
        f_scores_norm = (f_scores - np.min(f_scores)) / (np.max(f_scores) - np.min(f_scores) + 1e-10)
        variance_scores_norm = (variances[high_variance_mask] - np.min(variances[high_variance_mask])) / \
                              (np.max(variances[high_variance_mask]) - np.min(variances[high_variance_mask]) + 1e-10)
        
        # 4. 故障机理权重分配
        mechanism_weights = {
            'fault_frequency': 3.0,      # 最高权重：故障特征频率
            'impact_response': 2.5,      # 高权重：冲击响应
            'modulation': 2.0,           # 较高权重：调制特征
            'envelope': 1.8,             # 中高权重：包络分析
            'classic_indicators': 1.5,   # 中等权重：经典指标
            'frequency_domain': 1.2,     # 中低权重：频域特征
            'time_domain': 1.0           # 基础权重：时域特征
        }
        
        # 5. 基于故障机理的特征选择
        selected_features = []
        high_variance_features = np.array(feature_columns)[high_variance_mask]
        
        # 为每个特征计算综合评分（统计评分 + 机理权重）
        feature_scores = {}
        
        for i, feature in enumerate(high_variance_features):
            # 基础统计评分
            base_score = 0.4 * mi_scores_norm[i] + 0.4 * f_scores_norm[i] + 0.2 * variance_scores_norm[i]
            
            # 故障机理权重
            mechanism_weight = 1.0  # 默认权重
            for category, features_in_category in fault_mechanism_features.items():
                if feature in features_in_category:
                    mechanism_weight = mechanism_weights[category]
                    break
            
            # 综合评分 = 统计评分 × 机理权重
            final_score = base_score * mechanism_weight
            feature_scores[feature] = {
                'base_score': base_score,
                'mechanism_weight': mechanism_weight,
                'final_score': final_score,
                'mi_score': mi_scores_norm[i],
                'f_score': f_scores_norm[i],
                'variance_score': variance_scores_norm[i]
            }
        
        # 6. 分层选择策略
        # 确保每个重要机理类别都有代表性特征
        min_features_per_category = {
            'fault_frequency': max(3, target_features // 8),    # 至少3个故障频率特征
            'impact_response': max(2, target_features // 12),   # 至少2个冲击特征
            'modulation': max(2, target_features // 15),        # 至少2个调制特征
            'classic_indicators': max(2, target_features // 15) # 至少2个经典指标
        }
        
        # 先保证重要类别的最小特征数
        for category, min_count in min_features_per_category.items():
            category_features = fault_mechanism_features[category]
            category_scores = [(f, feature_scores.get(f, {}).get('final_score', 0)) 
                             for f in category_features if f in feature_scores]
            category_scores.sort(key=lambda x: x[1], reverse=True)
            
            selected_from_category = min(min_count, len(category_scores))
            for i in range(selected_from_category):
                if category_scores[i][0] not in selected_features:
                    selected_features.append(category_scores[i][0])
        
        # 7. 填充剩余特征位置
        remaining_slots = target_features - len(selected_features)
        if remaining_slots > 0:
            # 按综合评分排序，选择剩余特征
            all_scores = [(f, scores['final_score']) for f, scores in feature_scores.items() 
                         if f not in selected_features]
            all_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i in range(min(remaining_slots, len(all_scores))):
                selected_features.append(all_scores[i][0])
        
        # 8. 保存详细的特征重要性信息
        selected_feature_info = {f: feature_scores[f] for f in selected_features if f in feature_scores}
        
        self.feature_importance_scores = {
            'features': selected_features,
            'feature_details': selected_feature_info,
            'mechanism_categories': fault_mechanism_features,
            'selection_strategy': 'fault_mechanism_driven'
        }
        
        print(f"\n故障机理驱动的特征选择完成:")
        print(f"  从 {len(feature_columns)} 个特征中选择了 {len(selected_features)} 个")
        print(f"  各机理类别选择情况:")
        for category, features_in_category in fault_mechanism_features.items():
            selected_in_category = [f for f in selected_features if f in features_in_category]
            print(f"    {category}: {len(selected_in_category)}/{len(features_in_category)} 个")
        
        return selected_features
    
    def load_and_process_data(self, max_files_per_category: int = 65,
                            min_quality_threshold: float = 0.45) -> pd.DataFrame:
        """智能数据加载和处理"""
        print("开始智能数据加载和处理...")
        
        all_features = []
        all_labels = []
        all_filenames = []
        quality_stats = {'total': 0, 'accepted': 0, 'rejected': 0}
        
        category_counts = {'N': 0, 'B': 0, 'IR': 0, 'OR': 0}
        
        # 遍历数据文件，只处理DE数据
        for root, dirs, files in os.walk(self.data_path):
            # 只处理DE数据目录，跳过FE数据
            if 'FE_data' in root:
                continue
                
            for file in files:
                if not file.endswith('.mat'):
                    continue
                    
                file_path = os.path.join(root, file)
                quality_stats['total'] += 1
                
                try:
                    # 确定标签
                    if 'OR' in file or 'OR' in root:
                        label = 'OR'
                    elif 'IR' in file or 'IR' in root:
                        label = 'IR'
                    elif 'B' in file or ('B' in root and 'BA' not in root):
                        label = 'B'
                    elif 'N' in file or 'Normal' in root:
                        label = 'N'
                    else:
                        continue
                    
                    # 检查类别数量限制
                    if category_counts[label] >= max_files_per_category:
                        continue
                    
                    # 加载mat文件
                    mat_data = loadmat(file_path)
                    
                    # 严格选择DE数据
                    signal_data = None
                    for key in mat_data.keys():
                        if 'DE_time' in key:
                            signal_data = mat_data[key].flatten()
                            break
                    
                    if signal_data is None:
                        print(f"警告: {file} 缺少DE数据，跳过")
                        continue
                    
                    # 获取转速和采样频率
                    rpm = 1797  # 默认转速
                    for key in mat_data.keys():
                        if 'RPM' in key:
                            rpm = float(mat_data[key][0])
                            break
                    
                    # 详细的采样频率和数据源信息
                    fs = 12000 if '12k' in root else 48000
                    sampling_rate_category = '12kHz' if '12k' in root else '48kHz'
                    
                    # 传感器位置和轴承类型
                    bearing_type = 'SKF6205'  # 所有DE数据都对应SKF6205轴承
                    sensor_position = 'DE'  # 驱动端
                    
                    # 数据源路径信息
                    data_source_path = file_path
                    data_source_directory = os.path.basename(root)
                    
                    # 故障严重程度（从文件名推断）
                    fault_severity = 'Unknown'
                    if any(x in file for x in ['007', '0.007']):
                        fault_severity = '0.007_inch'
                    elif any(x in file for x in ['014', '0.014']):
                        fault_severity = '0.014_inch'
                    elif any(x in file for x in ['021', '0.021']):
                        fault_severity = '0.021_inch'
                    elif label == 'N':
                        fault_severity = 'Normal'
                    
                    # 负载条件（从文件名或路径推断）
                    load_condition = 'Unknown'
                    if any(x in file for x in ['0HP', '0_HP']):
                        load_condition = '0_HP'
                    elif any(x in file for x in ['1HP', '1_HP']):
                        load_condition = '1_HP'
                    elif any(x in file for x in ['2HP', '2_HP']):
                        load_condition = '2_HP'
                    elif any(x in file for x in ['3HP', '3_HP']):
                        load_condition = '3_HP'
                    
                    # 信号质量评估
                    if len(signal_data) < 5000:
                        continue
                        
                    # 截取信号（自适应长度）
                    signal_length = min(60000, len(signal_data))  # 约5秒数据
                    signal_data = signal_data[:signal_length]
                    
                    # 质量评估
                    quality_metrics = self.assess_signal_quality(signal_data, fs)
                    
                    if quality_metrics['overall_quality'] < min_quality_threshold:
                        quality_stats['rejected'] += 1
                        print(f"质量不达标: {file} (质量评分: {quality_metrics['overall_quality']:.3f})")
                        continue
                    
                    # 自适应预处理
                    signal_data = self.adaptive_signal_preprocessing(signal_data, fs)
                    
                    # 使用滑动窗口技术提取多个样本
                    windows = self.extract_sliding_windows(signal_data, fs)
                    
                    print(f"从 {file} 提取了 {len(windows)} 个窗口样本")
                    
                    # 为每个窗口提取特征
                    for window_idx, window_data in enumerate(windows):
                        # 特征提取
                        features = self.extract_comprehensive_features(window_data, fs, rpm, bearing_type)
                        
                        # 添加详细元信息
                        features['label'] = label
                        features['filename'] = f"{file}_window_{window_idx}"  # 添加窗口索引
                        features['original_filename'] = file  # 保留原始文件名
                        features['window_index'] = window_idx  # 窗口索引
                        features['total_windows'] = len(windows)  # 总窗口数
                        features['rpm'] = rpm
                        features['bearing_type'] = bearing_type
                        features['signal_quality'] = quality_metrics['overall_quality']
                        
                        # 新增的详细元数据
                        features['sampling_frequency'] = fs
                        features['sampling_rate_category'] = sampling_rate_category
                        features['sensor_position'] = sensor_position
                        features['data_source_path'] = data_source_path
                        features['data_source_directory'] = data_source_directory
                        features['fault_severity'] = fault_severity
                        features['load_condition'] = load_condition
                        features['signal_length'] = len(window_data)  # 窗口长度
                        features['signal_duration_seconds'] = len(window_data) / fs
                        features['original_signal_length'] = len(signal_data)  # 原始信号长度
                        features['original_signal_duration'] = len(signal_data) / fs
                        
                        # 质量评估详细信息
                        features['snr_db'] = quality_metrics.get('snr', 0)
                        features['dynamic_range'] = quality_metrics.get('dynamic_range', 0)
                        features['amplitude_score'] = quality_metrics.get('amplitude_score', 0)
                        features['length_score'] = quality_metrics.get('length_score', 0)
                        
                        all_features.append(features)
                        all_labels.append(label)
                        all_filenames.append(f"{file}_window_{window_idx}")
                    
                    category_counts[label] += 1
                    quality_stats['accepted'] += 1
                    
                    print(f"处理完成: {file} (质量: {quality_metrics['overall_quality']:.3f}, 类别: {label})")
                    
                except Exception as e:
                    print(f"处理文件 {file} 时出错: {str(e)}")
                    continue
        
        # 创建DataFrame
        self.features_df = pd.DataFrame(all_features)
        
        # 打印统计信息
        print(f"\n数据加载完成:")
        print(f"总文件数: {quality_stats['total']}")
        print(f"接受文件数: {quality_stats['accepted']}")
        print(f"拒绝文件数: {quality_stats['rejected']}")
        print(f"接受率: {quality_stats['accepted']/quality_stats['total']*100:.1f}%")
        print(f"\n各类别数量:")
        for label, count in category_counts.items():
            print(f"  {label}: {count}")
        
        return self.features_df
    
    def save_processed_features(self, filename: str = "enhanced_bearing_features.csv"):
        """保存处理后的特征"""
        if self.features_df is not None:
            filepath = os.path.join(self.results_dir, filename)
            self.features_df.to_csv(filepath, index=False)
            print(f"特征数据已保存到: {filepath}")
            
            # 保存特征选择结果
            if self.selected_features is not None:
                # 包含所有元数据字段
                metadata_columns = [
                    'label', 'filename', 'rpm', 'bearing_type', 'signal_quality',
                    'sampling_frequency', 'sampling_rate_category', 'sensor_position',
                    'data_source_path', 'data_source_directory', 'fault_severity',
                    'load_condition', 'signal_length', 'signal_duration_seconds',
                    'snr_db', 'dynamic_range', 'amplitude_score', 'length_score'
                ]
                selected_df = self.features_df[self.selected_features + metadata_columns]
                selected_filepath = os.path.join(self.results_dir, "selected_" + filename)
                selected_df.to_csv(selected_filepath, index=False)
                print(f"选择的特征数据已保存到: {selected_filepath}")
        else:
            print("错误: 没有特征数据可保存")
    
    def run_feature_extraction_pipeline(self, max_files_per_category: int = 65,
                                      target_features: int = 50):
        """运行特征提取管道"""
        print("=" * 60)
        print("智能轴承故障诊断特征提取管道")
        print("=" * 60)
        
        # 1. 数据加载和处理
        print("\n步骤1: 智能数据加载和质量控制")
        features_df = self.load_and_process_data(max_files_per_category)
        
        if features_df.empty:
            print("错误: 没有有效数据被加载")
            return None
        
        # 2. 智能特征选择
        print(f"\n步骤2: 智能特征选择")
        metadata_columns = [
            'label', 'filename', 'original_filename', 'window_index', 'total_windows',
            'rpm', 'bearing_type', 'signal_quality',
            'sampling_frequency', 'sampling_rate_category', 'sensor_position',
            'data_source_path', 'data_source_directory', 'fault_severity',
            'load_condition', 'signal_length', 'signal_duration_seconds',
            'original_signal_length', 'original_signal_duration',
            'snr_db', 'dynamic_range', 'amplitude_score', 'length_score'
        ]
        print(f"原始特征数: {len([col for col in features_df.columns if col not in metadata_columns])}")
        
        self.selected_features = self.intelligent_feature_selection(features_df, target_features)
        
        # 3. 保存结果
        print(f"\n步骤3: 保存处理结果")
        self.save_processed_features()
        
        print(f"\n特征提取管道完成!")
        print(f"最终特征数: {len(self.selected_features)}")
        print(f"数据样本数: {len(features_df)}")
        
        return features_df


def main():
    """主函数"""
    # 设置数据路径
    data_path = r"数据集_xlsx\数据集_xlsx\源域数据集"
    
    # 创建分析器
    analyzer = IntelligentBearingAnalyzer(data_path)
    
    print("=" * 60)
    print("滑动窗口增强的轴承故障诊断特征提取")
    print("=" * 60)
    print(f"滑动窗口参数:")
    print(f"  窗口大小: {analyzer.sliding_window_params['window_size']} 样本点")
    print(f"  重叠比例: {analyzer.sliding_window_params['overlap_ratio']*100:.0f}%")
    print(f"  每信号最少窗口数: {analyzer.sliding_window_params['min_windows_per_signal']}")
    print(f"  每信号最多窗口数: {analyzer.sliding_window_params['max_windows_per_signal']}")
    print("=" * 60)
    
    # 运行特征提取管道（基于源域数据分析的合理比例设置）
    # 源域数据分析结果：Ball(40文件), InnerRace(40文件), OuterRace(77文件), Normal(4文件)
    # 目标：使用约60%的数据，确保每类获得足够样本数量
    features_df = analyzer.run_feature_extraction_pipeline(
        max_files_per_category=65,  # 每类最多65个文件（约60%的数据利用率）
        target_features=50          # 目标特征数50个
    )
    
    if features_df is not None:
        print(f"\n处理完成! 特征数据形状: {features_df.shape}")
        print(f"选择的特征: {len(analyzer.selected_features)}")
        
        # 显示滑动窗口统计信息
        print(f"\n滑动窗口数据增强统计:")
        original_files = features_df['original_filename'].nunique()
        total_windows = len(features_df)
        avg_windows_per_file = total_windows / original_files
        print(f"  原始文件数: {original_files}")
        print(f"  生成窗口样本数: {total_windows}")
        print(f"  平均每文件窗口数: {avg_windows_per_file:.1f}")
        
        # 各类别样本分布
        print(f"\n各类别样本分布:")
        label_counts = features_df['label'].value_counts()
        for label, count in label_counts.items():
            original_files_in_label = features_df[features_df['label'] == label]['original_filename'].nunique()
            print(f"  {label}: {count} 个样本 (来自 {original_files_in_label} 个原始文件)")
        
        # 显示特征重要性前10
        if analyzer.feature_importance_scores and 'features' in analyzer.feature_importance_scores:
            print(f"\n特征重要性排名前10:")
            features = analyzer.feature_importance_scores['features']
            feature_details = analyzer.feature_importance_scores.get('feature_details', {})
            
            for i in range(min(10, len(features))):
                feature_name = features[i]
                # 使用final_score作为评分
                score = feature_details.get(feature_name, {}).get('final_score', 0.0)
                mechanism_weight = feature_details.get(feature_name, {}).get('mechanism_weight', 1.0)
                print(f"  {i+1:2d}. {feature_name:<30} (评分: {score:.4f}, 机理权重: {mechanism_weight:.1f})")
    
    return analyzer, features_df


if __name__ == "__main__":
    analyzer, features_df = main()