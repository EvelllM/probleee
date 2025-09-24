#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q1特征可视化分析
基于筛选后的26个特征进行深度可视化分析

主要功能：
1. 特征重要性可视化
2. 特征分布分析
3. 特征相关性分析
4. 故障类别对比分析
5. 特征差异性分析
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
def setup_chinese_font():
    """设置中文字体显示"""
    # 直接设置matplotlib的字体参数
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置所有文本相关的字体
    plt.rcParams['font.serif'] = plt.rcParams['font.sans-serif']
    plt.rcParams['font.monospace'] = plt.rcParams['font.sans-serif']
    
    print("已设置中文字体支持")

# 设置中文字体
setup_chinese_font()

# 定义全局字体属性，确保所有文本都使用中文字体
FONT_PROPS = {
    'fontfamily': 'Microsoft YaHei',
    'fontsize': 12
}

# 定义中文字体设置函数
def set_chinese_text(ax, title=None, xlabel=None, ylabel=None):
    """为图表设置中文字体"""
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', fontfamily='Microsoft YaHei')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontfamily='Microsoft YaHei')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontfamily='Microsoft YaHei')
    
    # 设置刻度标签字体
    for label in ax.get_xticklabels():
        label.set_fontfamily('Microsoft YaHei')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Microsoft YaHei')

# 设置图表样式
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")
sns.set_palette("husl")

class FeatureVisualizer:
    """特征可视化分析器"""
    
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.feature_categories = {}
        self.selected_features = []
        
    def load_data(self):
        """加载筛选后的特征数据"""
        print("=== 加载筛选后的特征数据 ===")
        
        try:
            # 加载最终筛选的数据集

            self.data = pd.read_excel('../shouDongChuLi/result/Final_selected_features_dataset.xlsx')
            print(f"成功加载数据，形状: {self.data.shape}")
        except Exception as e:
            print(f"Excel文件读取失败，尝试CSV文件: {e}")
            try:
                self.data = pd.read_csv('./shouDongChuLi/result/Final_selected_features_dataset.csv')

                print(f"成功加载CSV数据，形状: {self.data.shape}")
            except Exception as e2:
                print(f"数据加载失败: {e2}")
                return False
        
        # 分离元数据和特征数据
        meta_columns = ['label', 'filename', 'rpm', 'sampling_frequency', 'bearing_type', 
                       'signal_length', 'sensor_position', 'data_source_path', 
                       'data_source_directory', 'fault_severity', 'signal_duration_seconds']
        
        # 获取特征列（排除元数据列）
        self.feature_names = [col for col in self.data.columns if col not in meta_columns]
        self.X = self.data[self.feature_names].copy()
        self.y = self.label_encoder.fit_transform(self.data['label'])
        
        print(f"特征数量: {len(self.feature_names)}")
        print(f"样本数量: {self.X.shape[0]}")
        print(f"标签类别: {list(self.label_encoder.classes_)}")
        print(f"标签分布:\n{self.data['label'].value_counts()}")
        
        return True
    
    def define_feature_categories(self):
        """定义特征类别（基于特征选择报告）"""
        print("\n=== 定义特征类别 ===")
        
        self.feature_categories = {
            '时域统计特征': ['freq_rms'],
            '时域形状特征': ['kurtosis', 'spectral_skewness', 'skewness', 'margin_factor'],
            '频域统计特征': ['spectral_centroid', 'spectral_spread'],
            '频域能量特征': [
                'BSF_energy_concentration', 'BPFI_energy_concentration', 'BPFO_energy_concentration',
                'BPFO_H2_energy', 'FTF_H3_energy', 'FTF_H1_energy', 'log_energy', 'energy',
                'spectral_entropy', 'band_1_energy_ratio', 'band_3_energy_ratio', 
                'BPFI_H3_energy', 'BPFI_H1_energy'
            ],
            '调制和冲击特征': ['am_modulation_depth', 'fm_modulation_index', 'dynamic_range'],
            '其他特征': ['zero_crossing_rate', 'signal_quality', 'snr_db']
        }
        
        # 验证所有特征都被分类
        all_categorized = []
        for features in self.feature_categories.values():
            all_categorized.extend(features)
        
        missing_features = set(self.feature_names) - set(all_categorized)
        if missing_features:
            print(f"未分类的特征: {missing_features}")
            self.feature_categories['其他特征'].extend(list(missing_features))
        
        # 显示分类结果
        for category, features in self.feature_categories.items():
            actual_features = [f for f in features if f in self.feature_names]
            print(f"{category}: {len(actual_features)}个特征")
            if len(actual_features) <= 5:
                print(f"  特征: {actual_features}")
            else:
                print(f"  特征示例: {actual_features[:3]}...")
    
    def plot_feature_importance(self):
        """绘制特征重要性图"""
        print("\n=== 绘制特征重要性图 ===")
        
        # 使用随机森林计算特征重要性
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(self.X, self.y)
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 1. 水平条形图
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = ax1.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        ax1.set_yticks(range(len(importance_df)))
        ax1.set_yticklabels(importance_df['feature'], fontsize=10)
        set_chinese_text(ax1, title='随机森林特征重要性排序', xlabel='特征重要性')
        ax1.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
            ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center', fontsize=8)
        
        # 2. 累计重要性图
        importance_sorted = importance_df.sort_values('importance', ascending=False)
        cumsum_importance = importance_sorted['importance'].cumsum()
        
        ax2.plot(range(1, len(cumsum_importance)+1), cumsum_importance, 'b-o', linewidth=2, markersize=4)
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80%累计重要性')
        ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90%累计重要性')
        ax2.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95%累计重要性')
        
        set_chinese_text(ax2, title='特征累计重要性曲线', xlabel='特征数量', ylabel='累计重要性')
        ax2.legend(prop={'family': 'Microsoft YaHei'})
        ax2.grid(True, alpha=0.3)
        
        # 标注关键点
        for threshold in [0.8, 0.9, 0.95]:
            n_features = len(cumsum_importance[cumsum_importance <= threshold]) + 1
            if n_features <= len(cumsum_importance):
                ax2.annotate(f'{n_features}个特征\n达到{threshold*100}%', 
                           xy=(n_features, threshold), xytext=(n_features+2, threshold-0.05),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                           fontsize=9, ha='center')
        
        plt.tight_layout()
        # plt.savefig('1特征重要性分析.png', dpi=300, bbox_inches='tight')
        # plt.show()
        
        return importance_df
    
    def plot_feature_distributions(self):
        """绘制特征分布图"""
        print("\n=== 绘制特征分布图 ===")
        
        # 选择重要性最高的12个特征进行分布分析
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X, self.y)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(12)['feature'].tolist()
        
        # 创建子图
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, feature in enumerate(top_features):
            ax = axes[i]
            
            # 为每个故障类别绘制分布
            for j, label in enumerate(self.label_encoder.classes_):
                mask = self.data['label'] == label
                data_subset = self.X.loc[mask, feature]
                
                ax.hist(data_subset, bins=30, alpha=0.6, label=label, density=True)
            
            ax.set_title(f'{feature}\n(重要性: {importance_df[importance_df["feature"]==feature]["importance"].iloc[0]:.4f})', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('特征值', fontsize=10)
            ax.set_ylabel('密度', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Top 12 重要特征的分布对比', fontsize=16, fontweight='bold', fontfamily='Microsoft YaHei')
        plt.tight_layout()
        # plt.savefig('1特征分布分析.png', dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_correlation_analysis(self):
        """绘制特征相关性分析"""
        print("\n=== 绘制特征相关性分析 ===")
        
        # 计算相关性矩阵
        corr_matrix = self.X.corr()
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # 1. 完整相关性热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax1)
        set_chinese_text(ax1, title='特征相关性热力图')
        
        # 2. 高相关性特征对
        # 找出高相关性特征对（>0.7）
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.7:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('correlation', key=abs, ascending=False)
            
            # 绘制高相关性特征对
            y_pos = range(len(high_corr_df))
            colors = ['red' if x > 0 else 'blue' for x in high_corr_df['correlation']]
            
            bars = ax2.barh(y_pos, high_corr_df['correlation'], color=colors, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([f"{row['feature1']}\nvs\n{row['feature2']}" 
                               for _, row in high_corr_df.iterrows()], fontsize=8)
            set_chinese_text(ax2, title='高相关性特征对 (|r| > 0.7)', xlabel='相关系数')
            ax2.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for bar, val in zip(bars, high_corr_df['correlation']):
                ax2.text(val + (0.02 if val > 0 else -0.02), bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center', ha='left' if val > 0 else 'right', fontsize=8)
        else:
            ax2.text(0.5, 0.5, '没有发现高相关性特征对\n(|r| > 0.7)', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('高相关性特征对分析', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('1特征相关性分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return high_corr_pairs
    
    def plot_category_analysis(self):
        """绘制特征类别分析"""
        print("\n=== 绘制特征类别分析 ===")
        
        # 计算每个类别的特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X, self.y)
        
        importance_dict = dict(zip(self.feature_names, rf.feature_importances_))
        
        category_importance = {}
        category_counts = {}
        
        for category, features in self.feature_categories.items():
            actual_features = [f for f in features if f in self.feature_names]
            if actual_features:
                importances = [importance_dict[f] for f in actual_features]
                category_importance[category] = {
                    'total': sum(importances),
                    'mean': np.mean(importances),
                    'max': max(importances),
                    'features': actual_features
                }
                category_counts[category] = len(actual_features)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 特征类别数量分布
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        wedges, texts, autotexts = ax1.pie(counts, labels=categories, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        set_chinese_text(ax1, title='特征类别数量分布')
        # 设置饼图标签字体
        for text in texts:
            text.set_fontfamily('Microsoft YaHei')
        
        # 2. 类别总重要性
        total_importances = [category_importance[cat]['total'] for cat in categories]
        bars1 = ax2.bar(categories, total_importances, color=colors, alpha=0.8)
        set_chinese_text(ax2, title='各类别特征总重要性', ylabel='总重要性')
        ax2.tick_params(axis='x', rotation=45)
        # 设置x轴标签字体
        for label in ax2.get_xticklabels():
            label.set_fontfamily('Microsoft YaHei')
        
        # 添加数值标签
        for bar, val in zip(bars1, total_importances):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. 类别平均重要性
        mean_importances = [category_importance[cat]['mean'] for cat in categories]
        bars2 = ax3.bar(categories, mean_importances, color=colors, alpha=0.8)
        set_chinese_text(ax3, title='各类别特征平均重要性', ylabel='平均重要性')
        ax3.tick_params(axis='x', rotation=45)
        # 设置x轴标签字体
        for label in ax3.get_xticklabels():
            label.set_fontfamily('Microsoft YaHei')
        
        # 添加数值标签
        for bar, val in zip(bars2, mean_importances):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 各类别最重要特征
        max_importances = [category_importance[cat]['max'] for cat in categories]
        bars3 = ax4.bar(categories, max_importances, color=colors, alpha=0.8)
        set_chinese_text(ax4, title='各类别最高特征重要性', ylabel='最高重要性')
        ax4.tick_params(axis='x', rotation=45)
        # 设置x轴标签字体
        for label in ax4.get_xticklabels():
            label.set_fontfamily('Microsoft YaHei')
        
        # 添加数值标签
        for bar, val in zip(bars3, max_importances):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('1特征类别分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return category_importance
    
    def plot_fault_comparison(self):
        """绘制故障类别对比分析"""
        print("\n=== 绘制故障类别对比分析 ===")
        
        # 选择最重要的8个特征进行对比
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X, self.y)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(8)['feature'].tolist()
        
        # 创建图表
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(top_features):
            ax = axes[i]
            
            # 为每个故障类别计算统计量
            feature_stats = []
            labels = []
            
            for label in self.label_encoder.classes_:
                mask = self.data['label'] == label
                data_subset = self.X.loc[mask, feature]
                feature_stats.append(data_subset.values)
                labels.append(label)
            
            # 绘制箱线图
            box_plot = ax.boxplot(feature_stats, labels=labels, patch_artist=True)
            
            # 设置颜色
            colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            set_chinese_text(ax, 
                           title=f'{feature}\n(重要性: {importance_df[importance_df["feature"]==feature]["importance"].iloc[0]:.4f})',
                           ylabel='特征值')
            ax.tick_params(axis='x', rotation=45)
            # 设置x轴标签字体
            for label in ax.get_xticklabels():
                label.set_fontfamily('Microsoft YaHei')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Top 8 重要特征的故障类别对比', fontsize=16, fontweight='bold', fontfamily='Microsoft YaHei')
        plt.tight_layout()
        plt.savefig('1故障类别对比分析.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dimensionality_reduction(self):
        """绘制降维可视化"""
        print("\n=== 绘制降维可视化 ===")
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. PCA降维
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.label_encoder.classes_)))
        for i, label in enumerate(self.label_encoder.classes_):
            mask = self.data['label'] == label
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]], 
                       label=label, alpha=0.6, s=30)
        
        set_chinese_text(ax1, 
                       title='PCA降维可视化',
                       xlabel=f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})',
                       ylabel=f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
        ax1.legend(prop={'family': 'Microsoft YaHei'})
        ax1.grid(True, alpha=0.3)
        
        # 2. t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        for i, label in enumerate(self.label_encoder.classes_):
            mask = self.data['label'] == label
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[i]], 
                       label=label, alpha=0.6, s=30)
        
        set_chinese_text(ax2, 
                       title='t-SNE降维可视化',
                       xlabel='t-SNE 1',
                       ylabel='t-SNE 2')
        ax2.legend(prop={'family': 'Microsoft YaHei'})
        ax2.grid(True, alpha=0.3)
        
        # 3. PCA解释方差比例
        pca_full = PCA(random_state=42)
        pca_full.fit(X_scaled)
        
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        ax3.plot(range(1, len(cumsum_var)+1), cumsum_var, 'b-o', linewidth=2, markersize=4)
        ax3.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80%解释方差')
        ax3.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90%解释方差')
        ax3.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95%解释方差')
        
        set_chinese_text(ax3, 
                       title='PCA累计解释方差',
                       xlabel='主成分数量',
                       ylabel='累计解释方差比例')
        ax3.legend(prop={'family': 'Microsoft YaHei'})
        ax3.grid(True, alpha=0.3)
        
        # 4. 特征贡献度（前两个主成分）
        feature_contrib = pd.DataFrame(
            pca.components_[:2].T,
            columns=['PC1', 'PC2'],
            index=self.feature_names
        )
        
        # 计算每个特征对前两个主成分的总贡献
        feature_contrib['total_contrib'] = np.sqrt(feature_contrib['PC1']**2 + feature_contrib['PC2']**2)
        feature_contrib = feature_contrib.sort_values('total_contrib', ascending=True)
        
        # 绘制特征贡献度
        y_pos = range(len(feature_contrib))
        bars = ax4.barh(y_pos, feature_contrib['total_contrib'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(feature_contrib))))
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(feature_contrib.index, fontsize=8)
        set_chinese_text(ax4, 
                       title='特征对PCA的贡献度',
                       xlabel='对前两个主成分的总贡献度')
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('1降维可视化分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return X_pca, X_tsne, pca, tsne
    
    def generate_summary_report(self, importance_df, high_corr_pairs, category_importance):
        """生成可视化分析总结报告"""
        print("\n=== 生成可视化分析总结报告 ===")
        
        report_path = '特征可视化分析报告.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("特征可视化分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本信息
            f.write("1. 数据基本信息\n")
            f.write("-" * 30 + "\n")
            f.write(f"总样本数: {self.X.shape[0]}\n")
            f.write(f"特征数量: {len(self.feature_names)}\n")
            f.write(f"故障类别: {list(self.label_encoder.classes_)}\n")
            f.write(f"各类别样本数:\n")
            for label, count in self.data['label'].value_counts().items():
                f.write(f"  {label}: {count}\n")
            f.write("\n")
            
            # 特征重要性分析
            f.write("2. 特征重要性分析\n")
            f.write("-" * 30 + "\n")
            f.write("Top 10 重要特征:\n")
            top_10 = importance_df.sort_values('importance', ascending=False).head(10)
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"  {i:2d}. {row['feature']:<25} {row['importance']:.6f}\n")
            f.write("\n")
            
            # 特征类别分析
            f.write("3. 特征类别分析\n")
            f.write("-" * 30 + "\n")
            for category, info in category_importance.items():
                f.write(f"{category}:\n")
                f.write(f"  特征数量: {len(info['features'])}\n")
                f.write(f"  总重要性: {info['total']:.6f}\n")
                f.write(f"  平均重要性: {info['mean']:.6f}\n")
                f.write(f"  最高重要性: {info['max']:.6f}\n")
                f.write(f"  包含特征: {', '.join(info['features'])}\n\n")
            
            # 相关性分析
            f.write("4. 特征相关性分析\n")
            f.write("-" * 30 + "\n")
            if high_corr_pairs:
                f.write(f"发现 {len(high_corr_pairs)} 对高相关性特征 (|r| > 0.7):\n")
                for pair in high_corr_pairs:
                    f.write(f"  {pair['feature1']} vs {pair['feature2']}: {pair['correlation']:.4f}\n")
            else:
                f.write("未发现高相关性特征对 (|r| > 0.7)\n")
            f.write("\n")
            
            # 建议
            f.write("5. 分析建议\n")
            f.write("-" * 30 + "\n")
            f.write("基于可视化分析结果的建议:\n")
            f.write("1. 重点关注前10个重要特征，它们对故障诊断贡献最大\n")
            f.write("2. 频域能量特征类别包含最多特征，是故障诊断的核心\n")
            f.write("3. 不同故障类别在重要特征上表现出明显差异，有利于分类\n")
            if high_corr_pairs:
                f.write("4. 存在高相关性特征，可考虑进一步特征选择优化\n")
            f.write("5. 降维可视化显示特征能够有效区分不同故障类别\n")
        
        print(f"分析报告已保存: {report_path}")
    
    def run_complete_analysis(self):
        """运行完整的可视化分析"""
        print("开始特征可视化分析...")
        
        # 1. 加载数据
        if not self.load_data():
            return
        
        # 2. 定义特征类别
        self.define_feature_categories()
        
        # 3. 特征重要性分析
        importance_df = self.plot_feature_importance()
        
        # 4. 特征分布分析
        self.plot_feature_distributions()
        
        # 5. 相关性分析
        high_corr_pairs = self.plot_correlation_analysis()
        
        # 6. 特征类别分析
        category_importance = self.plot_category_analysis()
        
        # 7. 故障类别对比
        self.plot_fault_comparison()
        
        # 8. 降维可视化
        self.plot_dimensionality_reduction()
        
        # 9. 生成总结报告
        self.generate_summary_report(importance_df, high_corr_pairs, category_importance)
        
        print("\n=== 特征可视化分析完成 ===")
        print("生成的文件:")
        # print("- 1特征重要性分析.png")
        # print("- 1特征分布分析.png")
        print("- 1特征相关性分析.png")
        print("- 1特征类别分析.png")
        print("- 1故障类别对比分析.png")
        print("- 1降维可视化分析.png")
        print("- 1特征可视化分析报告.txt")

if __name__ == "__main__":
    # 创建可视化分析器并运行完整分析
    visualizer = FeatureVisualizer()
    visualizer.run_complete_analysis()