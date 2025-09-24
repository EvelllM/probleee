#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务一扩展：特征选择和工程
基于增强特征数据进行深度特征选择，为后续建模提供最优特征集

主要功能：
1. 特征质量评估和清洗
2. 特征重要性分析
3. 特征选择策略实施
4. 生成最终建模数据集
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class FeatureSelector:
    """特征选择器类"""
    
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        self.selected_features = []
        self.feature_categories = {}
        
    def load_data(self):
        """加载原始增强特征数据"""
        print("=== 数据加载 ===")
        
        try:
            # 优先尝试加载Excel文件
            self.data = pd.read_excel("../shouDongChuLi/(hand)selected_enhanced_bearing_features.xlsx")
            print(f"成功加载Excel文件")
        except Exception as e:
            print(f"Excel文件读取失败，尝试CSV文件: {e}")
            
        
        print(f"数据形状: {self.data.shape}")
        print(f"列名: {list(self.data.columns)}")
        print(f"标签分布:\n{self.data['label'].value_counts()}")
        
        return True
    
    def initial_feature_cleaning(self):
        """初步特征清洗"""
        print("\n=== 初步特征清洗 ===")
        
        # 分离特征和标签
        exclude_cols =  ['label', 'filename', 'rpm', 'sampling_frequency', 'bearing_type', 'signal_length',
                        'sensor_position', 'data_source_path', 'data_source_directory', 'fault_severity',
                        'signal_duration_seconds']
        
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        self.X = self.data[feature_cols].copy()
        self.y = self.label_encoder.fit_transform(self.data['label'])
        
        print(f"原始特征数量: {self.X.shape[1]}")
        print(f"样本数量: {self.X.shape[0]}")
        
        # 1. 只保留数值特征
        numeric_features = self.X.select_dtypes(include=[np.number]).columns
        self.X = self.X[numeric_features]
        print(f"数值特征数量: {self.X.shape[1]}")
        
        # 2. 检查和处理缺失值
        missing_info = self.X.isnull().sum()
        if missing_info.sum() > 0:
            print(f"发现缺失值: {missing_info[missing_info > 0]}")
            # 用中位数填充缺失值
            self.X = self.X.fillna(self.X.median())
            print("已用中位数填充缺失值")
        
        # 3. 移除常数特征
        constant_features = []
        for col in self.X.columns:
            if self.X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"移除常数特征: {constant_features}")
            self.X = self.X.drop(columns=constant_features)
        
        # 4. 移除极端异常值特征（方差过大或过小）
        feature_stats = self.X.describe()
        extreme_features = []
        
        for col in self.X.columns:
            std_val = feature_stats.loc['std', col]
            mean_val = feature_stats.loc['mean', col]
            
            # 检查变异系数是否过大（>10）或过小（<0.001）
            if mean_val != 0:
                cv = std_val / abs(mean_val)
                if cv > 10 or cv < 0.001:
                    extreme_features.append(col)
        
        if extreme_features:
            print(f"移除极端变异特征: {extreme_features[:5]}...")  # 只显示前5个
            self.X = self.X.drop(columns=extreme_features)
        
        print(f"清洗后特征数量: {self.X.shape[1]}")
        
    def categorize_features(self):
        """按故障机理分类特征"""
        print("\n=== 特征分类（按故障机理） ===")
        
        # 定义特征类别（基于轴承故障诊断理论）
        self.feature_categories = {
            '时域统计特征': {
                'keywords': ['mean', 'std', 'rms', 'peak_to_peak', 'abs_mean', 'var'],
                'description': '基础统计量，反映信号幅值特性',
                'features': []
            },
            '时域形状特征': {
                'keywords': ['skewness', 'kurtosis', 'shape_factor', 'crest_factor', 
                           'impulse_factor', 'margin_factor'],
                'description': '信号形状特征，反映冲击和调制特性',
                'features': []
            },
            '频域统计特征': {
                'keywords': ['freq_mean', 'freq_std', 'freq_rms', 'spectral_centroid', 
                           'spectral_spread', 'spectral_skewness', 'spectral_kurtosis'],
                'description': '频域统计特征，反映频谱分布特性',
                'features': []
            },
            '频域能量特征': {
                'keywords': ['energy', 'log_energy', 'spectral_entropy', 'band_', '_energy_ratio'],
                'description': '频带能量分布，反映故障频率成分',
                'features': []
            },
            '故障特征频率': {
                'keywords': ['BPFO', 'BPFI', 'BSF', 'FTF', '_H1_', '_H2_', '_H3_', 
                           '_energy_concentration'],
                'description': '轴承故障特征频率及其谐波能量',
                'features': []
            },
            '调制和冲击特征': {
                'keywords': ['impact_intensity', 'impact_regularity', 'am_modulation', 
                           'fm_modulation', 'dynamic_range'],
                'description': '调制深度和冲击特征，反映故障严重程度',
                'features': []
            },
            '其他特征': {
                'keywords': ['zero_crossing', 'entropy', 'complexity'],
                'description': '其他辅助特征',
                'features': []
            }
        }
        
        # 将特征分配到各个类别
        for feature in self.X.columns:
            assigned = False
            for category, info in self.feature_categories.items():
                for keyword in info['keywords']:
                    if keyword.lower() in feature.lower():
                        info['features'].append(feature)
                        assigned = True
                        break
                if assigned:
                    break
            
            # 如果没有分配到任何类别，放入"其他特征"
            if not assigned:
                self.feature_categories['其他特征']['features'].append(feature)
        
        # 显示分类结果
        for category, info in self.feature_categories.items():
            if info['features']:
                print(f"{category}: {len(info['features'])}个特征")
                print(f"  描述: {info['description']}")
                if len(info['features']) <= 5:
                    print(f"  特征: {info['features']}")
                else:
                    print(f"  特征示例: {info['features'][:3]}...")
                print()
    
    def correlation_analysis(self):
        """相关性分析和高相关特征移除"""
        print("\n=== 相关性分析 ===")
        
        # 计算相关性矩阵
        corr_matrix = self.X.corr().abs()
        
        # 找出高相关性特征对（>0.95）
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []
        
        for column in upper_tri.columns:
            high_corr_features = upper_tri.index[upper_tri[column] > 0.95].tolist()
            for feature in high_corr_features:
                high_corr_pairs.append((feature, column, upper_tri.loc[feature, column]))
        
        print(f"发现 {len(high_corr_pairs)} 对高相关性特征 (>0.95)")
        
        # 移除高相关性特征（保留方差较大的）
        features_to_remove = set()
        for feat1, feat2, corr_val in high_corr_pairs:
            if feat1 not in features_to_remove and feat2 not in features_to_remove:
                # 保留方差较大的特征
                if self.X[feat1].var() >= self.X[feat2].var():
                    features_to_remove.add(feat2)
                else:
                    features_to_remove.add(feat1)
        
        if features_to_remove:
            print(f"移除高相关性特征: {len(features_to_remove)}个")
            self.X = self.X.drop(columns=list(features_to_remove))
            print(f"剩余特征数量: {self.X.shape[1]}")
        
        return list(features_to_remove)
    
    def univariate_feature_selection(self):
        """单变量特征选择"""
        print("\n=== 单变量特征选择 ===")
        
        # 使用F检验选择特征
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(self.X, self.y)
        
        # 获取特征得分
        feature_scores = pd.DataFrame({
            'feature': self.X.columns,
            'f_score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('f_score', ascending=False)
        
        print("Top 15 F检验得分特征:")
        print(feature_scores.head(15))
        
        # 选择显著性特征（p < 0.05）
        significant_features = feature_scores[feature_scores['p_value'] < 0.05]['feature'].tolist()
        print(f"显著性特征数量 (p < 0.05): {len(significant_features)}")
        
        self.feature_importance['f_test'] = feature_scores
        return significant_features
    
    def tree_based_feature_importance(self):
        """基于树模型的特征重要性分析"""
        print("\n=== 基于随机森林的特征重要性分析 ===")
        
        # 使用随机森林计算特征重要性
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(self.X, self.y)
        
        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 15 重要特征:")
        print(feature_importance.head(15))
        
        # 计算累计重要性
        feature_importance['cumsum_importance'] = feature_importance['importance'].cumsum()
        
        # 选择累计重要性达到95%的特征
        n_features_95 = len(feature_importance[feature_importance['cumsum_importance'] <= 0.95]) + 1
        selected_features_95 = feature_importance.head(n_features_95)['feature'].tolist()
        
        print(f"累计重要性95%需要 {n_features_95} 个特征")
        
        # 选择累计重要性达到90%的特征（更严格）
        n_features_90 = len(feature_importance[feature_importance['cumsum_importance'] <= 0.90]) + 1
        selected_features_90 = feature_importance.head(n_features_90)['feature'].tolist()
        
        print(f"累计重要性90%需要 {n_features_90} 个特征")
        
        self.feature_importance['random_forest'] = feature_importance
        
        return selected_features_90, selected_features_95
    
    def recursive_feature_elimination(self):
        """递归特征消除"""
        print("\n=== 递归特征消除 ===")
        
        # 使用交叉验证的递归特征消除
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 设置合理的特征数量范围
        min_features = max(10, int(self.X.shape[1] * 0.1))  # 至少10个特征
        max_features = min(50, int(self.X.shape[1] * 0.8))  # 最多50个特征
        
        rfecv = RFECV(
            estimator=rf,
            step=1,
            cv=5,
            scoring='accuracy',
            min_features_to_select=min_features,
            n_jobs=-1
        )
        
        print(f"RFECV特征选择范围: {min_features} - {max_features}")
        
        # 如果特征太多，先用RFE减少到合理数量
        if self.X.shape[1] > max_features:
            print(f"特征数量过多，先用RFE减少到{max_features}个")
            rfe = RFE(estimator=rf, n_features_to_select=max_features)
            X_reduced = rfe.fit_transform(self.X, self.y)
            selected_features_rfe = self.X.columns[rfe.support_].tolist()
            
            # 在减少后的特征上运行RFECV
            X_for_rfecv = self.X[selected_features_rfe]
            rfecv.fit(X_for_rfecv, self.y)
            optimal_features = np.array(selected_features_rfe)[rfecv.support_].tolist()
        else:
            rfecv.fit(self.X, self.y)
            optimal_features = self.X.columns[rfecv.support_].tolist()
        
        print(f"RFECV选择的最优特征数量: {len(optimal_features)}")
        
        # 获取交叉验证得分（兼容不同sklearn版本）
        try:
            cv_scores = rfecv.cv_results_['mean_test_score']
            print(f"最优交叉验证得分: {cv_scores.max():.4f}")
        except AttributeError:
            try:
                cv_scores = rfecv.grid_scores_
                print(f"最优交叉验证得分: {cv_scores.max():.4f}")
            except AttributeError:
                print("无法获取交叉验证得分")
                cv_scores = None
        
        self.feature_importance['rfecv'] = {
            'selected_features': optimal_features,
            'n_features': len(optimal_features),
            'cv_scores': cv_scores
        }
        
        return optimal_features
    
    def comprehensive_feature_selection(self):
        """综合特征选择策略"""
        print("\n=== 综合特征选择策略 ===")
        
        # 1. 单变量选择
        significant_features = self.univariate_feature_selection()
        
        # 2. 树模型重要性选择
        rf_features_90, rf_features_95 = self.tree_based_feature_importance()
        
        # 3. 递归特征消除
        rfecv_features = self.recursive_feature_elimination()
        
        # 4. 综合选择策略
        print("\n=== 特征选择结果汇总 ===")
        print(f"F检验显著特征: {len(significant_features)}")
        print(f"随机森林90%重要性: {len(rf_features_90)}")
        print(f"随机森林95%重要性: {len(rf_features_95)}")
        print(f"RFECV最优特征: {len(rfecv_features)}")
        
        # 策略1: 取交集（最严格）
        intersection_features = list(set(significant_features) & 
                                   set(rf_features_90) & 
                                   set(rfecv_features))
        
        # 策略2: 至少被两种方法选中
        all_features = significant_features + rf_features_90 + rfecv_features
        feature_counts = pd.Series(all_features).value_counts()
        consensus_features = feature_counts[feature_counts >= 2].index.tolist()
        
        # 策略3: 随机森林90% + RFECV的并集（平衡性能和数量）
        union_features = list(set(rf_features_90) | set(rfecv_features))
        
        print(f"\n特征选择策略结果:")
        print(f"交集策略: {len(intersection_features)} 个特征")
        print(f"共识策略 (≥2种方法): {len(consensus_features)} 个特征")
        print(f"并集策略 (RF90% ∪ RFECV): {len(union_features)} 个特征")
        
        # 选择共识策略作为最终结果（平衡性能和稳定性）
        if len(consensus_features) >= 15:  # 确保有足够的特征
            self.selected_features = consensus_features
            strategy_used = "共识策略"
        elif len(union_features) >= 15:
            self.selected_features = union_features
            strategy_used = "并集策略"
        else:
            self.selected_features = rf_features_95  # 回退到95%重要性
            strategy_used = "随机森林95%策略"
        
        print(f"\n最终选择策略: {strategy_used}")
        print(f"最终特征数量: {len(self.selected_features)}")
        
        return self.selected_features
    
    def analyze_selected_features(self):
        """分析选中的特征"""
        print("\n=== 选中特征分析 ===")
        
        # 按类别统计选中的特征
        selected_by_category = {}
        for category, info in self.feature_categories.items():
            selected_in_category = [f for f in info['features'] if f in self.selected_features]
            if selected_in_category:
                selected_by_category[category] = selected_in_category
                print(f"{category}: {len(selected_in_category)}/{len(info['features'])} 个特征被选中")
                if len(selected_in_category) <= 5:
                    print(f"  选中特征: {selected_in_category}")
                else:
                    print(f"  选中特征示例: {selected_in_category[:3]}...")
        
        # 特征重要性排序
        if 'random_forest' in self.feature_importance:
            rf_importance = self.feature_importance['random_forest']
            selected_importance = rf_importance[rf_importance['feature'].isin(self.selected_features)]
            selected_importance = selected_importance.sort_values('importance', ascending=False)
            
            print(f"\n选中特征重要性排序 (Top 10):")
            print(selected_importance.head(10)[['feature', 'importance']])
        
        return selected_by_category
    
    def generate_final_dataset(self):
        """生成最终的建模数据集"""
        print("\n=== 生成最终数据集 ===")
        
        # 定义所有元特征（与initial_feature_cleaning中的exclude_cols保持一致）
        meta_features = ['label', 'filename', 'rpm', 'sampling_frequency', 'bearing_type', 'signal_length',
                        'sensor_position', 'data_source_path', 'data_source_directory', 'fault_severity',
                        'signal_duration_seconds']
        
        # 创建最终数据集：所有元特征 + 选中的特征
        final_columns = meta_features + self.selected_features
        final_data = self.data[final_columns].copy()
        
        print(f"最终数据集形状: {final_data.shape}")
        print(f"包含元特征: {len(meta_features)} 个 - {meta_features}")
        print(f"包含选中特征: {len(self.selected_features)} 个")
        print(f"总列数: {len(final_columns)} 个")
        print(f"样本数量: {final_data.shape[0]}")
        print(f"标签分布:\n{final_data['label'].value_counts()}")
        
        return final_data
    
    def save_results(self, final_data, selected_by_category):
        """保存结果"""
        print("\n=== 保存结果 ===")
        
        # 1. 保存最终数据集
        output_path = '../shouDongChuLi/result/Final_selected_features_dataset.xlsx'
        final_data.to_excel(output_path, index=False)
        print(f"最终数据集已保存: {output_path}")
        
        # 2. 保存CSV版本（备用）
        csv_path = '../shouDongChuLi/result/Final_selected_features_dataset.csv'
        final_data.to_csv(csv_path, index=False)
        print(f"CSV版本已保存: {csv_path}")
        
        # 3. 保存特征选择报告
        report_path = '../shouDongChuLi/result/Feature_selection_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("特征选择报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"原始特征数量: {len(self.data.columns) - 8}\n")  # 减去非特征列
            f.write(f"最终特征数量: {len(self.selected_features)}\n")
            f.write(f"特征减少比例: {(1 - len(self.selected_features)/(len(self.data.columns) - 8))*100:.1f}%\n\n")
            
            f.write("按类别选中的特征:\n")
            for category, features in selected_by_category.items():
                f.write(f"\n{category} ({len(features)}个):\n")
                for feature in features:
                    f.write(f"  - {feature}\n")
            
            f.write(f"\n\n最终选中的所有特征:\n")
            for i, feature in enumerate(self.selected_features, 1):
                f.write(f"{i:2d}. {feature}\n")
        
        print(f"特征选择报告已保存: {report_path}")
        
        # 4. 保存特征重要性数据
        if 'random_forest' in self.feature_importance:
            importance_path = '../shouDongChuLi/result/Feature_importance_detailed.csv'
            self.feature_importance['random_forest'].to_csv(importance_path, index=False)
            print(f"详细特征重要性已保存: {importance_path}")
    
    def create_visualization(self):
        """创建可视化图表"""
        print("\n=== 创建可视化 ===")
        
        try:
            # 1. 特征重要性图
            if 'random_forest' in self.feature_importance:
                plt.figure(figsize=(12, 8))
                
                # 选中特征的重要性
                rf_importance = self.feature_importance['random_forest']
                selected_importance = rf_importance[rf_importance['feature'].isin(self.selected_features)]
                selected_importance = selected_importance.sort_values('importance', ascending=True)
                
                plt.barh(range(len(selected_importance)), selected_importance['importance'])
                plt.yticks(range(len(selected_importance)), selected_importance['feature'])
                plt.xlabel('特征重要性')
                plt.title('最终选中特征的重要性排序')
                plt.tight_layout()
                
                plt.savefig('../shouDongChuLi/result/selected_features_importance.png',
                           dpi=300, bbox_inches='tight')
                plt.close()
                print("特征重要性图已保存")
            
            # 2. 特征类别分布图
            category_counts = {}
            for category, info in self.feature_categories.items():
                selected_in_category = [f for f in info['features'] if f in self.selected_features]
                if selected_in_category:
                    category_counts[category] = len(selected_in_category)
            
            if category_counts:
                plt.figure(figsize=(10, 6))
                categories = list(category_counts.keys())
                counts = list(category_counts.values())
                
                plt.bar(categories, counts)
                plt.xlabel('特征类别')
                plt.ylabel('选中特征数量')
                plt.title('各类别选中特征分布')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                plt.savefig('../shouDongChuLi/result/Feature_category_distribution.png',
                           dpi=300, bbox_inches='tight')
                plt.close()
                print("特征类别分布图已保存")
                
        except Exception as e:
            print(f"可视化创建失败: {e}")
    
    def run_complete_selection(self):
        """运行完整的特征选择流程"""
        print("开始特征选择流程...")
        
        # 1. 数据加载
        if not self.load_data():
            return False
        
        # 2. 初步特征清洗
        self.initial_feature_cleaning()
        
        # 3. 特征分类
        self.categorize_features()
        
        # 4. 相关性分析
        removed_features = self.correlation_analysis()
        
        # 5. 综合特征选择
        selected_features = self.comprehensive_feature_selection()
        
        # 6. 分析选中特征
        selected_by_category = self.analyze_selected_features()
        
        # 7. 生成最终数据集
        final_data = self.generate_final_dataset()
        
        # 8. 保存结果
        self.save_results(final_data, selected_by_category)
        
        # 9. 创建可视化
        self.create_visualization()
        
        print("\n" + "="*60)
        print("特征选择流程完成！")
        print(f"原始特征数量: {len(self.data.columns) - 8}")
        print(f"最终特征数量: {len(self.selected_features)}")
        print(f"数据集已保存: final_selected_features_dataset.xlsx")
        print("="*60)
        
        return True

def main():
    """主函数"""
    selector = FeatureSelector()
    success = selector.run_complete_selection()
    
    if success:
        print("\n特征选择成功完成！")
        print("生成的文件:")
        print("1. final_selected_features_dataset.xlsx - 最终建模数据集")
        print("2. final_selected_features_dataset.csv - CSV备份")
        print("3. feature_selection_report.txt - 特征选择报告")
        print("4. feature_importance_detailed.csv - 详细特征重要性")
        print("5. selected_features_importance.png - 特征重要性图")
        print("6. feature_category_distribution.png - 特征类别分布图")
    else:
        print("特征选择失败，请检查数据文件路径")

if __name__ == "__main__":
    main()