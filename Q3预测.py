import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TargetDomainPredictor:
    def __init__(self, model_path, target_data_path):
        """
        初始化目标域预测器
        
        Args:
            model_path: 训练好的模型文件路径
            target_data_path: 目标域标准化数据文件路径
        """
        self.model_path = model_path
        self.target_data_path = target_data_path
        self.model = None
        self.target_data = None
        self.sample_ids = None
        self.predictions = None
        self.prediction_probabilities = None
        self.label_encoder = None
        
        # 创建结果保存目录
        self.results_dir = "Q3迁移学习预测结果-2"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            print("正在加载训练好的模型...")
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 检查模型数据结构
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.label_encoder = model_data.get('label_encoder')
                print(f"成功加载模型: {type(self.model).__name__}")
                
                # 如果有标签编码器，打印类别信息
                if self.label_encoder:
                    print(f"模型类别: {self.label_encoder.classes_}")
                else:
                    # 创建默认的标签编码器（基于源域数据的常见类别）
                    self.label_encoder = LabelEncoder()
                    # 假设的轴承故障类别
                    default_classes = ['Normal', 'Inner_Race', 'Outer_Race', 'Ball', 'Cage']
                    self.label_encoder.fit(default_classes)
                    print(f"使用默认类别: {default_classes}")
            else:
                # 如果直接是模型对象
                self.model = model_data
                print(f"成功加载模型: {type(self.model).__name__}")
                
                # 创建默认的标签编码器
                self.label_encoder = LabelEncoder()
                default_classes = ['Normal', 'Inner_Race', 'Outer_Race', 'Ball', 'Cage']
                self.label_encoder.fit(default_classes)
                print(f"使用默认类别: {default_classes}")
            
            return True
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def load_target_data(self):
        """加载目标域标准化数据"""
        try:
            print("正在加载目标域数据...")
            self.target_data = pd.read_csv(self.target_data_path)
            
            # 提取样本ID（第一列）
            if self.target_data.columns[0] == 'Unnamed: 0':
                # 如果第一列是索引列，样本ID在索引中
                self.sample_ids = self.target_data.iloc[:, 0].tolist()
                self.target_data = self.target_data.iloc[:, 1:]
            else:
                # 如果第一列就是样本ID
                self.sample_ids = self.target_data.iloc[:, 0].tolist()
                self.target_data = self.target_data.iloc[:, 1:]
            
            print(f"成功加载目标域数据: {self.target_data.shape}")
            print(f"样本ID: {self.sample_ids}")
            print(f"特征列: {list(self.target_data.columns)}")
            
            # 检查数据质量
            print(f"缺失值数量: {self.target_data.isnull().sum().sum()}")
            print(f"无穷值数量: {np.isinf(self.target_data.values).sum()}")
            
            return True
            
        except Exception as e:
            print(f"加载目标域数据失败: {e}")
            return False
    
    def predict_target_domain(self):
        """对目标域数据进行预测"""
        try:
            print("\n正在进行目标域预测...")
            
            # 确保数据是数值型
            X = self.target_data.values.astype(float)
            
            # 处理可能的异常值
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 进行预测
            self.predictions = self.model.predict(X)
            self.prediction_probabilities = self.model.predict_proba(X)
            
            # 转换预测结果为类别名称
            predicted_labels = self.label_encoder.inverse_transform(self.predictions)
            
            print(f"预测完成，共预测 {len(self.predictions)} 个样本")
            
            # 统计预测结果
            unique, counts = np.unique(predicted_labels, return_counts=True)
            print("\n预测结果统计:")
            for label, count in zip(unique, counts):
                percentage = count / len(predicted_labels) * 100
                print(f"  {label}: {count} 个样本 ({percentage:.1f}%)")
            
            return predicted_labels
            
        except Exception as e:
            print(f"预测失败: {e}")
            return None
    
    def analyze_prediction_confidence(self):
        """分析预测置信度"""
        if self.prediction_probabilities is None:
            print("没有预测概率数据")
            return None
        
        print("\n=== 预测置信度分析 ===")
        
        # 计算最大概率（置信度）
        max_probabilities = np.max(self.prediction_probabilities, axis=1)
        
        # 统计置信度分布
        confidence_stats = {
            '平均置信度': np.mean(max_probabilities),
            '置信度中位数': np.median(max_probabilities),
            '置信度标准差': np.std(max_probabilities),
            '最小置信度': np.min(max_probabilities),
            '最大置信度': np.max(max_probabilities)
        }
        
        for key, value in confidence_stats.items():
            print(f"{key}: {value:.4f}")
        
        # 置信度区间统计
        confidence_ranges = {
            '高置信度 (>0.8)': np.sum(max_probabilities > 0.8),
            '中等置信度 (0.6-0.8)': np.sum((max_probabilities > 0.6) & (max_probabilities <= 0.8)),
            '低置信度 (0.4-0.6)': np.sum((max_probabilities > 0.4) & (max_probabilities <= 0.6)),
            '很低置信度 (<=0.4)': np.sum(max_probabilities <= 0.4)
        }
        
        print("\n置信度区间分布:")
        total_samples = len(max_probabilities)
        for range_name, count in confidence_ranges.items():
            percentage = count / total_samples * 100
            print(f"  {range_name}: {count} 个样本 ({percentage:.1f}%)")
        
        return confidence_stats, confidence_ranges
    
    def create_visualizations(self, predicted_labels):
        """创建预测结果可视化"""
        print("\n=== 生成可视化图表 ===")
        
        # 创建图表
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 预测类别分布饼图
        ax1 = plt.subplot(2, 3, 1)
        unique_labels, counts = np.unique(predicted_labels, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        wedges, texts, autotexts = ax1.pie(counts, labels=unique_labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('目标域预测类别分布', fontsize=14, fontweight='bold')
        
        # 2. 预测类别分布柱状图
        ax2 = plt.subplot(2, 3, 2)
        bars = ax2.bar(unique_labels, counts, color=colors, alpha=0.7)
        ax2.set_title('预测类别数量分布', fontsize=14, fontweight='bold')
        ax2.set_ylabel('样本数量')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 3. 置信度分布直方图
        ax3 = plt.subplot(2, 3, 3)
        max_probabilities = np.max(self.prediction_probabilities, axis=1)
        ax3.hist(max_probabilities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('预测置信度分布', fontsize=14, fontweight='bold')
        ax3.set_xlabel('置信度')
        ax3.set_ylabel('频次')
        ax3.axvline(np.mean(max_probabilities), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(max_probabilities):.3f}')
        ax3.legend()
        
        # 4. 各类别置信度箱线图
        ax4 = plt.subplot(2, 3, 4)
        confidence_by_class = []
        class_labels = []
        
        for label in unique_labels:
            mask = predicted_labels == label
            class_confidence = max_probabilities[mask]
            confidence_by_class.append(class_confidence)
            class_labels.append(label)
        
        box_plot = ax4.boxplot(confidence_by_class, labels=class_labels, patch_artist=True)
        ax4.set_title('各类别预测置信度分布', fontsize=14, fontweight='bold')
        ax4.set_ylabel('置信度')
        ax4.tick_params(axis='x', rotation=45)
        
        # 为箱线图添加颜色
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 5. 各样本预测结果展示
        ax5 = plt.subplot(2, 3, 5)
        
        # 为每个类别分配数值
        label_to_num = {label: i for i, label in enumerate(unique_labels)}
        numeric_predictions = [label_to_num[label] for label in predicted_labels]
        
        # 使用样本ID作为X轴标签，展示每个文件的预测结果
        ax5.scatter(range(len(self.sample_ids)), numeric_predictions, c=max_probabilities, 
                   cmap='viridis', alpha=0.7, s=50)
        ax5.set_title('各样本预测结果', fontsize=14, fontweight='bold')
        ax5.set_xlabel('样本文件')
        ax5.set_ylabel('预测类别')
        ax5.set_xticks(range(len(self.sample_ids)))
        ax5.set_xticklabels(self.sample_ids, rotation=45)  # 显示A, B, C...P
        ax5.set_yticks(range(len(unique_labels)))
        ax5.set_yticklabels(unique_labels)
        ax5.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(ax5.collections[0], ax=ax5)
        cbar.set_label('置信度')
        
        # 6. 置信度区间统计
        ax6 = plt.subplot(2, 3, 6)
        confidence_ranges = {
            '高置信度\n(>0.8)': np.sum(max_probabilities > 0.8),
            '中等置信度\n(0.6-0.8)': np.sum((max_probabilities > 0.6) & (max_probabilities <= 0.8)),
            '低置信度\n(0.4-0.6)': np.sum((max_probabilities > 0.4) & (max_probabilities <= 0.6)),
            '很低置信度\n(<=0.4)': np.sum(max_probabilities <= 0.4)
        }
        
        range_names = list(confidence_ranges.keys())
        range_counts = list(confidence_ranges.values())
        range_colors = ['green', 'orange', 'yellow', 'red']
        
        bars = ax6.bar(range_names, range_counts, color=range_colors, alpha=0.7)
        ax6.set_title('置信度区间分布', fontsize=14, fontweight='bold')
        ax6.set_ylabel('样本数量')
        
        # 添加百分比标签
        total_samples = len(max_probabilities)
        for bar, count in zip(bars, range_counts):
            percentage = count / total_samples * 100
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f'{self.results_dir}/target_domain_prediction_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_prediction_results(self, predicted_labels):
        """保存预测结果到文件"""
        print("\n=== 保存预测结果 ===")
        
        try:
            # 创建结果DataFrame
            results_df = pd.DataFrame({
                'sample_id': self.sample_ids,
                'predicted_label': predicted_labels,
                'predicted_class_id': self.predictions,
                'confidence': np.max(self.prediction_probabilities, axis=1)
            })
            
            # 添加各类别的概率
            class_names = self.label_encoder.classes_
            for i, class_name in enumerate(class_names):
                results_df[f'prob_{class_name}'] = self.prediction_probabilities[:, i]
            
            # 保存详细结果
            results_path = f'{self.results_dir}/target_domain_predictions.csv'
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            print(f"详细预测结果已保存到: {results_path}")
            
            # 创建汇总报告
            report_path = f'{self.results_dir}/prediction_summary_report.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("目标域迁移学习预测结果汇总报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型文件: {self.model_path}\n")
                f.write(f"目标域数据: {self.target_data_path}\n")
                f.write(f"总样本数: {len(predicted_labels)}\n\n")
                
                f.write("预测类别分布:\n")
                f.write("-" * 30 + "\n")
                unique_labels, counts = np.unique(predicted_labels, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    percentage = count / len(predicted_labels) * 100
                    f.write(f"{label}: {count} 个样本 ({percentage:.1f}%)\n")
                
                f.write("\n置信度统计:\n")
                f.write("-" * 30 + "\n")
                max_probabilities = np.max(self.prediction_probabilities, axis=1)
                f.write(f"平均置信度: {np.mean(max_probabilities):.4f}\n")
                f.write(f"置信度中位数: {np.median(max_probabilities):.4f}\n")
                f.write(f"置信度标准差: {np.std(max_probabilities):.4f}\n")
                f.write(f"最小置信度: {np.min(max_probabilities):.4f}\n")
                f.write(f"最大置信度: {np.max(max_probabilities):.4f}\n")
                
                f.write("\n置信度区间分布:\n")
                f.write("-" * 30 + "\n")
                confidence_ranges = {
                    '高置信度 (>0.8)': np.sum(max_probabilities > 0.8),
                    '中等置信度 (0.6-0.8)': np.sum((max_probabilities > 0.6) & (max_probabilities <= 0.8)),
                    '低置信度 (0.4-0.6)': np.sum((max_probabilities > 0.4) & (max_probabilities <= 0.6)),
                    '很低置信度 (<=0.4)': np.sum(max_probabilities <= 0.4)
                }
                
                total_samples = len(max_probabilities)
                for range_name, count in confidence_ranges.items():
                    percentage = count / total_samples * 100
                    f.write(f"{range_name}: {count} 个样本 ({percentage:.1f}%)\n")
            
            print(f"汇总报告已保存到: {report_path}")
            
            # 保存简化的标签文件（仅包含样本ID和预测标签）
            simple_results = pd.DataFrame({
                'sample_id': self.sample_ids,
                'predicted_label': predicted_labels
            })
            
            simple_path = f'{self.results_dir}/target_domain_labels.csv'
            simple_results.to_csv(simple_path, index=False, encoding='utf-8-sig')
            print(f"简化标签文件已保存到: {simple_path}")
            
            return results_df
            
        except Exception as e:
            print(f"保存结果失败: {e}")
            return None
    
    def run_complete_prediction(self):
        """运行完整的预测流程"""
        print("开始目标域迁移学习预测流程")
        print("=" * 50)
        
        # 1. 加载模型
        if not self.load_model():
            print("模型加载失败，终止预测流程")
            return False
        
        # 2. 加载目标域数据
        if not self.load_target_data():
            print("目标域数据加载失败，终止预测流程")
            return False
        
        # 3. 进行预测
        predicted_labels = self.predict_target_domain()
        if predicted_labels is None:
            print("预测失败，终止预测流程")
            return False
        
        # 4. 分析置信度
        self.analyze_prediction_confidence()
        
        # 5. 创建可视化
        self.create_visualizations(predicted_labels)
        
        # 6. 保存结果
        results_df = self.save_prediction_results(predicted_labels)
        
        print("\n" + "=" * 50)
        print("目标域迁移学习预测流程完成！")
        print(f"结果文件保存在: {self.results_dir}/")
        
        return True


def main():
    """主函数"""
    print("=" * 60)
    print("目标域迁移学习预测系统")
    print("=" * 60)
    
    # 文件路径
    #随机森林路径
    # model_path = r"d:\研究生\华为杯\pythonProject1\result-2\best_bearing_model.pkl"

    #stacking路径
    model_path = r"D:\研究生\华为杯\pythonProject1\result-22\rank_5_Stacking_Optimized_model.pkl"

    target_data_path = r"d:\研究生\华为杯\pythonProject1\Q3迁移学习\target_domain_features_standardized.csv"
    
    print(f"模型文件路径: {model_path}")
    print(f"数据文件路径: {target_data_path}")
    
    # 检查文件是否存在
    model_exists = os.path.exists(model_path)
    data_exists = os.path.exists(target_data_path)
    
    print(f"模型文件存在: {model_exists}")
    print(f"数据文件存在: {data_exists}")
    
    if not model_exists:
        print(f"错误: 模型文件不存在 - {model_path}")
        return
    
    if not data_exists:
        print(f"错误: 目标域数据文件不存在 - {target_data_path}")
        return
    
    try:
        # 创建预测器并运行
        print("\n正在初始化预测器...")
        predictor = TargetDomainPredictor(model_path, target_data_path)
        
        print("开始运行完整预测流程...")
        success = predictor.run_complete_prediction()
        
        if success:
            print("\n🎉 迁移学习预测成功完成！")
            print("📊 请查看生成的可视化图表和结果文件")
        else:
            print("\n❌ 迁移学习预测失败")
            
    except Exception as e:
        print(f"\n❌ 预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()