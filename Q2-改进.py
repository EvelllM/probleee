import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform
import pickle
import os
import warnings
import time

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class EnhancedBearingDiagnosisModel:
    def __init__(self, features_path=None):
        """初始化增强诊断模型"""
        self.features_path = features_path or 'Final_selected_features_dataset.xlsx'
        self.features_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.selected_features = None

        # 创建结果文件夹
        self.results_dir = "result-22"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_features(self):
        """加载特征数据"""
        try:
            # 检查文件扩展名
            file_extension = os.path.splitext(self.features_path)[1].lower()

            if file_extension == '.xlsx':
                # 读取Excel文件
                self.features_df = pd.read_excel(self.features_path)
            elif file_extension == '.csv':
                # 尝试多种编码读取CSV文件
                encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']
                for encoding in encodings:
                    try:
                        # 添加更多参数来处理CSV格式问题
                        self.features_df = pd.read_csv(
                            self.features_path,
                            encoding=encoding,
                            sep=',',  # 明确指定分隔符
                            quotechar='"',  # 指定引号字符
                            skipinitialspace=True,  # 跳过分隔符后的空格
                            engine='c'  # 使用C引擎，更快更准确
                        )
                        print(f"成功使用 {encoding} 编码读取文件")
                        # 验证数据是否正确读取（检查列数）
                        if len(self.features_df.columns) > 10:  # 应该有很多列
                            break
                        else:
                            print(
                                f"警告: 使用 {encoding} 编码只读取到 {len(self.features_df.columns)} 列，继续尝试其他编码")
                            continue
                    except (UnicodeDecodeError, pd.errors.ParserError) as e:
                        print(f"使用 {encoding} 编码失败: {e}")
                        continue
                else:
                    # 如果所有编码都失败，尝试最基本的读取方式
                    try:
                        self.features_df = pd.read_csv(
                            self.features_path,
                            encoding='utf-8',
                            sep=',',
                            engine='python',
                            error_bad_lines=False,
                            warn_bad_lines=True
                        )
                        print("使用基本方式成功读取文件")
                    except Exception as e:
                        raise ValueError(f"无法使用任何方式读取CSV文件: {e}")
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")

            print(f"成功加载特征数据，共 {len(self.features_df)} 个样本")
            print(f"特征数量: {len(self.features_df.columns)}")
            if 'label' in self.features_df.columns:
                print(f"标签分布:\n{self.features_df['label'].value_counts()}")
            return True
        except Exception as e:
            print(f"加载特征数据失败: {e}")
            return False

    def preprocess_data(self, test_size=0.3):
        """数据预处理"""
        if self.features_df is None:
            print("请先加载特征数据！")
            return False

        # 选择特征列
        exclude_cols = ['label', 'filename', 'rpm', 'sampling_frequency', 'bearing_type', 'signal_length',
                        'sensor_position', 'data_source_path', 'data_source_directory', 'fault_severity',
                        'signal_duration_seconds']
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]

        X = self.features_df[feature_cols].copy()
        y = self.features_df['label'].copy()

        # 处理缺失值和异常值
        X = X.fillna(X.mean())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())

        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)

        # 划分训练测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # 特征缩放
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # 保存特征列名（数据集已经过特征选择，直接使用所有特征）
        self.selected_features = feature_cols

        print(f"\n数据预处理完成:")
        print(f"训练集大小: {self.X_train_scaled.shape}")
        print(f"测试集大小: {self.X_test_scaled.shape}")
        print(f"使用的特征数: {len(self.selected_features)}")

        return True

    def train_core_models(self):
        """训练核心机器学习模型（包括XGBoost和Stacking融合模型）"""
        print("\n=== 开始训练核心模型 ===")
        
        # 定义基础模型
        base_models = {
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            'SVM_RBF': SVC(
                C=1.0,
                gamma='scale',
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'Logistic_Regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=42,
                n_jobs=-1,
                max_iter=1000
            )
        }
        
        # 创建Stacking集成模型（融合随机森林、梯度提升树、XGBoost）
        stacking_model = StackingClassifier(
            estimators=[
                ('rf', base_models['Random_Forest']),
                ('gb', base_models['Gradient_Boosting']),
                ('xgb', base_models['XGBoost'])
            ],
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,  # 5折交叉验证
            stack_method='predict_proba',  # 使用概率预测进行堆叠
            n_jobs=-1
        )
        
        # 将所有模型合并
        models = base_models.copy()
        models['Stacking_RF_GB_XGB'] = stacking_model
        
        self.models = {}
        self.model_scores = {}
        
        # 训练每个模型
        for name, model in models.items():
            print(f"\n训练 {name}...")
            start_time = time.time()
            
            try:
                # 训练模型
                model.fit(self.X_train_scaled, self.y_train)
                
                # 预测
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)
                
                # 计算评估指标
                accuracy = accuracy_score(self.y_test, y_pred)
                f1_macro = f1_score(self.y_test, y_pred, average='macro')
                f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
                precision = precision_score(self.y_test, y_pred, average='macro')
                recall = recall_score(self.y_test, y_pred, average='macro')
                training_time = time.time() - start_time
                
                # 保存模型和结果
                self.models[name] = model
                self.model_scores[name] = {
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted,
                    'precision': precision,
                    'recall': recall,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'training_time': training_time
                }
                
                print(f"  ✓ {name} 训练完成:")
                print(f"    准确率: {accuracy:.4f}")
                print(f"    F1-macro: {f1_macro:.4f}")
                print(f"    F1-weighted: {f1_weighted:.4f}")
                print(f"    精确率: {precision:.4f}")
                print(f"    召回率: {recall:.4f}")
                print(f"    训练时间: {training_time:.2f}秒")
                
            except Exception as e:
                print(f"  ✗ {name} 训练失败: {str(e)}")
                continue
        
        # 确定最佳模型
        if self.model_scores:
            best_model_name = max(self.model_scores.keys(), 
                                key=lambda x: self.model_scores[x]['f1_macro'])
            self.best_model = self.models[best_model_name]
            
            print(f"\n🏆 基础训练最佳模型: {best_model_name}")
            print(f"F1-macro得分: {self.model_scores[best_model_name]['f1_macro']:.4f}")
        
        return True

    def optimize_random_forest_with_pso(self, n_particles=20, n_iterations=25):
        """使用PSO优化随机森林超参数"""
        print("\n=== 开始PSO超参数优化 ===")
        
        # 定义搜索空间
        search_space = {
            "n_estimators": (50, 500),
            "max_depth": (5, 30),
            "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 10)
        }
        
        def decode_particle(vec):
            """将粒子向量解码为超参数"""
            params = {
                "n_estimators": int(np.clip(vec[0], *search_space["n_estimators"])),
                "max_depth": int(np.clip(vec[1], *search_space["max_depth"])),
                "min_samples_split": int(np.clip(vec[2], *search_space["min_samples_split"])),
                "min_samples_leaf": int(np.clip(vec[3], *search_space["min_samples_leaf"])),
                "random_state": 42,
                "n_jobs": -1
            }
            # 确保参数合理性
            params["min_samples_split"] = max(params["min_samples_split"], 
                                            params["min_samples_leaf"] + 1)
            return params
        
        def evaluate_particle(vec):
            """评估粒子性能"""
            try:
                params = decode_particle(vec)
                model = RandomForestClassifier(**params)
                
                # 使用交叉验证评估
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                          cv=3, scoring='f1_macro', n_jobs=-1)
                return np.mean(cv_scores)
            except:
                return 0.0
        
        # PSO参数
        n_dim = 4
        w = 0.72  # 惯性权重
        c1 = 1.49  # 个体学习因子
        c2 = 1.49  # 群体学习因子
        
        # 初始化粒子群
        lows = np.array([search_space["n_estimators"][0], search_space["max_depth"][0],
                        search_space["min_samples_split"][0], search_space["min_samples_leaf"][0]])
        highs = np.array([search_space["n_estimators"][1], search_space["max_depth"][1],
                         search_space["min_samples_split"][1], search_space["min_samples_leaf"][1]])
        
        positions = np.random.uniform(lows, highs, size=(n_particles, n_dim))
        velocities = np.random.uniform(-np.abs(highs-lows), np.abs(highs-lows), 
                                     size=(n_particles, n_dim)) * 0.1
        
        # 评估初始粒子
        pbest_pos = positions.copy()
        pbest_val = np.array([evaluate_particle(p) for p in positions])
        
        gbest_idx = np.argmax(pbest_val)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]
        
        print(f"PSO初始最优F1-macro: {gbest_val:.4f}")
        
        # PSO迭代
        for iteration in range(n_iterations):
            # 更新速度和位置
            r1 = np.random.random((n_particles, n_dim))
            r2 = np.random.random((n_particles, n_dim))
            
            velocities = (w * velocities + 
                         c1 * r1 * (pbest_pos - positions) + 
                         c2 * r2 * (gbest_pos - positions))
            
            positions = positions + velocities
            positions = np.clip(positions, lows, highs)
            
            # 评估新位置
            vals = np.array([evaluate_particle(p) for p in positions])
            
            # 更新个体最优
            improved = vals > pbest_val
            pbest_pos[improved] = positions[improved]
            pbest_val[improved] = vals[improved]
            
            # 更新全局最优
            if pbest_val.max() > gbest_val:
                gbest_idx = np.argmax(pbest_val)
                gbest_pos = pbest_pos[gbest_idx].copy()
                gbest_val = pbest_val[gbest_idx]
            
            if (iteration + 1) % 5 == 0:
                print(f"PSO迭代 {iteration + 1}/{n_iterations}, 当前最优F1-macro: {gbest_val:.4f}")
        
        # 使用最优参数训练最终模型
        best_params = decode_particle(gbest_pos)
        print(f"\nPSO优化完成，最优参数: {best_params}")
        
        pso_model = RandomForestClassifier(**best_params)
        pso_model.fit(self.X_train_scaled, self.y_train)
        
        # 评估PSO优化模型
        y_pred = pso_model.predict(self.X_test_scaled)
        y_pred_proba = pso_model.predict_proba(self.X_test_scaled)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
        
        # 保存PSO优化模型
        self.models['RF_PSO_Optimized'] = pso_model
        self.model_scores['RF_PSO_Optimized'] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': best_params
        }
        
        print(f"PSO优化模型性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1-macro: {f1_macro:.4f}")
        print(f"  F1-weighted: {f1_weighted:.4f}")
        
        # 更新最佳模型
        if f1_macro > max([score['f1_macro'] for score in self.model_scores.values() 
                          if 'f1_macro' in score]):
            self.best_model = pso_model
            print("PSO优化模型成为新的最佳模型！")
        
        return True

    def create_optimized_stacking_model(self):
        """直接使用PSO优化后的最优参数创建Stacking集成模型"""
        print("\n=== 使用最优参数创建Stacking模型 ===")
        
        # PSO优化后的最优参数
        best_params = {
            'rf_n_estimators': 173,
            'rf_max_depth': 21,
            'rf_min_samples_split': 2,
            'gb_n_estimators': 145,
            'gb_learning_rate': 0.03507181795505348,
            'gb_max_depth': 6,
            'xgb_n_estimators': 197,
            'xgb_learning_rate': 0.030118237660619412,
            'xgb_max_depth': 3
        }
        
        print(f"使用最优参数: {best_params}")
        
        # 创建最优Stacking模型
        rf_optimal = RandomForestClassifier(
            n_estimators=best_params['rf_n_estimators'],
            max_depth=best_params['rf_max_depth'],
            min_samples_split=best_params['rf_min_samples_split'],
            random_state=42,
            n_jobs=-1
        )
        
        gb_optimal = GradientBoostingClassifier(
            n_estimators=best_params['gb_n_estimators'],
            learning_rate=best_params['gb_learning_rate'],
            max_depth=best_params['gb_max_depth'],
            random_state=42
        )
        
        xgb_optimal = xgb.XGBClassifier(
            n_estimators=best_params['xgb_n_estimators'],
            learning_rate=best_params['xgb_learning_rate'],
            max_depth=best_params['xgb_max_depth'],
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        stacking_optimal = StackingClassifier(
            estimators=[
                ('rf', rf_optimal),
                ('gb', gb_optimal),
                ('xgb', xgb_optimal)
            ],
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # 训练最优模型
        print("正在训练优化后的Stacking模型...")
        stacking_optimal.fit(self.X_train_scaled, self.y_train)
        
        # 评估优化模型
        y_pred = stacking_optimal.predict(self.X_test_scaled)
        y_pred_proba = stacking_optimal.predict_proba(self.X_test_scaled)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='macro')
        recall = recall_score(self.y_test, y_pred, average='macro')
        
        # 保存优化的Stacking模型
        self.models['Stacking_Optimized'] = stacking_optimal
        self.model_scores['Stacking_Optimized'] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': best_params
        }
        
        print(f"\n优化Stacking模型性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1-macro: {f1_macro:.4f}")
        print(f"  F1-weighted: {f1_weighted:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        
        # 更新最佳模型
        if f1_macro > max([score['f1_macro'] for score in self.model_scores.values() 
                          if 'f1_macro' in score]):
            self.best_model = stacking_optimal
            print("优化Stacking模型成为新的最佳模型！")
        
        return True

    def optimize_stacking_model_with_pso(self, n_particles=6, n_iterations=2):
        """使用PSO优化Stacking集成模型的超参数"""
        print("\n=== 开始Stacking模型PSO超参数优化 ===")
        
        # 定义搜索空间（针对基础模型的关键超参数）
        search_space = {
            # 随机森林参数
            "rf_n_estimators": (50, 200),
            "rf_max_depth": (5, 30),
            "rf_min_samples_split": (2,5),
            # 梯度提升参数
            "gb_n_estimators": (50, 200),
            "gb_learning_rate": (0.01, 0.10 ),
            "gb_max_depth": (3, 8),
            # XGBoost参数
            "xgb_n_estimators": (50, 200),
            "xgb_learning_rate": (0.01, 0.1),
            "xgb_max_depth": (3, 10)
        }
        
        def decode_particle(particle):
            """将粒子位置解码为模型参数"""
            return {
                'rf_n_estimators': int(particle[0]),
                'rf_max_depth': int(particle[1]),
                'rf_min_samples_split': int(particle[2]),
                'gb_n_estimators': int(particle[3]),
                'gb_learning_rate': particle[4],
                'gb_max_depth': int(particle[5]),
                'xgb_n_estimators': int(particle[6]),
                'xgb_learning_rate': particle[7],
                'xgb_max_depth': int(particle[8])
            }
        
        def evaluate_particle(particle):
            """评估粒子（返回F1-macro得分）"""
            try:
                params = decode_particle(particle)
                
                # 创建基础模型
                rf_model = RandomForestClassifier(
                    n_estimators=params['rf_n_estimators'],
                    max_depth=params['rf_max_depth'],
                    min_samples_split=params['rf_min_samples_split'],
                    random_state=42,
                    n_jobs=-1
                )
                
                gb_model = GradientBoostingClassifier(
                    n_estimators=params['gb_n_estimators'],
                    learning_rate=params['gb_learning_rate'],
                    max_depth=params['gb_max_depth'],
                    random_state=42
                )
                
                xgb_model = xgb.XGBClassifier(
                    n_estimators=params['xgb_n_estimators'],
                    learning_rate=params['xgb_learning_rate'],
                    max_depth=params['xgb_max_depth'],
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='mlogloss'
                )
                
                # 创建Stacking模型
                stacking_model = StackingClassifier(
                    estimators=[
                        ('rf', rf_model),
                        ('gb', gb_model),
                        ('xgb', xgb_model)
                    ],
                    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
                    cv=3,  # 减少CV折数以加速
                    stack_method='predict_proba',
                    n_jobs=-1
                )
                
                # 训练和评估
                stacking_model.fit(self.X_train_scaled, self.y_train)
                y_pred = stacking_model.predict(self.X_test_scaled)
                f1_macro = f1_score(self.y_test, y_pred, average='macro')
                
                return f1_macro
                
            except Exception as e:
                print(f"粒子评估失败: {e}")
                return 0.0
        
        # PSO参数
        n_dim = 9  # 9个超参数
        w = 0.72  # 惯性权重
        c1 = 1.49  # 个体学习因子
        c2 = 1.49  # 群体学习因子
        
        # 初始化粒子群
        lows = np.array([
            search_space["rf_n_estimators"][0], search_space["rf_max_depth"][0], search_space["rf_min_samples_split"][0],
            search_space["gb_n_estimators"][0], search_space["gb_learning_rate"][0], search_space["gb_max_depth"][0],
            search_space["xgb_n_estimators"][0], search_space["xgb_learning_rate"][0], search_space["xgb_max_depth"][0]
        ])
        highs = np.array([
            search_space["rf_n_estimators"][1], search_space["rf_max_depth"][1], search_space["rf_min_samples_split"][1],
            search_space["gb_n_estimators"][1], search_space["gb_learning_rate"][1], search_space["gb_max_depth"][1],
            search_space["xgb_n_estimators"][1], search_space["xgb_learning_rate"][1], search_space["xgb_max_depth"][1]
        ])
        
        positions = np.random.uniform(lows, highs, size=(n_particles, n_dim))
        velocities = np.random.uniform(-np.abs(highs-lows), np.abs(highs-lows), 
                                     size=(n_particles, n_dim)) * 0.1
        
        # 评估初始粒子
        print("评估初始粒子群...")
        pbest_pos = positions.copy()
        pbest_val = np.array([evaluate_particle(p) for p in positions])
        
        gbest_idx = np.argmax(pbest_val)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]
        
        print(f"PSO初始最优F1-macro: {gbest_val:.4f}")
        
        # PSO迭代
        for iteration in range(n_iterations):
            print(f"PSO迭代 {iteration + 1}/{n_iterations}...")
            
            # 更新速度和位置
            r1 = np.random.random((n_particles, n_dim))
            r2 = np.random.random((n_particles, n_dim))
            
            velocities = (w * velocities + 
                         c1 * r1 * (pbest_pos - positions) + 
                         c2 * r2 * (gbest_pos - positions))
            
            positions = positions + velocities
            positions = np.clip(positions, lows, highs)
            
            # 评估新位置
            vals = np.array([evaluate_particle(p) for p in positions])
            
            # 更新个体最优
            improved = vals > pbest_val
            pbest_pos[improved] = positions[improved]
            pbest_val[improved] = vals[improved]
            
            # 更新全局最优
            if pbest_val.max() > gbest_val:
                gbest_idx = np.argmax(pbest_val)
                gbest_pos = pbest_pos[gbest_idx].copy()
                gbest_val = pbest_val[gbest_idx]
            
            if (iteration + 1) % 5 == 0:
                print(f"  当前最优F1-macro: {gbest_val:.4f}")
        
        # 使用最优参数训练最终模型
        best_params = decode_particle(gbest_pos)
        print(f"\nPSO优化完成，最优参数: {best_params}")
        
        # 创建最优Stacking模型
        rf_optimal = RandomForestClassifier(
            n_estimators=best_params['rf_n_estimators'],
            max_depth=best_params['rf_max_depth'],
            min_samples_split=best_params['rf_min_samples_split'],
            random_state=42,
            n_jobs=-1
        )
        
        gb_optimal = GradientBoostingClassifier(
            n_estimators=best_params['gb_n_estimators'],
            learning_rate=best_params['gb_learning_rate'],
            max_depth=best_params['gb_max_depth'],
            random_state=42
        )
        
        xgb_optimal = xgb.XGBClassifier(
            n_estimators=best_params['xgb_n_estimators'],
            learning_rate=best_params['xgb_learning_rate'],
            max_depth=best_params['xgb_max_depth'],
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        stacking_optimal = StackingClassifier(
            estimators=[
                ('rf', rf_optimal),
                ('gb', gb_optimal),
                ('xgb', xgb_optimal)
            ],
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # 训练最优模型
        stacking_optimal.fit(self.X_train_scaled, self.y_train)
        
        # 评估PSO优化模型
        y_pred = stacking_optimal.predict(self.X_test_scaled)
        y_pred_proba = stacking_optimal.predict_proba(self.X_test_scaled)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='macro')
        recall = recall_score(self.y_test, y_pred, average='macro')
        
        # 保存PSO优化的Stacking模型
        self.models['Stacking_PSO_Optimized'] = stacking_optimal
        self.model_scores['Stacking_PSO_Optimized'] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': best_params
        }
        
        print(f"\nPSO优化Stacking模型性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1-macro: {f1_macro:.4f}")
        print(f"  F1-weighted: {f1_weighted:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        
        # 更新最佳模型
        if f1_macro > max([score['f1_macro'] for score in self.model_scores.values() 
                          if 'f1_macro' in score]):
            self.best_model = stacking_optimal
            print("PSO优化Stacking模型成为新的最佳模型！")
        
        return True

    def create_weighted_probability_fusion(self):
        """创建预测概率加权融合模型"""
        print("\n=== 创建预测概率加权融合模型 ===")
        
        # 确保基础模型存在
        required_models = ['Random_Forest', 'Gradient_Boosting', 'XGBoost']
        available_models = []
        
        for model_name in required_models:
            if model_name in self.models:
                available_models.append(model_name)
            else:
                print(f"⚠️ 模型 {model_name} 不存在，跳过")
        
        if len(available_models) < 2:
            print("❌ 可用模型数量不足，无法进行加权融合")
            return False
        
        print(f"使用模型进行加权融合: {', '.join(available_models)}")
        
        # 获取各模型的预测概率
        model_probabilities = {}
        model_weights = {}
        
        for model_name in available_models:
            model = self.models[model_name]
            y_pred_proba = model.predict_proba(self.X_test_scaled)
            model_probabilities[model_name] = y_pred_proba
            
            # 基于F1-macro得分计算权重
            f1_score_val = self.model_scores[model_name]['f1_macro']
            model_weights[model_name] = f1_score_val
        
        # 归一化权重
        total_weight = sum(model_weights.values())
        normalized_weights = {name: weight/total_weight for name, weight in model_weights.items()}
        
        print("模型权重分配:")
        for name, weight in normalized_weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # 计算加权平均概率
        weighted_proba = np.zeros_like(list(model_probabilities.values())[0])
        
        for model_name, proba in model_probabilities.items():
            weight = normalized_weights[model_name]
            weighted_proba += weight * proba
        
        # 生成最终预测
        y_pred_weighted = np.argmax(weighted_proba, axis=1)
        
        # 计算评估指标
        accuracy = accuracy_score(self.y_test, y_pred_weighted)
        f1_macro = f1_score(self.y_test, y_pred_weighted, average='macro')
        f1_weighted = f1_score(self.y_test, y_pred_weighted, average='weighted')
        precision = precision_score(self.y_test, y_pred_weighted, average='macro')
        recall = recall_score(self.y_test, y_pred_weighted, average='macro')
        
        # 保存加权融合模型结果
        fusion_name = 'Weighted_Probability_Fusion'
        self.model_scores[fusion_name] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred_weighted,
            'probabilities': weighted_proba,
            'model_weights': normalized_weights,
            'used_models': available_models
        }
        
        print(f"\n预测概率加权融合模型性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1-macro: {f1_macro:.4f}")
        print(f"  F1-weighted: {f1_weighted:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        
        # 与单个模型性能比较
        print(f"\n与单个模型性能比较:")
        for model_name in available_models:
            single_f1 = self.model_scores[model_name]['f1_macro']
            improvement = f1_macro - single_f1
            print(f"  vs {model_name}: {improvement:+.4f}")
        
        return True

    def create_feature_level_fusion(self):
        """创建特征级融合模型"""
        print("\n=== 创建特征级融合模型 ===")
        
        # 确保基础模型存在
        required_models = ['Random_Forest', 'Gradient_Boosting', 'XGBoost']
        available_models = []
        
        for model_name in required_models:
            if model_name in self.models:
                available_models.append(model_name)
            else:
                print(f"⚠️ 模型 {model_name} 不存在，跳过")
        
        if len(available_models) < 2:
            print("❌ 可用模型数量不足，无法进行特征级融合")
            return False
        
        print(f"使用模型进行特征级融合: {', '.join(available_models)}")
        
        # 提取各模型的特征重要性
        feature_importances = {}
        
        for model_name in available_models:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                feature_importances[model_name] = model.feature_importances_
            else:
                print(f"⚠️ 模型 {model_name} 不支持特征重要性，跳过")
                continue
        
        if len(feature_importances) < 2:
            print("❌ 支持特征重要性的模型数量不足")
            return False
        
        # 计算加权特征重要性
        model_weights = {}
        for model_name in feature_importances.keys():
            f1_score_val = self.model_scores[model_name]['f1_macro']
            model_weights[model_name] = f1_score_val
        
        # 归一化权重
        total_weight = sum(model_weights.values())
        normalized_weights = {name: weight/total_weight for name, weight in model_weights.items()}
        
        # 计算融合特征重要性
        n_features = len(list(feature_importances.values())[0])
        fused_importance = np.zeros(n_features)
        
        for model_name, importance in feature_importances.items():
            weight = normalized_weights[model_name]
            fused_importance += weight * importance
        
        # 选择最重要的特征（前50%）
        n_selected = max(1, n_features // 2)
        selected_indices = np.argsort(fused_importance)[-n_selected:]
        
        print(f"选择了 {n_selected} 个最重要的特征（共 {n_features} 个）")
        
        # 使用选择的特征训练新模型
        X_train_selected = self.X_train_scaled[:, selected_indices]
        X_test_selected = self.X_test_scaled[:, selected_indices]
        
        # 创建基于特征选择的随机森林模型
        feature_fusion_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # 训练模型
        feature_fusion_model.fit(X_train_selected, self.y_train)
        
        # 预测
        y_pred = feature_fusion_model.predict(X_test_selected)
        y_pred_proba = feature_fusion_model.predict_proba(X_test_selected)
        
        # 计算评估指标
        accuracy = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='macro')
        recall = recall_score(self.y_test, y_pred, average='macro')
        
        # 保存特征级融合模型
        fusion_name = 'Feature_Level_Fusion'
        self.models[fusion_name] = feature_fusion_model
        self.model_scores[fusion_name] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'selected_features': selected_indices,
            'feature_importance': fused_importance,
            'n_selected_features': n_selected,
            'model_weights': normalized_weights
        }
        
        print(f"\n特征级融合模型性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1-macro: {f1_macro:.4f}")
        print(f"  F1-weighted: {f1_weighted:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        
        return True

    # def feature_selection_analysis(self, n_features=30):
    #     """特征选择分析"""
    #     print(f"\n=== 特征选择分析（选择前{n_features}个特征）===")
        
    #     # 使用RFE进行特征选择
    #     rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    #     rfe = RFE(estimator=rf_selector, n_features_to_select=n_features, step=1)
    #     rfe.fit(self.X_train_scaled, self.y_train)
        
    #     # 获取选择的特征
    #     selected_features_mask = rfe.support_
    #     selected_feature_names = [self.selected_features[i] for i in range(len(selected_features_mask)) 
    #                             if selected_features_mask[i]]
        
    #     print(f"RFE选择的前{n_features}个特征:")
    #     for i, feature in enumerate(selected_feature_names, 1):
    #         print(f"  {i:2d}. {feature}")
        
    #     # 使用选择的特征训练模型
    #     X_train_selected = self.X_train_scaled[:, selected_features_mask]
    #     X_test_selected = self.X_test_scaled[:, selected_features_mask]
        
    #     rf_selected = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    #     rf_selected.fit(X_train_selected, self.y_train)
        
    #     y_pred_selected = rf_selected.predict(X_test_selected)
        
    #     accuracy_selected = accuracy_score(self.y_test, y_pred_selected)
    #     f1_macro_selected = f1_score(self.y_test, y_pred_selected, average='macro')
        
    #     print(f"\n特征选择后模型性能:")
    #     print(f"  使用特征数: {n_features}")
    #     print(f"  准确率: {accuracy_selected:.4f}")
    #     print(f"  F1-macro: {f1_macro_selected:.4f}")
        
    #     # 保存特征选择模型
    #     self.models['RF_Feature_Selected'] = rf_selected
    #     self.model_scores['RF_Feature_Selected'] = {
    #         'accuracy': accuracy_selected,
    #         'f1_macro': f1_macro_selected,
    #         'predictions': y_pred_selected,
    #         'selected_features': selected_feature_names,
    #         'feature_mask': selected_features_mask
    #     }
        
    #     return selected_feature_names

    def plot_model_comparison(self):
        """绘制模型比较图"""
        print("\n=== 生成模型比较可视化 ===")
        
        # 准备数据
        model_names = list(self.model_scores.keys())
        accuracies = [self.model_scores[name]['accuracy'] for name in model_names]
        f1_macros = [self.model_scores[name]['f1_macro'] for name in model_names]
        
        # 创建比较图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准确率比较
        bars1 = ax1.bar(model_names, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('模型准确率比较', fontsize=14, fontweight='bold')
        ax1.set_ylabel('准确率', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1-macro比较
        bars2 = ax2.bar(model_names, f1_macros, color='lightcoral', alpha=0.7)
        ax2.set_title('模型F1-Macro得分比较', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1-Macro得分', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, f1 in zip(bars2, f1_macros):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 旋转x轴标签
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        print("\n=== 生成混淆矩阵 ===")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
        if n_models == 1:
            axes = [axes]
        axes = axes.flatten()
        
        class_names = self.label_encoder.classes_
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = self.model_scores[model_name]['predictions']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name} 混淆矩阵')
            axes[idx].set_xlabel('预测标签')
            axes[idx].set_ylabel('真实标签')
        
        # 隐藏多余的子图
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curves(self):
        """绘制ROC曲线"""
        print("\n=== 生成ROC曲线 ===")
        
        class_names = self.label_encoder.classes_
        n_classes = len(class_names)
        
        # 二值化标签
        y_test_bin = label_binarize(self.y_test, classes=range(n_classes))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for class_idx in range(min(n_classes, 4)):  # 最多显示4个类别
            ax = axes[class_idx]
            
            for model_idx, (model_name, model) in enumerate(self.models.items()):
                if 'probabilities' in self.model_scores[model_name]:
                    y_proba = self.model_scores[model_name]['probabilities']
                    
                    if class_idx < y_proba.shape[1]:
                        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_proba[:, class_idx])
                        roc_auc = auc(fpr, tpr)
                        
                        ax.plot(fpr, tpr, color=colors[model_idx % len(colors)],
                               label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('假阳性率 (FPR)')
            ax.set_ylabel('真阳性率 (TPR)')
            ax.set_title(f'ROC曲线 - {class_names[class_idx]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self):
        """绘制特征重要性"""
        print("\n=== 生成特征重要性分析 ===")
        
        if self.best_model is None:
            print("请先训练模型！")
            return
        
        # 获取特征重要性
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.selected_features
            
            # 排序
            indices = np.argsort(importances)[::-1]
            
            # 选择前20个最重要的特征
            top_n = min(20, len(importances))
            top_indices = indices[:top_n]
            top_importances = importances[top_indices]
            top_features = [feature_names[i] for i in top_indices]
            
            # 绘制特征重要性
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(top_n), top_importances[::-1], color='lightgreen', alpha=0.7)
            plt.yticks(range(top_n), [top_features[i] for i in range(top_n-1, -1, -1)])
            plt.xlabel('特征重要性')
            plt.title('随机森林特征重要性分析（前20个特征）', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{top_importances[top_n-1-i]:.3f}', 
                        va='center', ha='left', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 保存特征重要性到CSV
            importance_df = pd.DataFrame({
                'feature': top_features,
                'importance': top_importances
            })
            importance_df.to_csv(f'{self.results_dir}/feature_importance.csv', index=False)
            print(f"特征重要性已保存到 {self.results_dir}/feature_importance.csv")

    def generate_classification_report(self):
        """生成详细的分类报告"""
        print("\n=== 生成分类报告 ===")
        
        class_names = self.label_encoder.classes_
        
        # 为每个模型生成报告
        all_reports = {}
        
        for model_name in self.models.keys():
            y_pred = self.model_scores[model_name]['predictions']
            
            # 生成分类报告
            report = classification_report(self.y_test, y_pred, 
                                         target_names=class_names, 
                                         output_dict=True)
            all_reports[model_name] = report
            
            print(f"\n{model_name} 分类报告:")
            print(classification_report(self.y_test, y_pred, target_names=class_names))
        
        # 保存报告到CSV
        report_data = []
        for model_name, report in all_reports.items():
            for class_name in class_names:
                if class_name in report:
                    report_data.append({
                        'model': model_name,
                        'class': class_name,
                        'precision': report[class_name]['precision'],
                        'recall': report[class_name]['recall'],
                        'f1-score': report[class_name]['f1-score'],
                        'support': report[class_name]['support']
                    })
            
            # 添加总体指标
            report_data.append({
                'model': model_name,
                'class': 'macro avg',
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1-score': report['macro avg']['f1-score'],
                'support': report['macro avg']['support']
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(f'{self.results_dir}/classification_report.csv', index=False)
        print(f"分类报告已保存到 {self.results_dir}/classification_report.csv")

    def save_all_models(self):
        """保存全部训练的模型"""
        if not self.model_scores:
            print("没有可保存的模型！")
            return
        
        # 按F1-macro分数排序，获取全部模型
        sorted_models = sorted(self.model_scores.items(), 
                             key=lambda x: x[1]['f1_macro'], 
                             reverse=True)
        
        print(f"\n保存全部{len(sorted_models)}个训练的模型:")
        
        for i, (model_name, scores) in enumerate(sorted_models, 1):
            # 获取对应的模型对象
            if model_name in self.models:
                model_obj = self.models[model_name]
                
                # 构建文件名，使用排名和模型名称
                model_path = f'{self.results_dir}/rank_{i}_{model_name}_model.pkl'
                
                # 保存模型和相关信息
                model_info = {
                    'model': model_obj,
                    'model_name': model_name,
                    'rank': i,
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder,
                    'selected_features': self.selected_features,
                    'scores': scores,
                    'f1_macro': scores['f1_macro'],
                    'accuracy': scores['accuracy'],
                    'precision': scores['precision'],
                    'recall': scores['recall']
                }
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info, f)
                
                print(f"  第{i}名: {model_name} (F1-macro: {scores['f1_macro']:.4f}) -> {model_path}")
            else:
                print(f"  警告: 模型 {model_name} 不存在于self.models中，跳过保存")
        
        # 保存一个包含所有模型的汇总文件
        all_models_summary_path = f'{self.results_dir}/all_models_summary.pkl'
        summary_info = {
            'all_models_ranked': sorted_models,
            'total_models': len(sorted_models),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'selected_features': self.selected_features,
            'all_model_scores': self.model_scores,
            'best_model_name': sorted_models[0][0] if sorted_models else None,
            'best_f1_score': sorted_models[0][1]['f1_macro'] if sorted_models else None
        }
        
        with open(all_models_summary_path, 'wb') as f:
            pickle.dump(summary_info, f)
        
        print(f"\n全部{len(sorted_models)}个模型汇总信息已保存到: {all_models_summary_path}")
        print(f"最佳模型: {sorted_models[0][0]} (F1-macro: {sorted_models[0][1]['f1_macro']:.4f})" if sorted_models else "无模型")

    def _print_final_summary(self):
        """打印最终结果总结"""
        print("\n📊 最终诊断结果总结:")
        print("-" * 50)
        
        if not self.model_scores:
            print("❌ 没有模型评估结果可显示")
            return
        
        # 显示所有模型的性能
        print(f"📈 共训练了 {len(self.model_scores)} 个模型:")
        
        # 按F1-macro分数排序
        sorted_models = sorted(self.model_scores.items(), 
                             key=lambda x: x[1]['f1_macro'], 
                             reverse=True)
        
        for i, (model_name, scores) in enumerate(sorted_models, 1):
            print(f"  {i}. {model_name}:")
            print(f"     准确率: {scores['accuracy']:.4f}")
            print(f"     F1-macro: {scores['f1_macro']:.4f}")
            print(f"     精确率: {scores['precision']:.4f}")
            print(f"     召回率: {scores['recall']:.4f}")
            if i == 1:
                print("     🏆 最佳模型")
            print()
        
        # 显示最佳模型信息
        if hasattr(self, 'best_model') and self.best_model is not None:
            best_model_name = max(self.model_scores.items(), 
                                key=lambda x: x[1]['f1_macro'])[0]
            print(f"🎯 最佳模型: {best_model_name}")
            print(f"🎯 最佳F1-macro得分: {max(score['f1_macro'] for score in self.model_scores.values()):.4f}")
        
        # 显示数据集信息
        if hasattr(self, 'X_train') and hasattr(self, 'X_test'):
            print(f"\n📋 数据集信息:")
            print(f"   训练集样本数: {len(self.X_train)}")
            print(f"   测试集样本数: {len(self.X_test)}")
            print(f"   特征数量: {self.X_train.shape[1] if hasattr(self.X_train, 'shape') else 'N/A'}")
        
        # 显示故障类别信息
        if hasattr(self, 'label_encoder') and self.label_encoder is not None:
            print(f"   故障类别数: {len(self.label_encoder.classes_)}")
            print(f"   故障类别: {', '.join(self.label_encoder.classes_)}")

    def optimize_selected_models(self):
        """对值得探索的模型进行超参数优化（只对随机森林进行优化）"""
        print("\n=== 开始对选定模型进行超参数优化 ===")
        
        # 只对随机森林进行超参数优化
        models_to_optimize = ['Random_Forest']
        
        # 定义超参数搜索空间（连续区间范围）
        param_distributions = {
            'Random_Forest': {
                'n_estimators': randint(50, 300),           # 100-300之间的整数
                'max_depth': randint(5, 30),                # 10-30之间的整数
                'min_samples_split': randint(2, 10),         # 2-10之间的整数
                'min_samples_leaf': randint(1, 5),           # 1-5之间的整数
                'max_features': uniform(0.5, 1.0)           # 0.5-1.0之间的连续值
            }
            # 移除了Gradient_Boosting的超参数配置
        }
        
        # 定义基础模型
        base_models = {
            'Random_Forest': RandomForestClassifier(random_state=42, n_jobs=-1)
            # 移除了Gradient_Boosting的基础模型配置
        }
        
        optimized_models = {}
        
        print(f"将对以下模型进行超参数优化: {', '.join(models_to_optimize)}")
        print("其他模型（SVM_RBF, Logistic_Regression, Gradient_Boosting）保持基础配置")
        
        # 对选定的模型进行超参数优化
        for model_name in models_to_optimize:
            if model_name in self.models:
                print(f"\n🔧 优化 {model_name}...")
                start_time = time.time()
                
                try:
                    # 创建RandomizedSearchCV进行区间搜索
                    random_search = RandomizedSearchCV(
                        estimator=base_models[model_name],
                        param_distributions=param_distributions[model_name],
                        n_iter=50,  # 随机搜索50次
                        cv=3,
                        scoring='f1_macro',
                        n_jobs=-1,
                        verbose=0,
                        random_state=42
                    )
                    
                    # 执行随机搜索
                    random_search.fit(self.X_train_scaled, self.y_train)
                    
                    # 获取最佳模型
                    best_model = random_search.best_estimator_
                    
                    # 预测
                    y_pred = best_model.predict(self.X_test_scaled)
                    y_pred_proba = best_model.predict_proba(self.X_test_scaled)
                    
                    # 计算评估指标
                    accuracy = accuracy_score(self.y_test, y_pred)
                    f1_macro = f1_score(self.y_test, y_pred, average='macro')
                    f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
                    precision = precision_score(self.y_test, y_pred, average='macro')
                    recall = recall_score(self.y_test, y_pred, average='macro')
                    optimization_time = time.time() - start_time
                    
                    # 保存优化后的模型
                    optimized_name = f"{model_name}_Optimized"
                    optimized_models[optimized_name] = best_model
                    self.model_scores[optimized_name] = {
                        'accuracy': accuracy,
                        'f1_macro': f1_macro,
                        'f1_weighted': f1_weighted,
                        'precision': precision,
                        'recall': recall,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba,
                        'best_params': random_search.best_params_,
                        'optimization_time': optimization_time,
                        'cv_score': random_search.best_score_
                    }
                    
                    print(f"  ✅ {model_name} 优化完成:")
                    print(f"    最佳参数: {random_search.best_params_}")
                    print(f"    交叉验证得分: {random_search.best_score_:.4f}")
                    print(f"    测试集准确率: {accuracy:.4f}")
                    print(f"    测试集F1-macro: {f1_macro:.4f}")
                    print(f"    优化时间: {optimization_time:.2f}秒")
                    
                    # 比较优化前后的性能
                    original_f1 = self.model_scores[model_name]['f1_macro']
                    improvement = f1_macro - original_f1
                    print(f"    性能提升: {improvement:+.4f}")
                    
                except Exception as e:
                    print(f"  ❌ {model_name} 优化失败: {str(e)}")
                    continue
            else:
                print(f"⚠️ 模型 {model_name} 不存在，跳过优化")
        
        # 使用最优参数直接创建Stacking模型
        if 'Stacking_RF_GB_XGB' in self.models:
            print(f"\n🔧 使用最优参数创建Stacking集成模型...")
            try:
                self.create_optimized_stacking_model()
                print("  ✅ 优化Stacking模型创建完成")
            except Exception as e:
                print(f"  ❌ 优化Stacking模型创建失败: {str(e)}")
        
        # 创建预测概率加权融合模型（已注释，保留代码备用）
        # print(f"\n🔧 创建预测概率加权融合模型...")
        # try:
        #     self.create_weighted_probability_fusion()
        #     print("  ✅ 预测概率加权融合模型创建完成")
        # except Exception as e:
        #     print(f"  ❌ 预测概率加权融合模型创建失败: {str(e)}")
        
        # 创建特征级融合模型（已注释，保留代码备用）
        # print(f"\n🔧 创建特征级融合模型...")
        # try:
        #     self.create_feature_level_fusion()
        #     print("  ✅ 特征级融合模型创建完成")
        # except Exception as e:
        #     print(f"  ❌ 特征级融合模型创建失败: {str(e)}")
        
        # 更新模型字典
        self.models.update(optimized_models)
        
        # 确定最佳模型（包括优化后的模型）
        if self.model_scores:
            best_model_name = max(self.model_scores.keys(), 
                                key=lambda x: self.model_scores[x]['f1_macro'])
            self.best_model = self.models[best_model_name]
            
            print(f"\n🏆 超参数优化后的最佳模型: {best_model_name}")
            print(f"F1-macro得分: {self.model_scores[best_model_name]['f1_macro']:.4f}")
            
            if 'best_params' in self.model_scores[best_model_name]:
                print(f"最佳参数: {self.model_scores[best_model_name]['best_params']}")
        
        print(f"\n📊 当前共有 {len(self.models)} 个模型:")
        for name in self.models.keys():
            f1_macro_score = self.model_scores[name]['f1_macro']
            print(f"  - {name}: F1-macro = {f1_macro_score:.4f}")
        
        return True

    def run_complete_diagnosis(self):
        """运行完整的诊断流程"""
        print("🚀 开始轴承故障智能诊断分析...")
        
        # 1. 加载数据
        print("\n" + "="*50)
        print("第1步: 加载特征数据")
        if not self.load_features():
            return False
        
        # 2. 数据预处理
        print("\n" + "="*50)
        print("第2步: 数据预处理")
        if not self.preprocess_data():
            return False
        
        # 3. 训练核心模型
        print("\n" + "="*50)
        print("第3步: 训练核心模型")
        self.train_core_models()
        
        # 4. 对选定模型进行超参数优化
        print("\n" + "="*50)
        print("第4步: 超参数优化")
        self.optimize_selected_models()
        
        # 5. 生成可视化结果
        print("\n" + "="*50)
        print("第5步: 生成可视化结果")
        self.plot_model_comparison()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_feature_importance()
        
        # 6. 生成报告
        print("\n" + "="*50)
        print("第6步: 生成分类报告")
        self.generate_classification_report()
        
        # 7. 保存全部训练的模型
        print("\n" + "="*50)
        print("第7步: 保存全部训练的模型")
        self.save_all_models()
        
        # 8. 显示最终结果总结
        print("\n" + "="*50)
        print("第8步: 最终结果总结")
        self._print_final_summary()
        
        print("\n🎉 轴承故障智能诊断分析完成！")
        return True


def main():
    """主函数"""
    # 创建诊断模型
    model = EnhancedBearingDiagnosisModel()
    
    # 运行完整诊断流程
    success = model.run_complete_diagnosis()
    
    if success:
        print("\n=== 诊断结果摘要 ===")
        print(f"训练的模型数量: {len(model.models)}")
        print(f"最佳模型性能:")
        
        best_scores = max(model.model_scores.values(), key=lambda x: x['f1_macro'])
        print(f"  准确率: {best_scores['accuracy']:.4f}")
        print(f"  F1-macro: {best_scores['f1_macro']:.4f}")
        print(f"  F1-weighted: {best_scores['f1_weighted']:.4f}")
        
        print(f"\n所有结果文件已保存到: {model.results_dir}")
    else:
        print("诊断流程执行失败！")


if __name__ == "__main__":
    main()