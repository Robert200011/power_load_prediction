# 电力负荷预测系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.ai/)

基于机器学习的电力负荷预测系统，采用模块化设计，支持一键运行完整的数据挖掘流程。

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [详细说明](#详细说明)
- [实验结果](#实验结果)
- [未来改进](#未来改进)
- [参考资料](#参考资料)

## 项目简介

电力负荷预测是电网调度和能源管理的核心任务。准确的负荷预测可以优化发电计划、降低运营成本、提高电网稳定性。本项目构建了一个完整的机器学习预测系统，通过比较多种算法，实现高精度的电力负荷预测。

### 主要目标

- 构建完整的数据挖掘流程（预处理→特征工程→建模→评估）
- 比较不同复杂度的机器学习算法性能
- 实现模块化、可扩展的代码架构
- 自动生成可视化结果和评估报告

## 功能特性

✅ **模块化设计** - 各个步骤独立封装，易于维护和扩展  
✅ **一键运行** - 支持完整流程自动执行，支持断点续跑  
✅ **多模型对比** - 线性回归、随机森林、XGBoost三种算法  
✅ **自动化评估** - 自动生成性能对比图、预测可视化、特征重要性分析  
✅ **高预测精度** - XGBoost模型R²达到97.91%  
✅ **完整文档** - 包含Jupyter Notebook交互式分析

## 项目结构

```
power_load_prediction/
├── main.py                          # 主程序入口，一键运行完整流程
├── requirements.txt                 # 项目依赖
├── README.md                        # 项目说明文档
│
├── src/                             # 源代码模块
│   ├── data_preprocessing.py       # 数据预处理模块
│   ├── feature_engineering.py      # 特征工程模块
│   ├── model_training.py           # 模型训练模块
│   └── evaluation.py               # 模型评估模块
│
├── data/                            # 数据目录
│   ├── synthetic_load.csv          # 原始数据集
│   ├── processed_data.csv          # 预处理后的数据
│   └── featured_data.csv           # 特征工程后的数据
│
├── models/                          # 训练好的模型
│   ├── linear_regression.pkl       # 线性回归模型
│   ├── random_forest.pkl           # 随机森林模型
│   ├── xgboost.pkl                 # XGBoost模型
│   └── scaler.pkl                  # 特征标准化器
│
├── results/                         # 评估结果
│   ├── metrics_comparison.png      # 模型性能对比图
│   ├── predictions.png             # 预测结果可视化
│   ├── feature_importance.png      # 特征重要性分析
│   └── evaluation_report.txt       # 详细评估报告
│
└── notebooks/                       # Jupyter Notebooks
    └── complete_analysis.ipynb     # 完整的交互式分析
```

## 环境配置

### 系统要求

- Python 3.8+
- pip 或 conda 包管理器

### 安装依赖

```bash
# 使用pip安装
pip install -r requirements.txt

# 或者单独安装
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter joblib
```

### 依赖包说明

| 包名 | 版本 | 用途 |
|------|------|------|
| pandas | ≥1.3.0 | 数据处理和分析 |
| numpy | ≥1.21.0 | 数值计算 |
| matplotlib | ≥3.4.0 | 数据可视化 |
| seaborn | ≥0.11.0 | 统计图表 |
| scikit-learn | ≥0.24.0 | 机器学习算法 |
| xgboost | ≥1.4.0 | 梯度提升算法 |
| joblib | ≥1.0.0 | 模型持久化 |
| jupyter | ≥1.0.0 | 交互式分析 |

## 快速开始

### 方式一：一键运行（推荐）

```bash
python main.py
```

程序会自动执行以下步骤：
1. 数据预处理
2. 特征工程
3. 模型训练
4. 模型评估和结果生成

> 💡 **提示**：程序支持断点续跑。如果检测到已完成的步骤，会自动跳过。

### 方式二：分步执行

```bash
# 步骤1：数据预处理
python src/data_preprocessing.py

# 步骤2：特征工程
python src/feature_engineering.py

# 步骤3：模型训练
python src/model_training.py

# 步骤4：模型评估
python src/evaluation.py
```

### 方式三：Jupyter Notebook

```bash
jupyter notebook notebooks/complete_analysis.ipynb
```

交互式地探索数据和模型结果。

## 详细说明

### 1. 数据预处理

**输入**: `data/synthetic_load.csv`  
**输出**: `data/processed_data.csv`

主要处理步骤：

- **时间特征提取**
  - 年、月、日、小时、星期、季度
  - 是否周末（二值特征）

- **缺失值处理**
  - 检测缺失值数量
  - 使用前向填充法（forward fill）保持数据连续性

- **异常值检测**
  - Z-score方法（阈值：3倍标准差）
  - 用中位数替换异常值

```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor('data/synthetic_load.csv')
preprocessor.load_data()
preprocessor.handle_datetime()
preprocessor.handle_missing_values()
preprocessor.handle_outliers()
preprocessor.save_processed_data('data/processed_data.csv')
```

### 2. 特征工程

**输入**: `data/processed_data.csv`  
**输出**: `data/featured_data.csv`

创建的特征类型：

- **滞后特征** (Lag Features)
  - 1小时、2小时、3小时前的负荷
  - 24小时前的负荷（日周期性）
  - 168小时前的负荷（周周期性）

- **滚动窗口特征** (Rolling Features)
  - 24小时滚动均值和标准差
  - 168小时滚动均值和标准差

- **交互特征** (Interaction Features)
  - 温度 × 假期
  - 温度 × 小时
  - 周末 × 小时

```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer('data/processed_data.csv')
engineer.load_data()
feature_cols, target = engineer.prepare_data_for_modeling()
engineer.save_featured_data('data/featured_data.csv')
```

### 3. 模型训练

**输入**: `data/featured_data.csv`  
**输出**: `models/*.pkl`

训练三种模型：

| 模型 | 参数配置 | 特点 |
|------|----------|------|
| **线性回归** | StandardScaler标准化 | 基准模型，可解释性强 |
| **随机森林** | n_estimators=50, max_depth=20 | 集成学习，抗过拟合 |
| **XGBoost** | n_estimators=50, max_depth=10, lr=0.1 | 高性能，处理非线性 |

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer('data/featured_data.csv')
trainer.load_data()
trainer.prepare_train_test_split(test_size=0.2)
trainer.train_all_models()
trainer.save_models('models/')
```

### 4. 模型评估

**输入**: `models/*.pkl`, `data/featured_data.csv`  
**输出**: `results/`目录下的图表和报告

评估指标：
- **MAE** (Mean Absolute Error) - 平均绝对误差
- **RMSE** (Root Mean Squared Error) - 均方根误差
- **R²** (Coefficient of Determination) - 决定系数

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator('models/', 'data/featured_data.csv')
evaluator.load_models()
evaluator.load_data()
evaluator.evaluate_all_models()
evaluator.plot_metrics_comparison('results/metrics_comparison.png')
evaluator.plot_predictions('results/predictions.png')
evaluator.plot_feature_importance('results/feature_importance.png')
evaluator.generate_report('results/evaluation_report.txt')
```

## 实验结果

### 模型性能对比

| 模型 | MAE | RMSE | R² Score |
|------|-----|------|----------|
| Linear Regression | 30.52 | 38.68 | 0.9567 |
| Random Forest | 19.17 | 27.25 | 0.9785 |
| **XGBoost** | **19.10** | **26.87** | **0.9791** |

### 最佳模型

**XGBoost** 在所有指标上表现最优：
- R² = **0.9791**（解释了97.91%的数据变异）
- RMSE = **26.87**（预测误差最小）
- MAE = **19.10**（平均绝对误差最低）

### 关键发现

1. **滞后特征最重要** - 历史负荷是最强的预测因子
2. **时序相关性显著** - 1小时和24小时的滞后特征贡献最大
3. **集成方法优于线性模型** - 随机森林和XGBoost显著优于线性回归

### Top 5 重要特征

1. `nat_demand_lag_1` - 1小时前的负荷
2. `nat_demand_lag_24` - 24小时前的负荷
3. `nat_demand_rolling_mean_24` - 24小时滚动均值
4. `nat_demand_rolling_mean_168` - 一周滚动均值
5. `nat_demand_lag_2` - 2小时前的负荷

## 未来改进

### 短期优化

- [ ] 超参数自动调优（网格搜索、随机搜索、贝叶斯优化）
- [ ] 交叉验证评估模型稳定性
- [ ] 添加更多特征（傅里叶变换、小波变换）

### 中期扩展

- [ ] 引入深度学习模型（LSTM、GRU、Transformer）
- [ ] 集成更多外部数据（天气预报、节假日、经济指标）
- [ ] 实现多步预测（预测未来多个时间点）

### 长期规划

- [ ] 构建实时预测系统
- [ ] 自动模型更新和重训练机制
- [ ] 部署为Web服务或API
- [ ] 可视化仪表板（Dashboard）

## 参考资料

### 相关论文

- [Short-Term Load Forecasting: A Deep Learning Approach](https://arxiv.org/abs/1906.08863)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

### 技术文档

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)

### 数据集说明

本项目使用合成的电力负荷数据集，包含以下字段：
- `datetime`: 时间戳（小时级）
- `nat_demand`: 电力需求（目标变量）
- `T2M_toc`: 温度数据
- `holiday`: 是否假期
- 其他气象和日历特征

