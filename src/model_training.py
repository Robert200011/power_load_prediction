"""
模型训练模块
负责训练多个机器学习模型
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import time


class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self):
        """加载特征工程后的数据"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.data_path)
        print(f"数据加载完成！共 {len(self.df)} 条记录")
        return self.df

    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        """准备训练集和测试集"""
        print("\n=== 准备训练集和测试集 ===")

        # 选择特征和目标
        target = 'nat_demand'
        exclude_cols = ['datetime', 'nat_demand']

        feature_cols = [col for col in self.df.columns
                        if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']]

        X = self.df[feature_cols]
        y = self.df[target]

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )

        print(f"训练集大小: {len(self.X_train)}")
        print(f"测试集大小: {len(self.X_test)}")
        print(f"特征数量: {self.X_train.shape[1]}")

        # 标准化特征
        print("\n正在标准化特征...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("特征标准化完成")

        return feature_cols

    def train_linear_regression(self):
        """训练线性回归模型"""
        print("\n=== 训练线性回归模型 ===")
        start_time = time.time()

        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_train)

        train_time = time.time() - start_time
        print(f"训练完成！耗时: {train_time:.2f} 秒")

        self.models['Linear Regression'] = model
        return model

    def train_random_forest(self, n_estimators=100, max_depth=20, random_state=42):
        """训练随机森林模型"""
        print("\n=== 训练随机森林模型 ===")
        start_time = time.time()

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        model.fit(self.X_train, self.y_train)

        train_time = time.time() - start_time
        print(f"训练完成！耗时: {train_time:.2f} 秒")

        self.models['Random Forest'] = model
        return model

    def train_xgboost(self, n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42):
        """训练XGBoost模型"""
        print("\n=== 训练XGBoost模型 ===")
        start_time = time.time()

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)

        train_time = time.time() - start_time
        print(f"训练完成！耗时: {train_time:.2f} 秒")

        self.models['XGBoost'] = model
        return model

    def train_all_models(self):
        """训练所有模型"""
        print("\n" + "=" * 50)
        print("开始训练所有模型")
        print("=" * 50)

        self.train_linear_regression()
        self.train_random_forest(n_estimators=50)  # 减少树的数量加快训练
        self.train_xgboost(n_estimators=50)

        print("\n所有模型训练完成！")

    def save_models(self, output_dir):
        """保存训练好的模型"""
        print("\n正在保存模型...")

        os.makedirs(output_dir, exist_ok=True)

        for name, model in self.models.items():
            filename = name.replace(' ', '_').lower() + '.pkl'
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"已保存: {filename}")

        # 保存标准化器
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"已保存: scaler.pkl")

        print(f"\n所有模型已保存到: {output_dir}")

    def get_models(self):
        """返回训练好的模型"""
        return self.models

    def get_data_splits(self):
        """返回数据划分"""
        return self.X_train, self.X_test, self.y_train, self.y_test


def main():
    """主函数"""
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'featured_data.csv')
    models_dir = os.path.join(base_dir, 'models')

    # 创建训练器
    trainer = ModelTrainer(data_path)
    trainer.load_data()

    # 准备数据
    feature_cols = trainer.prepare_train_test_split()

    # 训练所有模型
    trainer.train_all_models()

    # 保存模型
    trainer.save_models(models_dir)

    print("\n模型训练完成！")


if __name__ == "__main__":
    main()