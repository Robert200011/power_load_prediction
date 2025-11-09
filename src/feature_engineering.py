"""
特征工程模块
负责特征选择、特征变换、特征创建等
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib


class FeatureEngineer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()

    def load_data(self):
        """加载预处理后的数据"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.data_path)
        print(f"数据加载完成！共 {len(self.df)} 条记录")
        return self.df

    def create_lag_features(self, target_col='nat_demand', lags=[1, 2, 3, 24, 168]):
        """创建滞后特征"""
        print(f"\n正在创建滞后特征...")

        for lag in lags:
            self.df[f'{target_col}_lag_{lag}'] = self.df[target_col].shift(lag)

        print(f"创建了 {len(lags)} 个滞后特征: {lags}")

    def create_rolling_features(self, target_col='nat_demand', windows=[24, 168]):
        """创建滚动窗口特征"""
        print(f"\n正在创建滚动窗口特征...")

        for window in windows:
            # 滚动平均
            self.df[f'{target_col}_rolling_mean_{window}'] = \
                self.df[target_col].rolling(window=window).mean()

            # 滚动标准差
            self.df[f'{target_col}_rolling_std_{window}'] = \
                self.df[target_col].rolling(window=window).std()

        print(f"创建了 {len(windows) * 2} 个滚动特征（窗口: {windows}）")

    def create_interaction_features(self):
        """创建交互特征"""
        print(f"\n正在创建交互特征...")

        # 温度与假期的交互
        self.df['temp_holiday'] = self.df['T2M_toc'] * self.df['holiday']

        # 温度与小时的交互
        self.df['temp_hour'] = self.df['T2M_toc'] * self.df['hour']

        # 周末与小时的交互
        self.df['weekend_hour'] = self.df['is_weekend'] * self.df['hour']

        print("创建了3个交互特征")

    def select_features(self):
        """选择用于建模的特征"""
        print(f"\n正在选择特征...")

        # 目标变量
        target = 'nat_demand'

        # 排除的列
        exclude_cols = ['datetime', 'nat_demand']

        # 选择所有数值型特征
        feature_cols = [col for col in self.df.columns
                        if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']]

        print(f"选择了 {len(feature_cols)} 个特征")

        return feature_cols, target

    def handle_infinite_values(self):
        """处理无穷值"""
        print("\n检查并处理无穷值...")

        # 检查无穷值
        inf_count = np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum()

        if inf_count > 0:
            print(f"发现 {inf_count} 个无穷值")
            # 将无穷值替换为NaN，然后填充
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.df.fillna(method='ffill', inplace=True)
            self.df.fillna(method='bfill', inplace=True)
            print("无穷值已处理")
        else:
            print("未发现无穷值")

    def prepare_data_for_modeling(self):
        """准备用于建模的数据"""
        print("\n=== 准备建模数据 ===")

        # 创建特征
        self.create_lag_features()
        self.create_rolling_features()
        self.create_interaction_features()

        # 处理无穷值
        self.handle_infinite_values()

        # 删除因滞后特征产生的NaN
        print(f"\n删除前数据量: {len(self.df)}")
        self.df.dropna(inplace=True)
        print(f"删除后数据量: {len(self.df)}")

        # 选择特征
        feature_cols, target = self.select_features()

        return feature_cols, target

    def scale_features(self, X_train, X_test):
        """标准化特征"""
        print("\n正在标准化特征...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("特征标准化完成")

        return X_train_scaled, X_test_scaled

    def save_scaler(self, output_path):
        """保存标准化器"""
        joblib.dump(self.scaler, output_path)
        print(f"标准化器已保存到: {output_path}")

    def save_featured_data(self, output_path):
        """保存特征工程后的数据"""
        self.df.to_csv(output_path, index=False)
        print(f"\n特征工程后的数据已保存到: {output_path}")


def main():
    """主函数"""
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed_data.csv')
    output_path = os.path.join(base_dir, 'data', 'featured_data.csv')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

    # 创建特征工程器
    engineer = FeatureEngineer(data_path)
    engineer.load_data()

    # 执行特征工程
    feature_cols, target = engineer.prepare_data_for_modeling()

    print(f"\n特征列表:")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i}. {col}")

    # 保存数据
    engineer.save_featured_data(output_path)

    print("\n特征工程完成！")


if __name__ == "__main__":
    main()