"""
数据预处理模块
负责加载数据、处理缺失值、异常值等
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.data_path)
        print(f"数据加载完成！共 {len(self.df)} 条记录")
        print(f"特征数量: {self.df.shape[1]}")
        return self.df

    def basic_info(self):
        """显示基本信息"""
        print("\n=== 数据基本信息 ===")
        print(f"数据形状: {self.df.shape}")
        print(f"\n前5行数据:")
        print(self.df.head())
        print(f"\n数据类型:")
        print(self.df.dtypes)
        print(f"\n缺失值统计:")
        print(self.df.isnull().sum())
        print(f"\n描述性统计:")
        print(self.df.describe())

    def handle_datetime(self):
        """处理日期时间特征"""
        print("\n正在处理日期时间特征...")
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])

        # 提取时间特征
        self.df['year'] = self.df['datetime'].dt.year
        self.df['month'] = self.df['datetime'].dt.month
        self.df['day'] = self.df['datetime'].dt.day
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['dayofweek'] = self.df['datetime'].dt.dayofweek
        self.df['quarter'] = self.df['datetime'].dt.quarter

        # 是否周末
        self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)

        print("时间特征提取完成！")

    def handle_missing_values(self):
        """处理缺失值"""
        print("\n正在处理缺失值...")
        missing_count = self.df.isnull().sum().sum()

        if missing_count > 0:
            print(f"发现 {missing_count} 个缺失值")
            # 对数值列使用前向填充
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(method='ffill')
            print("缺失值处理完成！")
        else:
            print("没有发现缺失值")

    def handle_outliers(self, column='nat_demand', threshold=3):
        """处理异常值（使用Z-score方法）"""
        print(f"\n正在检测 {column} 的异常值...")

        mean = self.df[column].mean()
        std = self.df[column].std()
        z_scores = np.abs((self.df[column] - mean) / std)

        outliers = z_scores > threshold
        outlier_count = outliers.sum()

        print(f"发现 {outlier_count} 个异常值 (阈值: {threshold} 倍标准差)")

        if outlier_count > 0:
            # 用中位数替换异常值
            median = self.df[column].median()
            self.df.loc[outliers, column] = median
            print(f"异常值已用中位数 ({median:.2f}) 替换")

        return outlier_count

    def get_processed_data(self):
        """返回处理后的数据"""
        return self.df

    def save_processed_data(self, output_path):
        """保存处理后的数据"""
        self.df.to_csv(output_path, index=False)
        print(f"\n处理后的数据已保存到: {output_path}")


def main():
    """主函数"""
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'synthetic_load.csv')
    output_path = os.path.join(base_dir, 'data', 'processed_data.csv')

    # 创建预处理器
    preprocessor = DataPreprocessor(data_path)

    # 执行预处理流程
    preprocessor.load_data()
    preprocessor.basic_info()
    preprocessor.handle_datetime()
    preprocessor.handle_missing_values()
    preprocessor.handle_outliers()

    # 保存处理后的数据
    preprocessor.save_processed_data(output_path)

    print("\n数据预处理完成！")


if __name__ == "__main__":
    main()