import pandas as pd

def load_dataset(path="data/synthetic_load.csv"):
    """加载电力负荷数据"""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"✅ 数据加载成功，样本数: {len(df)}, 字段: {list(df.columns)}")
    return df

if __name__ == "__main__":
    df = load_dataset()
    print(df.head())

