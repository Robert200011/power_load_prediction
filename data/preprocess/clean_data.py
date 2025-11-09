import pandas as pd
from load_data import load_dataset

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """基础清洗：排序、去重、填充缺失值"""
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    # 填充缺失值
    df["temperature"] = df["temperature"].interpolate(method="linear")
    df["load"] = df["load"].interpolate(method="linear")

    print(f"✅ 数据清洗完成: {len(df)} 条记录")
    return df

if __name__ == "__main__":
    df = load_dataset()
    df = clean_data(df)
    print(df.head())
