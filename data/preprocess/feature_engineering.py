import pandas as pd
import numpy as np
from clean_data import clean_data
from load_data import load_dataset
from sklearn.preprocessing import MinMaxScaler

def create_lag_features(df, lag=24):
    """生成前 lag 小时的历史特征"""
    for i in range(1, lag + 1):
        df[f"lag_{i}"] = df["load"].shift(i)
    return df

def prepare_data_for_lstm(df, lookback=24):
    """构造LSTM输入序列"""
    df = create_lag_features(df, lag=lookback)
    df = df.dropna().reset_index(drop=True)

    features = [col for col in df.columns if col not in ["timestamp", "load"]]
    X = df[features].values
    y = df["load"].values

    # 缩放
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # reshape for LSTM: [samples, time_steps, features]
    X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    return X_lstm, y_scaled, scaler_X, scaler_y

if __name__ == "__main__":
    df = load_dataset()
    df = clean_data(df)
    X, y, _, _ = prepare_data_for_lstm(df)
    print(f"✅ LSTM输入维度: {X.shape}, 标签维度: {y.shape}")
