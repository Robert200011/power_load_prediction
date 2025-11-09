import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data.preprocess.feature_engineering import prepare_data_for_lstm
from data.preprocess.clean_data import clean_data
from data.preprocess.load_data import load_dataset
import math

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    df = load_dataset()
    df = clean_data(df)
    X, y, scaler_X, scaler_y = prepare_data_for_lstm(df)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model((X.shape[1], X.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=[es])

    preds = model.predict(X_test)
    preds_inv = scaler_y.inverse_transform(preds)
    y_test_inv = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_inv, preds_inv)
    rmse = math.sqrt(mean_squared_error(y_test_inv, preds_inv))

    print(f"✅ 模型评估结果: MAE={mae:.2f}, RMSE={rmse:.2f}")

    model.save("models/lstm_load_forecast.h5")
    print("💾 模型已保存至 models/lstm_load_forecast.h5")
