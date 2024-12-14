import data

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def create_sequences(data, time_step=30):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i : (i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


def evaluate(model, scaler, X_test, y_test, title):
    predictions = model.predict(X_test)

    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title(title + " prediction")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def main():
    topic = ""
    if len(sys.argv) > 1:
        topic = sys.argv[1]
    if topic == "" or topic is None:
        topic = "humid_1"

    df = data.get_data(topic)

    if topic == "humid_1":
        df = df[(df["Value"] >= 30) & (df["Value"] <= 60)]
    if topic == "temp_1":
        df = df[(df["Value"] >= 15) & (df["Value"] <= 35)]
    if topic == "light_intensity":
        df = df[(df["Value"] >= 100) & (df["Value"] <= 500)]
    df["Value"] = df["Value"].fillna(method="ffill")
    df["smooth"] = df["Value"].rolling(window=5).mean()
    df = df.dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[["smooth"]])

    X, y = create_sequences(scaled_data, 100)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    evaluate(model, scaler, X_test, y_test, topic)
    model.save(topic + ".keras")


if __name__ == "__main__":
    main()
