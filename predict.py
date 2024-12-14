import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt


def main():
    model = load_model("humid_1.keras")  # Change the name to your saved model

    last_100_days = scaled_data[-100:].reshape(
        1, 100, 1
    )  # Reshape into 3D for LSTM input
    n_steps = 10
    predictions = []

    for _ in range(n_steps):
        predicted_value_scaled = model.predict(last_100_days)
        predicted_value = scaler.inverse_transform(predicted_value_scaled)
        predictions.append(predicted_value[0, 0])

        # Update the input sequence with the new prediction
        last_100_days = np.append(
            last_100_days[:, 1:, :], predicted_value_scaled.reshape(1, 1, 1), axis=1
        )

    dates = pd.date_range(df["Date"].iloc[-1], periods=n_steps + 1, freq="D")[1:]

    plt.figure(figsize=(10, 6))

    # Plot the actual values (last part of the data)
    plt.plot(df["Date"], scaler.inverse_transform(scaled_data), label="Actual values")

    # Plot the predicted values
    plt.plot(
        dates,
        predictions,
        label=f"Predicted next {n_steps} values",
        color="orange",
        linestyle="--",
    )

    plt.title(f"{topic} Prediction for the Next {n_steps} Steps")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    main()
