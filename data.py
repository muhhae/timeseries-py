from confluent_kafka import Consumer
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


def get_data(topic):
    c = Consumer(
        {
            "bootstrap.servers": "localhost",
            "group.id": "pykafka_test_again",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
            "offset.store.method": "none",
        }
    )

    c.subscribe([topic])
    timestamps = []
    value = []

    for i in range(50000):
        msg = c.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print("Consumer error: {}".format(msg.error()))
            continue
        if msg.key() == b"timestamp":
            time_str = msg.value().decode("utf-8")
            try:
                time_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
                timestamps.append(time_obj)
            except:
                time_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                timestamps.append(time_obj)

        else:
            value.append(msg.value().decode("utf-8"))

    c.close()

    df = pd.DataFrame({"Date": timestamps, "Value": value})
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")
    df.reset_index(drop=True, inplace=True)
    return df


def visualize(df):
    df["Group"] = df.index // 100

    grouped_mean = df.groupby("Group")["Value"].mean()
    grouped_mean_dates = df.groupby("Group").first()["Date"]

    result = pd.DataFrame({"Date": grouped_mean_dates, "Mean Value": grouped_mean})

    plt.figure(figsize=(10, 6))
    plt.plot(result["Date"], result["Mean Value"], marker="o", color="g")
    plt.title("Graph")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    data = get_data("humid_1")
    visualize(data)


if __name__ == "__main__":
    main()
