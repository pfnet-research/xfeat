"""Groupby aggregation benchmark."""
import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cudf

from xfeat.helper import aggregation


def main():
    key_nunique = 5_000
    n_samples = 10_000_000

    records = []
    for i in range(1, 8):
        df = pd.DataFrame({
            "group_value": np.random.randint(-1, 1, n_samples * i),
            "group_key": np.random.randint(0, key_nunique, n_samples * i),
        })
        df_cudf = cudf.from_pandas(df)

        # Pandas version
        time_records = []
        for _ in range(5):
            t = time.process_time()
            aggregation(df, "group_key", ["group_value"], ["mean"])
            elapsed_time = time.process_time() - t
            time_records.append(elapsed_time)

        records.append({
            "n_samples": n_samples * i,
            "n_unique_keys": key_nunique,
            "process_time_mean": np.mean(time_records),
            "process_time_std": np.std(time_records),
            "method": "CPU-pandas",
        })

        # cuDF version
        time_records = []
        for _ in range(5):
            t = time.process_time()
            aggregation(df_cudf, "group_key", ["group_value"], ["mean"])
            elapsed_time = time.process_time() - t
            time_records.append(elapsed_time)

        records.append({
            "n_samples": n_samples * i,
            "n_unique_keys": key_nunique,
            "process_time_mean": np.mean(time_records),
            "process_time_std": np.std(time_records),
            "method": "GPU-cuDF",
        })

    pd.DataFrame(records).to_csv("./benchmark_feature_aggregation.csv", index=False)


def plot():
    df = pd.read_csv("benchmark_feature_aggregation.csv")
    df["n_samples"] = df["n_samples"] / 1000.0 / 1000.0 / 10.0

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title("Benchmark groupby aggregation (xfeat.helper.aggregate)",
              fontsize=24, pad=24)
    ax = sns.barplot(
        x="n_samples",
        y="process_time_mean",
        hue="method",
        data=df,
        ax=ax,
        palette=["#c7c7c7", "#ce76de"],
    )
    ax.set_ylabel("process time [sec, log]", fontsize=24)
    ax.set_xlabel("num samples [*1e7]", fontsize=24)
    ax.set_yscale("log")
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    ax.legend(loc=0, fontsize=20)
    fig.autofmt_xdate(rotation=20)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    sns.despine(left=True, bottom=True)

    plt.savefig("./_docs/benchmark_groupby_aggregation.png")


if __name__ == '__main__':
    main()
    plot()
