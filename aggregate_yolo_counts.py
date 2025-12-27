"""
Aggregate YOLO traffic counts to 1-minute resolution
Author: Traffic Flow Prediction Project
Description:
- Input: YOLO inference results (CSV)
- Output:
    1) aggregated CSV (1-minute resolution)
    2) numpy tensor (T, N, F) for STGNN/STGCN
"""

import pandas as pd
import numpy as np
import argparse
import os

# --------------------------------------------------
# CONFIG (bạn có thể sửa ở đây)
# --------------------------------------------------

CAM_ORDER = [
    "cam1", "cam2", "cam3", "cam4",
    "cam5", "cam6", "cam7", "cam8"
]

FEATURES = ["motorbike", "car", "bus", "truck"]
AGG_METHOD = "mean"     # "mean" hoặc "median"
TIME_FREQ = "1min"      # window aggregation

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def aggregate_per_minute(df):
    # floor timestamp xuống phút
    df["minute"] = df["timestamp"].dt.floor(TIME_FREQ)

    # aggregate
    agg_df = (
        df
        .groupby(["camera_id", "minute"], as_index=False)
        .agg({f: AGG_METHOD for f in FEATURES})
    )

    return agg_df


def fill_missing_minutes(agg_df):
    # đảm bảo mỗi camera có đầy đủ phút
    agg_df = (
        agg_df
        .set_index("minute")
        .groupby("camera_id")
        .apply(lambda x: x.resample(TIME_FREQ).mean())
        .reset_index()
    )

    # forward fill
    agg_df[FEATURES] = agg_df[FEATURES].fillna(method="ffill")

    return agg_df


def build_stgnn_tensor(agg_df, use_total=False):
    minutes = sorted(agg_df["minute"].unique())
    T = len(minutes)
    N = len(CAM_ORDER)
    F = 1 if use_total else len(FEATURES)

    X = np.zeros((T, N, F), dtype=np.float32)

    for t, minute in enumerate(minutes):
        minute_df = agg_df[agg_df["minute"] == minute]

        for i, cam in enumerate(CAM_ORDER):
            row = minute_df[minute_df["camera_id"] == cam]
            if not row.empty:
                if use_total:
                    X[t, i, 0] = row[FEATURES].sum(axis=1).values[0]
                else:
                    X[t, i, :] = row[FEATURES].values[0]

    return X, minutes


def normalize_tensor(X):
    mean = X.mean()
    std = X.std() + 1e-6
    return (X - mean) / std, mean, std


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("📥 Loading data...")
    df = load_data(args.input_csv)

    print("🧮 Aggregating to 1-minute resolution...")
    agg_df = aggregate_per_minute(df)

    print("🧹 Filling missing minutes...")
    agg_df = fill_missing_minutes(agg_df)

    # save aggregated CSV
    agg_csv_path = os.path.join(args.output_dir, "traffic_1min_aggregated.csv")
    agg_df.to_csv(agg_csv_path, index=False)
    print(f"✅ Saved aggregated CSV: {agg_csv_path}")

    print("📐 Building STGNN tensor...")
    X, minutes = build_stgnn_tensor(
        agg_df,
        use_total=args.use_total
    )

    if args.normalize:
        print("📊 Normalizing tensor...")
        X, mean, std = normalize_tensor(X)
        np.save(os.path.join(args.output_dir, "norm_mean.npy"), mean)
        np.save(os.path.join(args.output_dir, "norm_std.npy"), std)

    # save numpy tensor
    npy_path = os.path.join(args.output_dir, "traffic_tensor.npy")
    np.save(npy_path, X)
    print(f"✅ Saved STGNN tensor: {npy_path}")

    print("\n🎉 DONE")
    print("Tensor shape:", X.shape)
    print("Format: (T, N, F)")
    print("T =", X.shape[0], "timesteps")
    print("N =", X.shape[1], "cameras")
    print("F =", X.shape[2], "features")


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to YOLO counts CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save aggregated results"
    )
    parser.add_argument(
        "--use_total",
        action="store_true",
        help="Use total vehicle count instead of per-class features"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize output tensor"
    )

    args = parser.parse_args()
    main(args)
