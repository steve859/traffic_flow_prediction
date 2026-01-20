import json
import os
from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Add project code to import path (mounted at /opt/project in Docker)
import sys

PROJECT_DIR = os.getenv("PROJECT_DIR", "/opt/project")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.stgcn_model import STGCN, load_adj_tensor, load_stgcn_state_dict  # noqa: E402


KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "traffic_counts_raw")

WINDOW_SECONDS = int(os.getenv("WINDOW_SECONDS", "60"))
INPUT_LEN = int(os.getenv("STGCN_INPUT_LEN", "4"))
HORIZON = int(os.getenv("STGCN_HORIZON", "1"))

MODEL_PATH = os.getenv("STGCN_MODEL_PATH", f"{PROJECT_DIR}/model/stgcn_hcm_final.pt")
ADJ_PATH = os.getenv("ADJ_MATRIX_PATH", f"{PROJECT_DIR}/adj_matrix.npy")
SCALER_PATH = os.getenv("SCALER_PARAMS_PATH", f"{PROJECT_DIR}/model/scaler_params.json")

POSTGRES_URL = os.getenv("POSTGRES_JDBC_URL", "jdbc:postgresql://postgres:5432/traffic")
POSTGRES_USER = os.getenv("POSTGRES_USER", "traffic")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "traffic")
POSTGRES_TABLE = os.getenv("POSTGRES_TABLE", "traffic_predictions")

CHECKPOINT_LOCATION = os.getenv("SPARK_CHECKPOINT_LOCATION", "/opt/spark-checkpoints/stgcn")

SESSION_TZ = os.getenv("TZ", "Asia/Ho_Chi_Minh")


with open(SCALER_PATH, "r", encoding="utf-8") as f:
    _scaler = json.load(f)

CAMERA_ORDER: List[str] = list(_scaler["camera_order"])
MEAN = np.array(_scaler["mean"], dtype=np.float32)
SCALE = np.array(_scaler["scale"], dtype=np.float32)


_model = None  # type: Optional[STGCN]
_A_tensor = None  # type: Optional[torch.Tensor]


def _get_model(num_nodes: int) -> Tuple[STGCN, torch.Tensor]:
    global _model, _A_tensor

    if _model is None:
        device = "cpu"
        _model = STGCN(num_nodes=num_nodes, in_channels=1, horizon=HORIZON).to(device)
        state_dict = load_stgcn_state_dict(MODEL_PATH, device=device)
        _model.load_state_dict(state_dict, strict=False)
        _model.eval()

    if _A_tensor is None:
        _A_tensor = load_adj_tensor(ADJ_PATH, device="cpu")

    return _model, _A_tensor


INPUT_SCHEMA = T.StructType(
    [
        T.StructField("camera_id", T.StringType(), True),
        T.StructField("timestamp", T.StringType(), True),
        T.StructField("counts", T.MapType(T.StringType(), T.LongType()), True),
        T.StructField("total", T.LongType(), True),
        T.StructField("source", T.StringType(), True),
    ]
)


OUTPUT_SCHEMA = T.StructType(
    [
        T.StructField("window_end", T.TimestampType(), False),
        T.StructField("pred_time", T.TimestampType(), False),
        T.StructField("camera_id", T.StringType(), False),
        T.StructField("pred_flow", T.DoubleType(), False),
    ]
)


def _driver_state_path() -> str:
    return os.path.join(CHECKPOINT_LOCATION, "driver_state.json")


def _load_driver_state() -> Tuple[List[List[float]], Optional[pd.Timestamp]]:
    path = _driver_state_path()
    if not os.path.exists(path):
        return [], None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = [list(map(float, row)) for row in (data.get("history") or [])]
    last_ts = data.get("last_window_end")
    last_window_end = pd.to_datetime(last_ts) if last_ts else None
    return history, last_window_end


def _save_driver_state(history: List[List[float]], last_window_end: Optional[pd.Timestamp]) -> None:
    os.makedirs(CHECKPOINT_LOCATION, exist_ok=True)
    path = _driver_state_path()
    payload = {
        "history": history,
        "last_window_end": (last_window_end.isoformat() if last_window_end is not None else None),
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _write_to_postgres(batch_df, batch_id: int) -> None:  # noqa: ARG001
    (
        batch_df.write.format("jdbc")
        .option("url", POSTGRES_URL)
        .option("dbtable", POSTGRES_TABLE)
        .option("user", POSTGRES_USER)
        .option("password", POSTGRES_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .mode("append")
        .save()
    )


def _predict_and_write_batch(batch_df, batch_id: int) -> None:  # noqa: ARG001
    if batch_df.rdd.isEmpty():
        return

    history, last_window_end = _load_driver_state()

    pdf = batch_df.select("window_end", *CAMERA_ORDER).orderBy("window_end").toPandas()
    if pdf.empty:
        return

    out_rows: List[dict] = []

    for _, row in pdf.iterrows():
        window_end = pd.to_datetime(row["window_end"], utc=False)

        if last_window_end is not None and window_end <= last_window_end:
            continue

        raw_vec = np.array([float(row.get(cam, 0.0)) for cam in CAMERA_ORDER], dtype=np.float32)
        norm_vec = (raw_vec - MEAN) / np.where(SCALE == 0, 1.0, SCALE)

        history.append(norm_vec.tolist())
        if len(history) > INPUT_LEN:
            history = history[-INPUT_LEN:]

        if len(history) == INPUT_LEN:
            model, A = _get_model(num_nodes=len(CAMERA_ORDER))
            x = torch.tensor(np.array(history, dtype=np.float32)).unsqueeze(0).unsqueeze(-1)

            with torch.no_grad():
                delta = model(x, A)  # (1, N, horizon)

            delta_1 = delta[0, :, 0].cpu().numpy().astype(np.float32)  # (N,)
            last_step = np.array(history[-1], dtype=np.float32)

            pred_norm = last_step + delta_1
            pred = (pred_norm * SCALE) + MEAN
            pred_time = window_end + timedelta(seconds=WINDOW_SECONDS)

            for cam, v in zip(CAMERA_ORDER, pred.tolist()):
                out_rows.append(
                    {
                        "window_end": window_end.to_pydatetime(),
                        "pred_time": pred_time.to_pydatetime(),
                        "camera_id": cam,
                        "pred_flow": float(v),
                    }
                )

        last_window_end = window_end

    if out_rows:
        spark = batch_df.sparkSession
        out_pdf = pd.DataFrame(out_rows)
        out_sdf = spark.createDataFrame(out_pdf, schema=OUTPUT_SCHEMA)
        _write_to_postgres(out_sdf, batch_id)

    _save_driver_state(history, last_window_end)


def main() -> None:
    spark = (
        SparkSession.builder.appName("hcmc-stgcn-streaming")
        .config("spark.sql.session.timeZone", SESSION_TZ)
        .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", os.getenv("KAFKA_STARTING_OFFSETS", "latest"))
        .load()
    )

    parsed = (
        raw.select(F.col("timestamp").alias("kafka_ts"), F.col("value").cast("string").alias("value"))
        .select(F.from_json("value", INPUT_SCHEMA).alias("j"))
        .select("j.*")
    )

    with_time = parsed.withColumn(
        "event_time",
        F.coalesce(
            F.to_timestamp("timestamp"),
            F.current_timestamp(),
        ),
    )

    total_flow = with_time.withColumn(
        "total_flow",
        F.coalesce(
            F.col("total").cast("double"),
            F.expr("aggregate(map_values(counts), cast(0 as double), (acc, x) -> acc + x)"),
        ),
    )

    agg = (
        total_flow.groupBy(F.window("event_time", f"{WINDOW_SECONDS} seconds"), F.col("camera_id"))
        .agg(F.sum("total_flow").alias("flow"))
    )

    agg2 = (
        agg.select(
            F.col("window").getField("end").alias("window_end"),
            F.col("camera_id"),
            F.col("flow"),
        )
        .withWatermark("window_end", "2 minutes")
    )

    wide = (
        agg2.groupBy(F.col("window_end"))
        .pivot("camera_id", CAMERA_ORDER)
        .agg(F.first("flow"))
        .fillna(0.0)
        .select("window_end", *CAMERA_ORDER)
        .withColumn("key", F.lit("hcmc"))
    )

    query = (
        wide.writeStream.outputMode("update")
        .foreachBatch(_predict_and_write_batch)
        .option("checkpointLocation", CHECKPOINT_LOCATION)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
