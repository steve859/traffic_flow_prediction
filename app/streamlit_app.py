import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st


def _safe_secret_get(key: str, default: str = "") -> str:
    try:
        # st.secrets may raise if no secrets.toml exists.
        return str(st.secrets.get(key, default))
    except Exception:
        return default


st.set_page_config(page_title="Traffic Dashboard", layout="wide")

st.title("Traffic Flow Prediction Dashboard")
st.caption("Graph visualization for 8 linked cameras (adjacency + correlation)")


DEFAULT_PROJECT_DIR = os.getenv("PROJECT_DIR", "/app")

DEFAULT_ADJ_PATH = os.getenv("ADJ_MATRIX_PATH", f"{DEFAULT_PROJECT_DIR}/adj_matrix.npy")
DEFAULT_SCALER_PATH = os.getenv("SCALER_PARAMS_PATH", f"{DEFAULT_PROJECT_DIR}/model/scaler_params.json")

DEFAULT_PG_HOST = os.getenv("POSTGRES_HOST", "postgres")
DEFAULT_PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DEFAULT_PG_DB = os.getenv("POSTGRES_DB", "traffic")
DEFAULT_PG_USER = os.getenv("POSTGRES_USER", "traffic")
DEFAULT_PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "traffic")
DEFAULT_PG_TABLE = os.getenv("POSTGRES_TABLE", "traffic_predictions")


@st.cache_data(show_spinner=False)
def _load_scaler_camera_order(scaler_path: str) -> List[str]:
    path = Path(scaler_path)
    if not path.exists():
        return [f"cam{i}" for i in range(1, 9)]
    data = json.loads(path.read_text(encoding="utf-8"))
    order = data.get("camera_order")
    if isinstance(order, list) and order:
        return list(map(str, order))
    return [f"cam{i}" for i in range(1, 9)]


@st.cache_data(show_spinner=False)
def _load_adj_matrix(adj_path: str) -> np.ndarray:
    path = Path(adj_path)
    if not path.exists():
        return np.zeros((8, 8), dtype=np.float32)
    A = np.load(str(path))
    A = np.asarray(A, dtype=np.float32)
    return A


def _pg_connect(
    host: str,
    port: int,
    db: str,
    user: str,
    password: str,
):
    return psycopg2.connect(host=host, port=port, dbname=db, user=user, password=password)


def _fetch_db_health(
    host: str,
    port: int,
    db: str,
    user: str,
    password: str,
    table: str,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], int]:
    query = f"""
        select
            max(created_at) as last_write,
            max(window_end) as max_window_end,
            count(*) as total_rows
        from {table}
    """
    with _pg_connect(host, port, db, user, password) as conn:
        df = pd.read_sql_query(query, conn)
    if df.empty:
        return None, None, 0

    last_write_raw = df.loc[0, "last_write"]
    max_window_end_raw = df.loc[0, "max_window_end"]
    total_rows = int(df.loc[0, "total_rows"] or 0)

    last_write = pd.to_datetime(last_write_raw, utc=False) if last_write_raw is not None else None
    max_window_end = pd.to_datetime(max_window_end_raw, utc=False) if max_window_end_raw is not None else None
    return last_write, max_window_end, total_rows


def _fetch_predictions_long(
    host: str,
    port: int,
    db: str,
    user: str,
    password: str,
    table: str,
    lookback_minutes: int,
    limit_rows: int,
) -> pd.DataFrame:
    # NOTE: When replaying historical data, filtering with `now() - interval` can return 0 rows.
    # We instead fetch the latest rows and apply lookback relative to the latest window_end.
    query = f"""
        select window_end, pred_time, camera_id, pred_flow
        from {table}
        order by window_end desc
        limit %s
    """
    with _pg_connect(host, port, db, user, password) as conn:
        df = pd.read_sql_query(query, conn, params=(int(limit_rows),))
    if df.empty:
        return df
    df["window_end"] = pd.to_datetime(df["window_end"], utc=False)
    df["pred_time"] = pd.to_datetime(df["pred_time"], utc=False)
    df["camera_id"] = df["camera_id"].astype(str)
    df["pred_flow"] = pd.to_numeric(df["pred_flow"], errors="coerce")
    df = df.dropna(subset=["window_end", "camera_id", "pred_flow"])

    latest = df["window_end"].max()
    if pd.notna(latest) and int(lookback_minutes) > 0:
        cutoff = latest - pd.Timedelta(minutes=int(lookback_minutes))
        df = df[df["window_end"] >= cutoff]

    return df.sort_values("window_end")


def _wide_from_long(df_long: pd.DataFrame, camera_order: List[str]) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()
    wide = (
        df_long.pivot_table(index="window_end", columns="camera_id", values="pred_flow", aggfunc="mean")
        .sort_index()
        .reindex(columns=camera_order)
    )
    return wide


def _correlation_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.empty or len(wide) < 2:
        return pd.DataFrame()
    corr = wide.corr(min_periods=2)
    return corr


def _graph_figure(
    camera_order: List[str],
    node_values: Dict[str, float],
    edge_weights: np.ndarray,
    edge_threshold: float,
    title: str,
) -> go.Figure:
    n = len(camera_order)
    if n == 0:
        return go.Figure()

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = {cam: (float(np.cos(a)), float(np.sin(a))) for cam, a in zip(camera_order, angles)}

    edge_x: List[float] = []
    edge_y: List[float] = []
    edge_w: List[float] = []

    W = np.asarray(edge_weights, dtype=np.float32)
    if W.shape != (n, n):
        W = np.zeros((n, n), dtype=np.float32)

    max_w = float(np.nanmax(W)) if np.isfinite(W).any() else 0.0
    norm = (max_w if max_w > 0 else 1.0)

    for i in range(n):
        for j in range(i + 1, n):
            w = float(W[i, j])
            if not np.isfinite(w):
                continue
            if w < edge_threshold:
                continue
            x0, y0 = coords[camera_order[i]]
            x1, y1 = coords[camera_order[j]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_w.append(w / norm)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(120,120,120,0.55)", width=2),
            hoverinfo="skip",
            name="edges",
        )
    )

    node_x = [coords[c][0] for c in camera_order]
    node_y = [coords[c][1] for c in camera_order]
    values = [float(node_values.get(c, 0.0)) for c in camera_order]

    vmin = float(np.nanmin(values)) if len(values) else 0.0
    vmax = float(np.nanmax(values)) if len(values) else 1.0
    denom = (vmax - vmin) if vmax != vmin else 1.0
    sizes = [12 + 28 * ((v - vmin) / denom) for v in values]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=camera_order,
            textposition="bottom center",
            marker=dict(
                size=sizes,
                color=values,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="flow"),
                line=dict(width=1, color="rgba(20,20,20,0.6)"),
            ),
            hovertemplate="%{text}<br>flow=%{marker.color:.2f}<extra></extra>",
            name="cameras",
        )
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        height=520,
    )
    return fig


with st.sidebar:
    st.header("Data")
    st.subheader("Postgres")
    pg_host = st.text_input("Host", DEFAULT_PG_HOST)
    pg_port = st.number_input("Port", min_value=1, max_value=65535, value=DEFAULT_PG_PORT, step=1)
    pg_db = st.text_input("DB", DEFAULT_PG_DB)
    pg_user = st.text_input("User", DEFAULT_PG_USER)
    pg_password = st.text_input("Password", DEFAULT_PG_PASSWORD, type="password")
    pg_table = st.text_input("Table", DEFAULT_PG_TABLE)

    lookback_minutes = st.slider("Lookback (minutes)", min_value=10, max_value=24 * 60, value=180, step=10)
    limit_rows = st.slider("Max rows", min_value=200, max_value=20000, value=5000, step=200)

    st.subheader("Graph")
    adj_path = st.text_input("Adjacency path", DEFAULT_ADJ_PATH)
    scaler_path = st.text_input("Scaler path", DEFAULT_SCALER_PATH)
    edge_threshold = st.slider("Edge threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    corr_power = st.slider("Correlation strength", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    st.subheader("Realtime")
    auto_refresh = st.toggle("Auto refresh", value=False)
    refresh_seconds = st.number_input("Refresh every (seconds)", min_value=1, max_value=60, value=5, step=1)
    refresh = st.button("Refresh")


camera_order = _load_scaler_camera_order(scaler_path)
A = _load_adj_matrix(adj_path)

tab_graph, tab_heat, tab_series, tab_debug = st.tabs(
    ["Graph", "Heatmaps", "Timeseries", "Debug JSONL"],
)

if refresh:
    st.cache_data.clear()


with st.expander("Status", expanded=True):
    try:
        last_write, max_window_end, total_rows = _fetch_db_health(
            pg_host,
            int(pg_port),
            pg_db,
            pg_user,
            pg_password,
            pg_table,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{total_rows:,}")
        c2.metric("Last write", "-" if last_write is None else str(last_write))
        c3.metric("Max window_end", "-" if max_window_end is None else str(max_window_end))
    except Exception as e:
        st.warning(f"Cannot query DB status: {e}")


with tab_graph:
    try:
        df_long = _fetch_predictions_long(
            pg_host,
            int(pg_port),
            pg_db,
            pg_user,
            pg_password,
            pg_table,
            lookback_minutes,
            limit_rows,
        )
    except Exception as e:
        st.error(f"Cannot query Postgres: {e}")
        df_long = pd.DataFrame()

    if df_long.empty:
        st.info("No predictions found yet in Postgres.")
    else:
        wide = _wide_from_long(df_long, camera_order)
        latest = wide.iloc[-1].to_dict() if not wide.empty else {}

        corr = _correlation_matrix(wide)
        corr_mat = corr.to_numpy(dtype=np.float32) if not corr.empty else np.zeros_like(A)

        # Combine adjacency with correlation to express "linked density relationship".
        # - adjacency says which cameras are connected
        # - correlation says how similar the flow dynamics are
        corr01 = np.nan_to_num(np.abs(corr_mat), nan=0.0, posinf=0.0, neginf=0.0)
        W = (1.0 - corr_power) * np.clip(A, 0.0, None) + corr_power * (np.clip(A, 0.0, None) * corr01)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = _graph_figure(
                camera_order=camera_order,
                node_values=latest,
                edge_weights=W,
                edge_threshold=float(edge_threshold),
                title="Camera graph (node color/size = latest predicted flow; edge = adjacency × correlation)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.metric("Windows", int(len(wide)))
            st.metric("Latest window_end", str(wide.index.max()))
            st.write("Latest predicted flow")
            st.dataframe(pd.Series(latest).rename("pred_flow"), use_container_width=True)


with tab_heat:
    left, right = st.columns(2)
    with left:
        st.subheader("Adjacency matrix")
        st.plotly_chart(
            go.Figure(
                data=go.Heatmap(
                    z=A,
                    x=camera_order,
                    y=camera_order,
                    colorscale="Blues",
                    colorbar=dict(title="A"),
                )
            ).update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10)),
            use_container_width=True,
        )
    with right:
        st.subheader("Correlation (pred_flow)")
        try:
            df_long = _fetch_predictions_long(
                pg_host,
                int(pg_port),
                pg_db,
                pg_user,
                pg_password,
                pg_table,
                lookback_minutes,
                limit_rows,
            )
            wide = _wide_from_long(df_long, camera_order)
            corr = _correlation_matrix(wide)
        except Exception:
            corr = pd.DataFrame()

        if corr.empty:
            st.info("Not enough points to compute correlation yet.")
        else:
            st.plotly_chart(
                go.Figure(
                    data=go.Heatmap(
                        z=corr.to_numpy(),
                        x=list(corr.columns),
                        y=list(corr.index),
                        zmin=-1,
                        zmax=1,
                        colorscale="RdBu",
                        colorbar=dict(title="corr"),
                    )
                ).update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10)),
                use_container_width=True,
            )


with tab_series:
    st.subheader("Predicted flow over time")
    try:
        df_long = _fetch_predictions_long(
            pg_host,
            int(pg_port),
            pg_db,
            pg_user,
            pg_password,
            pg_table,
            lookback_minutes,
            limit_rows,
        )
        wide = _wide_from_long(df_long, camera_order)
    except Exception as e:
        st.error(f"Cannot query Postgres: {e}")
        wide = pd.DataFrame()

    if wide.empty:
        st.info("No timeseries to plot yet.")
    else:
        st.line_chart(wide, use_container_width=True)


with tab_debug:
    st.subheader("Legacy JSONL inspector (consumer_output)")
    output_dir = Path(
        _safe_secret_get("OUTPUT_DIR", "")
        or st.text_input("OUTPUT_DIR", "/app/data/consumer_output")
    )
    raw_files = sorted(output_dir.glob("raw-*.jsonl"))
    agg_files = sorted(output_dir.glob("agg-*.jsonl"))
    pred_files = sorted(output_dir.glob("pred-*.jsonl"))

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw files", str(len(raw_files)))
    col2.metric("Agg files", str(len(agg_files)))
    col3.metric("Pred files", str(len(pred_files)))

    def tail_jsonl(path: Path, n: int = 50):
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    section = st.selectbox("Section", ["raw", "agg", "pred"], index=1)
    limit = st.slider("Lines", min_value=10, max_value=200, value=50, step=10)

    if section == "raw" and raw_files:
        f = st.selectbox("File", [p.name for p in raw_files], index=len(raw_files) - 1)
        data = tail_jsonl(output_dir / f, limit)
        st.subheader(f"Raw tail: {f}")
        st.json(data)
    elif section == "agg" and agg_files:
        f = st.selectbox("File", [p.name for p in agg_files], index=len(agg_files) - 1)
        data = tail_jsonl(output_dir / f, limit)
        st.subheader(f"Agg tail: {f}")
        st.json(data)
    elif section == "pred" and pred_files:
        f = st.selectbox("File", [p.name for p in pred_files], index=len(pred_files) - 1)
        data = tail_jsonl(output_dir / f, limit)
        st.subheader(f"Pred tail: {f}")
        st.json(data)
    else:
        st.info("No JSONL files yet. This tab is optional for debugging.")


# Optional auto-refresh at the very end so the UI renders first.
if auto_refresh:
    time.sleep(float(refresh_seconds))
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
