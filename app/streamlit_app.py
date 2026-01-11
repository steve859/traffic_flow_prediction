import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Traffic Dashboard", layout="wide")

st.title("Traffic Flow Prediction Dashboard")

output_dir = Path(st.secrets.get("OUTPUT_DIR", "") or st.sidebar.text_input("OUTPUT_DIR", "/app/data/consumer_output"))

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


st.sidebar.header("Inspect")
section = st.sidebar.selectbox("Section", ["raw", "agg", "pred"], index=1)
limit = st.sidebar.slider("Lines", min_value=10, max_value=200, value=50, step=10)

if section == "raw" and raw_files:
    f = st.sidebar.selectbox("File", [p.name for p in raw_files], index=len(raw_files) - 1)
    data = tail_jsonl(output_dir / f, limit)
    st.subheader(f"Raw tail: {f}")
    st.json(data)
elif section == "agg" and agg_files:
    f = st.sidebar.selectbox("File", [p.name for p in agg_files], index=len(agg_files) - 1)
    data = tail_jsonl(output_dir / f, limit)
    st.subheader(f"Agg tail: {f}")
    st.json(data)
elif section == "pred" and pred_files:
    f = st.sidebar.selectbox("File", [p.name for p in pred_files], index=len(pred_files) - 1)
    data = tail_jsonl(output_dir / f, limit)
    st.subheader(f"Pred tail: {f}")
    st.json(data)
else:
    st.info("No data yet. Start the Kafka producer + consumer.")
