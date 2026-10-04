"""
Predictive maintenance monitor.  Run from the maintenance/ folder:  streamlit run app.py
(Run python src/maintenance_pipeline.py first so outputs/ exists.)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parent / "outputs"

st.set_page_config(page_title="Machine Risk Monitor", layout="wide")
st.title("Machine Failure Risk Monitor")

pred = pd.read_csv(OUT / "test_predictions.csv")
tm = pd.read_csv(OUT / "test_metrics.csv").iloc[0]
comparison = pd.read_csv(OUT / "model_comparison.csv", index_col=0)
thr = float(tm["threshold"])

# --- Headline numbers (held-out engines) ------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recall (at-risk cycles caught)", f"{tm['recall']:.0%}")
c2.metric("Precision", f"{tm['precision']:.0%}")
c3.metric("Engines warned in time", f"{tm['engines_warned_in_time_pct']:.0f}%")
c4.metric("False-alarm cycles / engine", f"{tm['false_alarm_cycles_per_engine']:.1f}")

# --- Fleet view at a chosen "today" ----------------------------------------
st.subheader("Fleet status")
as_of = st.slider(
    "Simulated current cycle (move it to replay the fleet over time)",
    int(pred["cycle"].min()), int(pred["cycle"].max()), 150,
)
snap = pred[pred["cycle"] == as_of].copy()  # engines still running at that cycle
snap["status"] = np.select(
    [snap["risk"] >= thr, snap["risk"] >= 0.5 * thr], ["ALERT", "WATCH"], "OK"
)
snap = snap.sort_values("risk", ascending=False)

a, w, o = st.columns(3)
a.metric("ALERT", int((snap["status"] == "ALERT").sum()))
w.metric("WATCH", int((snap["status"] == "WATCH").sum()))
o.metric("OK", int((snap["status"] == "OK").sum()))
st.caption(f"{len(snap)} engines still running at cycle {as_of}. "
           f"ALERT = risk above {thr:.2f}; WATCH = above {0.5 * thr:.2f}.")
st.dataframe(
    snap[["unit", "cycle", "risk", "status"]].rename(columns={"unit": "engine"}),
    width="stretch", hide_index=True,
)
st.download_button(
    "Download alert list as CSV",
    snap[snap["status"] != "OK"].to_csv(index=False).encode("utf-8"),
    file_name=f"maintenance_alerts_cycle_{as_of}.csv",
    mime="text/csv",
)

# --- Single engine history -------------------------------------------------
st.subheader("Engine risk history")
unit = st.selectbox("Engine", sorted(pred["unit"].unique()))
e = pred[pred["unit"] == unit].set_index("cycle")[["risk"]].copy()
e["alert threshold"] = thr
st.line_chart(e)
st.caption(f"This engine failed at cycle {int(pred.loc[pred['unit'] == unit, 'cycle'].max())}.")

# --- Model comparison + plots ----------------------------------------------
st.subheader("Model comparison (validation engines)")
st.dataframe(comparison, width="stretch")

plots = [p for p in ["pr_curve.png", "risk_over_time.png", "feature_importance.png"]
         if (OUT / p).exists()]
if plots:
    st.subheader("Plots")
    cols = st.columns(len(plots))
    for col, p in zip(cols, plots):
        col.image(str(OUT / p), caption=p.replace("_", " ").replace(".png", "").title())