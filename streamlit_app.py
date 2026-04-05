import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")
st.title("Credit Scoring — Altman Z'' + ML")

# Load scored data (run main.py first to generate this)
CSV = "scored_companies.csv"
if not os.path.exists(CSV):
    st.error("Run main.py first to generate scored_companies.csv")
    st.stop()

df = pd.read_csv(CSV)

# ── Top metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies Scored", len(df))
c2.metric("Distressed (Z'')", (df["zone"] == "Distress").sum())
c3.metric("Grey Zone",        (df["zone"] == "Grey Zone").sum())
c4.metric("High ML Risk (>50%)", (df["ml_pd_score"] > 0.5).sum())

st.divider()

# ── Main scatter: Z-Score vs ML PD ──────────────────────────────────────────
st.subheader("Z''-Score vs ML Probability of Default")
st.caption("The disagreement quadrant (top-right) is most interesting — safe by Z-Score, risky by ML.")

fig = px.scatter(
    df, x="zscore", y="ml_pd_score",
    color="zone",
    hover_data=["company", "sector", "X1", "X2", "X3", "X4"],
    color_discrete_map={
        "Safe":      "#22c55e",
        "Grey Zone": "#f59e0b",
        "Distress":  "#ef4444"
    },
    labels={"zscore": "Altman Z''-Score", "ml_pd_score": "ML Probability of Default"},
    height=500
)

# Add quadrant lines
fig.add_hline(y=0.5,  line_dash="dash", line_color="white", opacity=0.4)
fig.add_vline(x=2.6,  line_dash="dash", line_color="white", opacity=0.4)

# Annotation for disagreement quadrant
fig.add_annotation(x=3.5, y=0.75, text="⚠ Disagreement Zone",
                    font=dict(color="#f59e0b", size=11), showarrow=False)

fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ── Company table ────────────────────────────────────────────────────────────
st.subheader("All Companies — Ranked by ML Risk")

col1, col2 = st.columns([2, 1])
with col1:
    sector_filter = st.multiselect(
        "Filter by sector", options=df["sector"].unique(), default=df["sector"].unique()
    )
with col2:
    zone_filter = st.multiselect(
        "Filter by zone", options=["Safe", "Grey Zone", "Distress"],
        default=["Safe", "Grey Zone", "Distress"]
    )

filtered = df[df["sector"].isin(sector_filter) & df["zone"].isin(zone_filter)]
filtered_display = filtered[[
    "company", "sector", "zscore", "zone", "ml_pd_score",
    "X1", "X2", "X3", "X4"
]].sort_values("ml_pd_score", ascending=False)

st.dataframe(
    filtered_display.style.background_gradient(subset=["ml_pd_score"], cmap="RdYlGn_r"),
    hide_index=True,
    use_container_width=True
)

# ── Ratio breakdown by zone ──────────────────────────────────────────────────
st.subheader("Ratio Averages by Zone")
zone_avg = df.groupby("zone")[["X1", "X2", "X3", "X4"]].mean().reset_index()
fig2 = px.bar(
    zone_avg.melt(id_vars="zone", var_name="Ratio", value_name="Mean Value"),
    x="Ratio", y="Mean Value", color="zone", barmode="group",
    color_discrete_map={"Safe": "#22c55e", "Grey Zone": "#f59e0b", "Distress": "#ef4444"},
    template="plotly_dark", height=350
)
st.plotly_chart(fig2, use_container_width=True)