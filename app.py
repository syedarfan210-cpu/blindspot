import streamlit as st, pandas as pd
from monitor import *

st.set_page_config("BlindSpot", "🔍", layout="wide")

@st.cache_resource
def setup():
    tr = make_data(6000, 0.0, seed=1)
    weeks = {w: make_data(2500, d, seed=10 + w) for w, d in WEEK_DRIFT.items()}
    return tr, weeks, train(tr)

tr, weeks, model = setup()
st.title("🔍 BlindSpot: Your AI says 94% accurate. Who is it failing?")
st.caption("Reliability monitor for a deployed loan-approval model (simulated data)")
w = st.sidebar.slider("Simulate week of deployment", 1, 4, 1)
st.sidebar.info("Week 1 = normal. Weeks 2-4 = the real world drifts. Labels arrive one week late.")
res = analyse(model, tr, weeks[w], [weeks[i] for i in range(1, w)])

t1, t2, t3, t4 = st.tabs(["Overview", "Subgroups", "Alerts", "Recommendation"])
with t1:
    a, b, c = st.columns(3)
    a.metric("Overall accuracy", f"{res['overall']:.1%}")
    wg = res["table"][res["table"].n >= 30].iloc[0]
    b.metric("Worst group accuracy", f"{wg.accuracy:.1%}", f"{wg.accuracy - res['overall']:.1%}")
    c.metric("Approval-rate shift", f"{res['pred_shift']:+.1%}")
    st.subheader("Data drift (PSI per feature)")
    st.bar_chart(pd.Series(res["drift"]))
    st.caption("PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 major drift")
with t2:
    st.subheader("Accuracy by segment: the average hides this")
    st.bar_chart(res["table"].set_index("segment").accuracy)
    st.dataframe(res["table"], use_container_width=True)
with t3:
    if not res["alerts"]:
        st.success("No hidden subgroup failures detected.")
    for al in res["alerts"]:
        (st.error if al["severity"] == "HIGH" else st.warning)(f"**{al['severity']}**: {al['text']}")
    if res["alerts"]:
        st.subheader("Examples of wrong, high-confidence predictions")
        st.dataframe(res["bad"][["segment", "age", "income", "credit_score", "debt_ratio", "pred", "y", "conf"]]
                     .rename(columns={"pred": "model said", "y": "actual"}), use_container_width=True)
with t4:
    lvl, msg = res["rec"]
    {"OK": st.success, "WATCH": st.warning, "ACT": st.error}[lvl](msg)
    c = res["cand"]
    if c:
        st.subheader("Safe-retraining check: candidate vs current")
        st.table(pd.DataFrame({"Current": [c["cur_overall"], c["cur_worst"]],
                               "Candidate": [c["new_overall"], c["new_worst"]]},
                              index=["Overall accuracy", f"Worst group ({c['worst_seg']})"]).style.format("{:.1%}"))
        st.write("✅ Candidate passes safety check" if c["safe"] else "⛔ Candidate rejected: not safer for the worst group")
    else:
        st.info("No labelled history yet, so no candidate model can be trained.")
