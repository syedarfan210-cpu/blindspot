import pandas as pd, numpy as np, streamlit as st
from blindspot_engine import *

st.set_page_config("BlindSpot v2", "🔍", layout="wide")
st.title("🔍 BlindSpot: reliability monitor for any AI model")
st.caption("Finds the hidden groups, rare cases and drifting inputs that a good average accuracy conceals.")

DEMO_FEATS = ["age", "income", "credit_score", "debt_ratio", "rural"]
src = st.sidebar.radio("Data source", ["Demo (loan approval)", "Train a model on my CSV", "I already have predictions (any AI model)"])
gap = st.sidebar.slider("Alert when a group is this many points below overall", 5, 40, 10) / 100
min_n = int(st.sidebar.number_input("Minimum group size", 10, 500, 30))
pairs = st.sidebar.checkbox("Also search combinations of two features", True)

@st.cache_resource(show_spinner="Training model...")
def train_cached(df, target, feats, kind, ref_frac, n_periods):
    return run_train(df, target, list(feats), kind, ref_frac, n_periods)

@st.cache_resource(show_spinner="Training safe-retrain candidate...")
def cand_cached(key, period, _ctx):
    return candidate_preds(_ctx, period)

def seg_defaults(df, feats):
    ok = [c for c in feats if df[c].nunique() >= 2]
    return sorted(ok, key=lambda c: df[c].nunique())[:3]

ctx, key, title, seg_def, mode = None, "", "", [], "train"
if src.startswith("Demo"):
    df = demo_data(); kind = st.sidebar.selectbox("Model to monitor", MODELS)
    ctx = train_cached(df, "approved", tuple(DEMO_FEATS), kind, 0.4, 4); key = f"demo{kind}"
    title, seg_def = "Loan approval demo", ["age", "rural"]
    st.sidebar.download_button("Download demo CSV", df.to_csv(index=False), "demo_loans.csv")
elif src.startswith("Train"):
    up = st.file_uploader("Upload a CSV with one row per case, oldest rows first", type="csv")
    if up:
        df = pd.read_csv(up); st.write(f"{len(df)} rows, {len(df.columns)} columns")
        target = st.selectbox("Column the model should predict (a category, e.g. approved yes/no)", df.columns, index=len(df.columns) - 1)
        feats = st.multiselect("Input features", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target][:12])
        c1, c2, c3 = st.columns(3); kind = c1.selectbox("Model", MODELS)
        ref_frac = c2.slider("Share of rows used as the healthy baseline", 0.3, 0.7, 0.5); n_per = c3.slider("Production periods", 2, 8, 4)
        if df[target].nunique() > 20: st.error("This column has too many distinct values. BlindSpot needs a category column.")
        elif feats and st.button("Run BlindSpot", type="primary"): st.session_state["go"] = (len(df), target, tuple(feats), kind, ref_frac, n_per)
        g = st.session_state.get("go")
        if g and g[0] == len(df) and g[1] == target:
            ctx = train_cached(df, g[1], g[2], g[3], g[4], g[5]); key = str(g); title = up.name; seg_def = seg_defaults(df, list(g[2]))
else:
    mode = "pred"
    c1, c2 = st.columns(2)
    a = c1.file_uploader("Reference data: when the model was healthy", type="csv"); b = c2.file_uploader("Current production data", type="csv")
    if a and b:
        ref, cur = pd.read_csv(a), pd.read_csv(b); common = [c for c in ref.columns if c in cur.columns]
        if len(common) < 2: st.error("The two files need at least 2 shared columns.")
        else:
            d1, d2, d3, d4, d5 = st.columns(5)
            yc = d1.selectbox("True label column", common); pc = d2.selectbox("Model prediction column", common, index=min(1, len(common) - 1))
            cc = d3.selectbox("Confidence column (optional)", ["(none)"] + common); nc = d4.selectbox("Candidate model predictions (optional)", ["(none)"] + common)
            pe = d5.selectbox("Time/period column (optional)", ["(none)"] + common)
            used = [x for x in (yc, pc, cc, nc, pe) if x != "(none)"]; feats = [c for c in common if c not in used]
            for d in (ref, cur):
                d.dropna(subset=[yc, pc], inplace=True); d["_y"], d["_pred"] = d[yc].astype(str), d[pc].astype(str)
                if cc != "(none)": d["_conf"] = pd.to_numeric(d[cc], errors="coerce")
            if nc != "(none)": cur["_cand"] = cur[nc].astype(str)
            cur["_period"] = cur[pe] if pe != "(none)" else "All data"
            ctx = dict(ref=ref.reset_index(drop=True), base=ref.reset_index(drop=True), cur=cur.reset_index(drop=True), feats=feats)
            key, title, seg_def = "pred" + a.name + b.name, b.name, seg_defaults(ref, feats)

if ctx is None:
    st.info("Pick a data source in the sidebar and load your data. The demo needs nothing."); st.stop()

feats = ctx["feats"]
seg_cols = st.sidebar.multiselect("Groups to check (features)", feats, default=[c for c in seg_def if c in feats])
drift_cols = st.sidebar.multiselect("Features to check for drift", feats, default=feats[:12])
if not seg_cols: st.warning("Choose at least one feature to check for hidden group failures."); st.stop()
periods = sorted(ctx["cur"]["_period"].unique(), key=str)
period = st.sidebar.select_slider("Period of deployment", options=periods) if len(periods) > 1 else periods[0]

def period_df(p):
    d = ctx["cur"][ctx["cur"]._period == p].copy()
    if mode == "train" and isinstance(p, (int, np.integer)) and p > 1 and "_cand" not in d: d["_cand"] = cand_cached(key, int(p), ctx)
    return d
args = dict(seg_cols=seg_cols, drift_cols=drift_cols, base=ctx["base"], gap=gap, min_n=min_n, pairs=pairs)
res = analyse(ctx["ref"], period_df(period), **args)
nice = lambda d: d.rename(columns={"_y": "actual", "_pred": "model said", "_conf": "confidence"})

t1, t2, t3, t4, t5 = st.tabs(["Overview", "Groups", "Alerts", "Recommendation", "Report"])
with t1:
    m = st.columns(4)
    m[0].metric("Overall accuracy", f"{res['overall']:.1%}", f"{(res['overall'] - res['base']) * 100:+.1f} pts vs baseline")
    if res["worst"]: m[1].metric("Worst group accuracy", f"{res['worst']['accuracy']:.1%}", f"{(res['worst']['accuracy'] - res['overall']) * 100:+.1f} pts vs overall", help=res["worst"]["group"])
    m[2].metric("Cases this period", res["n"])
    if "conf" in res: m[3].metric("Wrong but confident", f"{res['conf']['conf_errors']:.0%}", help="Share of errors made with confidence above 80%")
    if len(periods) > 1:
        st.subheader("The average hides the failure")
        trend = pd.DataFrame([(str(p), analyse(ctx["ref"], ctx["cur"][ctx["cur"]._period == p], **args)) for p in periods], columns=["p", "r"])
        st.line_chart(pd.DataFrame({"Overall accuracy": [r["overall"] for r in trend.r], "Worst group accuracy": [r["worst"]["accuracy"] if r["worst"] else np.nan for r in trend.r]}, index=trend.p))
    st.subheader("Data drift (PSI per feature)")
    if len(res["drift"]): st.bar_chart(res["drift"].set_index("feature").psi); st.dataframe(res["drift"], use_container_width=True)
    st.caption("PSI below 0.1 is stable, 0.1 to 0.25 moderate, above 0.25 major drift.")
with t2:
    st.subheader("Accuracy by group (lowest first)")
    if len(res["sub"]): st.bar_chart(res["sub"].head(15).set_index("group").accuracy); st.dataframe(res["sub"], use_container_width=True)
    st.subheader("Rare cases: how well each class is caught")
    st.dataframe(res["classes"], use_container_width=True)
with t3:
    if not res["alerts"]: st.success("No hidden group failures detected for this period.")
    for al in res["alerts"]:
        (st.error if al["severity"] == "HIGH" else st.warning)(f"**{al['severity']}**: {al['group']} has {al['accuracy']:.0%} accuracy, {al['gap'] * 100:.0f} points below overall ({al['n']} cases).")
    if res["alerts"]:
        st.subheader("Examples of wrong predictions" + (" (most confident first)" if "conf" in res else ""))
        cols = [c for c in feats + ["_y", "_pred", "_conf"] if c in res["wrong"].columns]; st.dataframe(nice(res["wrong"][cols]), use_container_width=True)
with t4:
    lv, msg, acts = res["rec"]; {"OK": st.success, "WATCH": st.warning, "ACT": st.error}[lv](msg)
    for a_ in acts: st.write("• " + a_)
    if res.get("cand"):
        c = res["cand"]; st.subheader("Safe-retraining check: candidate vs current")
        st.table(pd.DataFrame({"Current": [c["cur_overall"], c["cur_worst"]], "Candidate": [c["new_overall"], c["new_worst"]]}, index=["Overall accuracy", "Worst group accuracy"]).style.format("{:.1%}"))
        st.write("✅ Candidate passes: better on the worst group with no overall loss." if c["safe"] else "⛔ Candidate rejected: not safer for the worst group.")
with t5:
    txt = report(res, f"{title}, period {period}"); st.markdown(txt); st.download_button("Download report (.md)", txt, "blindspot_report.md")
