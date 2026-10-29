import itertools, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS = ["Random Forest", "Gradient Boosting", "Logistic Regression"]

# ---------- demo data (loan approval, drift grows over time) ----------
def demo_data():
    def chunk(n, drift, seed):
        r = np.random.default_rng(seed)
        age = r.integers(18, 70, n); rural = (r.random(n) < 0.3).astype(int)
        income = r.normal(50000 - 9000 * drift * (age < 25), 15000, n).clip(10000)
        credit = r.normal(680, 60, n).clip(300, 850); debt = r.beta(2, 5, n)
        sc = (income - 50000) / 15000 + (credit - 680) / 60 - 3 * (debt - 0.28)
        sc = np.where((age < 25) & (rural == 1), sc * (1 - 2 * drift), sc)
        y = (sc + r.normal(0, 0.5, n) > 0).astype(int)
        return pd.DataFrame(dict(age=age, income=income.round(), credit_score=credit.round(), debt_ratio=debt.round(3), rural=rural, approved=y))
    parts = [chunk(4000, 0, 1)] + [chunk(1500, d, 10 + i) for i, d in enumerate([0, .4, .8, 1.0])]
    return pd.concat(parts, ignore_index=True)

# ---------- model helpers ----------
def _prep(X):
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]): X[c] = X[c].astype(str)
    return X

def make_model(kind, X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]; cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
                             ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
    est = {"Random Forest": RandomForestClassifier(150, min_samples_leaf=3, random_state=0, n_jobs=-1),
           "Gradient Boosting": GradientBoostingClassifier(random_state=0),
           "Logistic Regression": LogisticRegression(max_iter=500)}[kind]
    return Pipeline([("p", pre), ("m", est)])

def _predict(model, X):
    X = _prep(X); p = model.predict(X)
    try: conf = model.predict_proba(X).max(axis=1)
    except Exception: conf = None
    return p, conf

def run_train(df, target, feats, kind, ref_frac=0.5, n_periods=4):
    """Train on the early part of a time-ordered CSV; treat the rest as 'production' split into periods."""
    df = df.dropna(subset=[target]).reset_index(drop=True)
    n_ref = int(len(df) * ref_frac); k = int(n_ref * 0.8)
    ref, cur = df.iloc[:n_ref].copy(), df.iloc[n_ref:].copy()
    model = make_model(kind, _prep(ref[feats])).fit(_prep(ref[feats].iloc[:k]), ref[target].iloc[:k])
    for d in (ref, cur):
        d["_y"] = d[target].values; d["_pred"], c = _predict(model, d[feats])
        if c is not None: d["_conf"] = c
    cur["_period"] = np.concatenate([[i + 1] * len(a) for i, a in enumerate(np.array_split(np.arange(len(cur)), n_periods))])
    return dict(ref=ref, base=ref.iloc[k:], cur=cur, feats=feats, target=target, kind=kind)

def candidate_preds(ctx, period):
    """Retrain on reference + EARLIER periods only (labels arrive late), predict the chosen period."""
    d = pd.concat([ctx["ref"], ctx["cur"][ctx["cur"]._period < period]]); f, t = ctx["feats"], ctx["target"]
    m = make_model(ctx["kind"], _prep(d[f])).fit(_prep(d[f]), d[t])
    return _predict(m, ctx["cur"][ctx["cur"]._period == period][f])[0]

# ---------- monitoring ----------
def psi(ref, cur, bins=10):
    ref, cur = pd.Series(ref).dropna(), pd.Series(cur).dropna()
    if len(ref) == 0 or len(cur) == 0: return float("nan")
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > 10:
        q = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(q) < 3: return 0.0
        q[0], q[-1] = -np.inf, np.inf; a, b = np.histogram(ref, q)[0], np.histogram(cur, q)[0]
    else:
        cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        a = ref.astype(str).value_counts().reindex(cats, fill_value=0).values; b = cur.astype(str).value_counts().reindex(cats, fill_value=0).values
    pa, pb = a / a.sum() + 1e-4, b / b.sum() + 1e-4
    return float(np.sum((pb - pa) * np.log(pb / pa)))

def seg_frame(ref, df, cols, nb=4):
    out = {}
    for c in cols:
        if pd.api.types.is_numeric_dtype(ref[c]) and ref[c].nunique() > 8:
            q = np.unique(np.quantile(ref[c].dropna(), np.linspace(0, 1, nb + 1)))
            labs = [f"{c} <={q[1]:.3g}"] + [f"{c} {q[i]:.3g}-{q[i+1]:.3g}" for i in range(1, len(q) - 2)] + [f"{c} >{q[-2]:.3g}"]
            idx = np.digitize(df[c].values, q[1:-1]); out[c] = pd.Series([labs[i] if i < len(labs) else f"{c} missing" for i in idx])
        else: out[c] = (c + " = " + df[c].astype(str)).reset_index(drop=True)
    return pd.DataFrame(out)

def subgroups(seg, y, p, pairs, min_n):
    ok, rows = (y == p), []
    def add(name, m):
        if m.sum() >= min_n: rows.append(dict(group=name, n=int(m.sum()), accuracy=float(ok[m].mean())))
    for c in seg.columns:
        for v in seg[c].unique(): add(v, (seg[c] == v).values)
    if pairs:
        for a, b in itertools.combinations(seg.columns, 2):
            for va in seg[a].unique():
                for vb in seg[b].unique(): add(f"{va} & {vb}", ((seg[a] == va) & (seg[b] == vb)).values)
    return pd.DataFrame(rows, columns=["group", "n", "accuracy"]).sort_values("accuracy").reset_index(drop=True)

def analyse(ref, cur, seg_cols, drift_cols, base=None, gap=0.10, min_n=30, pairs=True):
    cur = cur.reset_index(drop=True); y, p = cur._y.values, cur._pred.values; ok = y == p
    base = ref if base is None else base; overall = float(ok.mean()); base_acc = float((base._y == base._pred).mean())
    seg = seg_frame(ref, cur, seg_cols); sub = subgroups(seg, y, p, pairs, min_n)
    drift = pd.DataFrame([(c, psi(ref[c], cur[c])) for c in drift_cols], columns=["feature", "psi"])
    drift["level"] = pd.cut(drift.psi, [-1, .1, .25, 99], labels=["stable", "moderate", "major"])
    cls = pd.DataFrame([(str(c), int((y == c).sum()), float((y == c).mean()), float(ok[y == c].mean())) for c in np.unique(y)], columns=["class", "n", "share", "recall"])
    alerts = []
    for _, r in sub.iterrows():
        g = overall - r.accuracy
        if g >= gap: alerts.append(dict(group=r.group, n=r.n, accuracy=r.accuracy, gap=g, severity="HIGH" if g >= 0.25 else "MEDIUM"))
    alerts = sorted(alerts, key=lambda a: -a["gap"])[:6]
    res = dict(overall=overall, base=base_acc, drift=drift, sub=sub, classes=cls, alerts=alerts, n=len(cur),
               worst=(sub.iloc[0].to_dict() if len(sub) else None), pred_shift=float(pd.Series(p).astype(str).value_counts(normalize=True).max() - pd.Series(base._pred).astype(str).value_counts(normalize=True).max()))
    if "_conf" in cur:
        res["conf"] = dict(mean=float(cur._conf.mean()), conf_errors=float(((cur._conf > .8) & ~ok).sum() / max(1, (~ok).sum())))
    wr = cur[~ok]; res["wrong"] = (wr.sort_values("_conf", ascending=False) if "_conf" in wr else wr).head(8)
    if "_cand" in cur:
        cp = cur._cand.values; cs = subgroups(seg, y, cp, pairs, min_n); co = float((y == cp).mean())
        cw, ow = (cs.accuracy.min() if len(cs) else co), (sub.accuracy.min() if len(sub) else overall)
        res["cand"] = dict(cur_overall=overall, new_overall=co, cur_worst=float(ow), new_worst=float(cw), safe=bool(cw > ow + .05 and co > overall - .01))
    res["rec"] = recommend(res); return res

def recommend(r):
    acts, major = [], (r["drift"].psi.max() if len(r["drift"]) else 0) >= .25
    if not r["alerts"] and not major and r["overall"] > r["base"] - .05:
        return ("OK", "No action needed. Keep monitoring.", acts)
    if not r["alerts"]:
        return ("WATCH", "Input data has drifted or accuracy slipped, but no single group is failing yet.", ["Monitor weekly.", "Collect labels for the drifted features."])
    w = r["alerts"][0]
    acts = [f"Send cases from '{w['group']}' to human review now.", "Tighten the approval/decision threshold for the failing groups."]
    if r.get("cand"):
        acts.append("A retrained candidate passed the safety check: stage it for deployment." if r["cand"]["safe"] else "Do NOT deploy the retrained candidate: it is not safer for the worst group.")
    else: acts.append("Provide a candidate model (or labels) to run the safe-retraining check.")
    if r.get("conf") and r["conf"]["conf_errors"] > .5: acts.append("Many wrong predictions are high-confidence: the model is overconfident, add a fallback model.")
    return ("ACT", f"Hidden failure: '{w['group']}' is {w['gap']*100:.0f} points below overall accuracy.", acts)

def report(r, title):
    L = [f"# BlindSpot Reliability Report: {title}", f"Cases analysed: {r['n']}", f"Baseline accuracy: {r['base']:.1%}", f"Current overall accuracy: {r['overall']:.1%}"]
    if r["worst"]: L.append(f"Worst group: {r['worst']['group']} ({r['worst']['accuracy']:.1%}, n={r['worst']['n']})")
    L += ["", "## Alerts"] + ([f"- [{a['severity']}] {a['group']}: {a['accuracy']:.1%} accuracy, {a['gap']*100:.0f} pts below overall (n={a['n']})" for a in r["alerts"]] or ["- None"])
    L += ["", "## Data drift (PSI)"] + [f"- {x.feature}: {x.psi:.3f} ({x.level})" for x in r["drift"].itertuples()]
    lv, msg, acts = r["rec"]; L += ["", f"## Recommendation: {lv}", msg] + [f"- {a}" for a in acts]
    return "\n".join(L)
