import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATS = ["age", "income", "credit_score", "debt_ratio", "rural"]
WEEK_DRIFT = {1: 0.0, 2: 0.4, 3: 0.8, 4: 1.0}  # how much the world has changed

def make_data(n, drift=0.0, seed=0):
    r = np.random.default_rng(seed)
    age = r.integers(18, 70, n)
    rural = (r.random(n) < 0.3).astype(int)
    income = r.normal(50000 - 9000 * drift * (age < 25), 15000, n).clip(10000)
    credit = r.normal(680, 60, n).clip(300, 850)
    debt = r.beta(2, 5, n)
    score = (income - 50000) / 15000 + (credit - 680) / 60 - 3 * (debt - 0.28)
    yr = (age < 25) & (rural == 1)
    # concept drift: for young rural applicants the true repayment rule slowly inverts
    score = np.where(yr, score * (1 - 2 * drift), score)
    y = (score + r.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame(dict(age=age, income=income, credit_score=credit,
                             debt_ratio=debt, rural=rural, y=y))

def train(df):
    m = RandomForestClassifier(150, min_samples_leaf=5, random_state=0, n_jobs=-1)
    return m.fit(df[FEATS], df.y)

def psi(a, b, bins=10):
    q = np.quantile(a, np.linspace(0, 1, bins + 1)); q[0], q[-1] = -np.inf, np.inf
    pa = np.histogram(a, q)[0] / len(a) + 1e-6
    pb = np.histogram(b, q)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))

def segments(df):
    d = df.copy()
    d["ok"] = d.pred == d.y
    band = pd.cut(d.age, [17, 24, 39, 54, 100], labels=["18-24", "25-39", "40-54", "55+"])
    d["segment"] = band.astype(str) + " / " + np.where(d.rural == 1, "Rural", "Urban")
    return d

def seg_table(df):
    d = segments(df)
    t = d.groupby("segment").ok.agg(accuracy="mean", n="size").reset_index()
    return t.sort_values("accuracy")

def evaluate(model, df):
    d = df.copy(); d["pred"] = model.predict(d[FEATS])
    d["conf"] = model.predict_proba(d[FEATS]).max(axis=1)
    return d

def worst_group(d, min_n=30):
    t = seg_table(d); t = t[t.n >= min_n]
    return t.iloc[0]

def analyse(model, train_df, week_df, prior_weeks):
    d = evaluate(model, week_df)
    overall = d.ok.mean() if "ok" in d else (d.pred == d.y).mean()
    drift = {f: psi(train_df[f], d[f]) for f in FEATS if f != "rural"}
    pred_shift = d.pred.mean() - train_df.y.mean()
    tab = seg_table(d)
    alerts = []
    for _, r in tab[tab.n >= 30].iterrows():
        gap = overall - r.accuracy
        if gap > 0.10:
            sub = segments(d); sub = sub[sub.segment == r.segment]
            ref = train_df[(train_df.age < 25) & (train_df.rural == 1)] if r.segment == "18-24 / Rural" else train_df
            p = psi(ref.income, sub.income)
            alerts.append(dict(segment=r.segment, accuracy=r.accuracy, gap=gap, n=int(r.n),
                severity="HIGH" if gap > 0.25 else "MEDIUM",
                text=f"Accuracy for {r.segment} is {r.accuracy:.0%} vs {overall:.0%} overall "
                     f"({int(r.n)} cases). Income PSI for this group = {p:.2f}."))
    bad = segments(d); bad = bad[~bad.ok].sort_values("conf", ascending=False)
    cand = None
    if prior_weeks:  # labels arrive late: only earlier weeks are usable
        cdf = pd.concat([train_df] + prior_weeks)
        cm = train(cdf); c = evaluate(cm, week_df); c["ok"] = c.pred == c.y
        cw = worst_group(c); ow = worst_group(segments(d))
        cand = dict(cur_overall=overall, new_overall=c.ok.mean(),
                    cur_worst=ow.accuracy, new_worst=cw.accuracy, worst_seg=ow.segment,
                    safe=bool(cw.accuracy > ow.accuracy + 0.05 and c.ok.mean() > overall - 0.01))
    if not alerts and max(drift.values()) < 0.1:
        rec = ("OK", "No action needed. Keep monitoring.")
    elif not alerts:
        rec = ("WATCH", "Input drift detected but no group is failing yet. Monitor closely.")
    elif cand and cand["safe"]:
        rec = ("ACT", "Route the failing group to human review now. A retrained candidate passed the safety "
                      "check (better on the worst group, no overall loss), so it can be staged for deployment.")
    else:
        rec = ("ACT", "Route the failing group to human review and tighten its approval threshold. "
                      "Do NOT deploy a retrained model yet: the candidate did not pass the safety check.")
    return dict(overall=overall, drift=drift, pred_shift=pred_shift, table=tab,
                alerts=alerts, bad=bad.head(5), cand=cand, rec=rec)
