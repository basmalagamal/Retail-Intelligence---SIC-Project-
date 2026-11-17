import json
import os
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query

from .recsys import CATEGORY_COL, ROOT, locate

MODEL_DIR = Path(os.getenv("CHURN_DIR", ROOT / "models"))
BASE = pd.Timestamp("2018-01-01")
CHURN_CLASS = 1
SAMPLE_N = int(os.getenv("CHURN_SAMPLE", 5000))
HIGH, MED = 0.7, 0.3
RANGES = {"age": (0, 120), "unique_article_ratio": (0, 1), "preferred_group_share": (0, 1)}
router = APIRouter(prefix="/api/churn")
state = {}


def group_of(n):
    if n in ("age", "fashion_news_frequency", "club_member_status"):
        return "Profile"
    if n.startswith(("preferred_", "product_group")) or n == "has_history_before_cutoff":
        return "Preferences"
    if "basket" in n:
        return "Baskets"
    if "last_" in n or n.startswith(("recent_", "previous_")) or n.endswith("_change_30d"):
        return "Recent activity"
    return "Lifetime activity"


def tier(p):
    return "high" if p >= HIGH else "medium" if p >= MED else "low"


class ChurnService:
    def __init__(self, rs):
        self.rs = rs
        self.pipe = joblib.load(MODEL_DIR / "random_forest_model.joblib")
        self.names = json.load(open(MODEL_DIR / "feature_names.json"))
        mp = MODEL_DIR / "evaluation_metrics.json"
        self.metrics = json.load(open(mp)) if mp.exists() else {}

        cp, tp = locate("customers_sampled.csv"), locate("transactions_sampled.csv")
        if cp is None or tp is None:
            raise FileNotFoundError("customers_sampled.csv / transactions_sampled.csv not found - run scripts/fetch_data.py")
        cu = pd.read_csv(cp, usecols=lambda c: c in {"customer_id", "age", "fashion_news_frequency", "club_member_status"})
        cu["customer_id"] = cu["customer_id"].astype(str).str.strip()
        self.cust = cu.drop_duplicates("customer_id").reset_index(drop=True)
        cidx = pd.Index(self.cust["customer_id"])
        art_idx = pd.Index(rs.ids)
        groups = rs.meta[CATEGORY_COL].astype(str)
        self.group_names = list(groups.unique())
        grp_code = pd.Categorical(groups, categories=self.group_names).codes.astype(np.int16)

        U, D, P, C, G, A = [], [], [], [], [], []
        for ch in pd.read_csv(tp, usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
                              dtype={"article_id": str, "customer_id": str}, chunksize=1_000_000):
            u = cidx.get_indexer(ch["customer_id"].str.strip())
            a = art_idx.get_indexer(ch["article_id"].str.strip().str.zfill(10))  # ids may lose leading zeros in CSV
            d = (pd.to_datetime(ch["t_dat"], errors="coerce") - BASE).dt.days.to_numpy(dtype=float)
            ok = (u >= 0) & np.isfinite(d)
            U.append(u[ok].astype(np.int32)); D.append(d[ok].astype(np.int16))
            P.append(pd.to_numeric(ch["price"], errors="coerce").fillna(0).to_numpy(np.float32)[ok])
            C.append(pd.to_numeric(ch["sales_channel_id"], errors="coerce").fillna(0).to_numpy(np.int8)[ok])
            G.append(np.where(a[ok] >= 0, grp_code[np.maximum(a[ok], 0)], -1).astype(np.int16))
            A.append(a[ok].astype(np.int32))
        u = np.concatenate(U); order = np.argsort(u, kind="stable")
        self.t_d, self.t_p, self.t_c, self.t_g, self.t_a = (np.concatenate(x)[order] for x in (D, P, C, G, A))
        self.off = np.searchsorted(u[order], np.arange(len(self.cust) + 1))
        self.cutoff = int(self.t_d.max()) + 1
        self.cidx = cidx

    def features(self, i):
        s, e = self.off[i], self.off[i + 1]
        d, p, ch, g, art = (a[s:e] for a in (self.t_d, self.t_p, self.t_c, self.t_g, self.t_a))
        d, p = d.astype(np.float64), p.astype(np.float64)
        c = self.cust.iloc[i]
        f = dict.fromkeys(self.names, 0.0)
        f.update(age=c.get("age", np.nan), has_history_before_cutoff=int(len(d) > 0), preferred_product_group="No prior history",
                 fashion_news_frequency=str(c["fashion_news_frequency"]) if pd.notna(c.get("fashion_news_frequency")) else "Unknown",
                 club_member_status=str(c["club_member_status"]) if pd.notna(c.get("club_member_status")) else "Unknown")
        if len(d) == 0:
            return f
        n, co = len(d), self.cutoff
        days, inv = np.unique(d, return_inverse=True)
        bsize, bspend = np.bincount(inv), np.bincount(inv, weights=p)
        f.update(purchase_days=len(days), total_items=n, unique_articles=len(np.unique(art)), total_spend=p.sum(), avg_price=p.mean(),
                 median_price=np.median(p), std_price=p.std(), channel_diversity=len(np.unique(ch)),
                 recency_days=co - d.max(), customer_tenure_days=co - d.min(), items_per_purchase_day=n / len(days),
                 spend_per_purchase_day=p.sum() / len(days), unique_article_ratio=len(np.unique(art)) / n,
                 avg_basket_size=bsize.mean(), max_basket_size=bsize.max(), avg_basket_spend=bspend.mean(), max_basket_spend=bspend.max())

        def win(lo, hi):
            m = (d >= co - hi) & (d < co - lo)
            return int(m.sum()), float(p[m].sum()), len(np.unique(d[m]))
        for k in (30, 90, 180):
            it, sp, pdays = win(0, k)
            f.update({f"items_last_{k}d": it, f"spend_last_{k}d": sp, f"purchase_days_last_{k}d": pdays,
                      f"unique_articles_last_{k}d": len(np.unique(art[(d >= co - k)]))})
        (ri, rs_, rd), (pi, ps, pd_) = win(0, 30), win(30, 60)
        f.update(recent_items=ri, recent_spend=rs_, recent_purchase_days=rd, previous_items=pi, previous_spend=ps,
                 previous_purchase_days=pd_, items_change_30d=ri - pi, spend_change_30d=rs_ - ps, purchase_days_change_30d=rd - pd_)
        known = g[g >= 0].astype(int)
        if len(known):
            cnt = np.bincount(known)
            f.update(product_group_diversity=int((cnt > 0).sum()), preferred_group_share=cnt.max() / len(known),
                     preferred_product_group=self.group_names[int(cnt.argmax())])
        return f

    def frame(self, idx):
        rows = [self.features(i) for i in idx]
        return pd.DataFrame(rows)[self.names]

    def predict(self, df):
        return self.pipe.predict_proba(df)[:, list(self.pipe.classes_).index(CHURN_CLASS)]

    @cached_property
    def sample(self):
        idx = np.random.default_rng(42).choice(len(self.cust), min(SAMPLE_N, len(self.cust)), replace=False)
        df = self.frame(idx)
        df.insert(0, "customer_id", self.cust["customer_id"].to_numpy()[idx])
        df.insert(1, "prob", self.predict(df[self.names]))
        return df

    def summary(self):
        s = self.sample
        tiers = s["prob"].map(tier).value_counts()
        hist = np.histogram(s["prob"], bins=10, range=(0, 1))[0]
        rf = self.pipe.steps[-1][1]
        imp = pd.Series(rf.feature_importances_, index=self.pipe.steps[0][1].get_feature_names_out())
        imp.index = [x.split("__", 1)[1] for x in imp.index]
        risky = s[s["prob"] >= HIGH].sort_values(["prob", "total_spend"], ascending=False).head(15)
        by = lambda col: [{"label": str(k), "avg": round(float(v), 3), "n": int(n)} for (k, v), n in
                          zip(s.groupby(col)["prob"].mean().items(), s.groupby(col)["prob"].size())]
        return {
            "scored": len(s), "customers_total": len(self.cust), "avg_prob": round(float(s["prob"].mean()), 3),
            "tiers": {t: int(tiers.get(t, 0)) for t in ("high", "medium", "low")}, "histogram": hist.tolist(),
            "by_news": by("fashion_news_frequency"), "by_club": by("club_member_status"),
            "drivers": [{"feature": k, "importance": round(float(v), 4)} for k, v in imp.nlargest(8).items()],
            "metrics": {k: round(self.metrics[k], 4) for k in ("accuracy", "roc_auc", "pr_auc") if k in self.metrics},
            "high_risk": [{"customer_id": r.customer_id, "prob": round(float(r.prob), 3), "total_spend": round(float(r.total_spend), 2),
                           "recency_days": int(r.recency_days), "news": r.fashion_news_frequency} for r in risky.itertuples()],
        }

    def customer(self, cid):
        i = self.cidx.get_indexer([cid])[0]
        if i < 0:
            raise KeyError(cid)
        return self.result(self.features(i), cid)

    def result(self, f, cid=None):
        p = float(self.predict(pd.DataFrame([f])[self.names])[0])
        rd = f["recency_days"]
        last = None if f["has_history_before_cutoff"] == 0 or pd.isna(rd) else (BASE + pd.Timedelta(days=self.cutoff - int(rd))).date().isoformat()
        num = lambda x: None if pd.isna(x) else round(float(x), 3)
        info = {"age": None if pd.isna(f["age"]) else int(f["age"]), "club_member_status": f["club_member_status"],
                "fashion_news_frequency": f["fashion_news_frequency"], "last_purchase": last,
                **{k: num(f[k]) for k in ("purchase_days", "total_items", "total_spend", "avg_basket_size", "recency_days",
                                          "customer_tenure_days", "items_last_30d", "items_last_90d", "spend_last_90d", "spend_change_30d")},
                "preferred_product_group": f["preferred_product_group"]}
        pref = f["preferred_product_group"]
        prods = self.rs.bestsellers(pref if pref != "No prior history" else None, 7)
        return {"customer_id": cid, "probability": round(p, 4), "tier": tier(p), "info": info,
                "recommendations": recommend(f, p), "suggested_products": prods}

    @cached_property
    def schema(self):
        sp = MODEL_DIR / "feature_schema.csv"
        dtypes = dict(pd.read_csv(sp).values) if sp.exists() else {}
        cats = {}
        for name, t, cols in self.pipe.steps[0][1].transformers_:
            if hasattr(t, "categories_"):
                cats.update({c: [str(x) for x in cs] for c, cs in zip(cols, t.categories_)})
        out = []
        for n in self.names:
            dt = dtypes.get(n, "object" if n in cats else "float64")
            lo, hi = RANGES.get(n, (None, None) if n.endswith("_change_30d") else (0, None))
            out.append({"name": n, "dtype": dt, "group": group_of(n),
                        "kind": "select" if n in cats or dt.startswith("int") else "number",
                        "options": cats.get(n) or (["0", "1"] if dt.startswith("int") else None), "min": lo, "max": hi})
        return out

    def validate(self, raw):
        errs, f, missing = {}, {}, []
        for k in set(raw) - set(self.names):
            errs[k] = "not a feature of this model"
        for sp in self.schema:
            n, v = sp["name"], raw.get(sp["name"])
            if v is None or (isinstance(v, str) and not v.strip()):
                missing.append(n); f[n] = None; continue
            if sp["kind"] == "select":
                v = str(int(v)) if isinstance(v, (bool, int, float)) and not isinstance(v, str) and float(v).is_integer() else str(v).strip()
                if v not in sp["options"]:
                    errs[n] = f"must be one of: {', '.join(sp['options'])}"
                f[n] = int(v) if sp["dtype"].startswith("int") and v in sp["options"] else v
            else:
                try:
                    if isinstance(v, bool):
                        raise ValueError
                    x = float(v)
                    if not np.isfinite(x):
                        raise ValueError
                except (TypeError, ValueError):
                    errs[n] = "must be a finite number"; continue
                if sp["min"] is not None and x < sp["min"] or sp["max"] is not None and x > sp["max"]:
                    errs[n] = f"must be between {sp['min'] if sp['min'] is not None else '-inf'} and {sp['max'] if sp['max'] is not None else 'inf'}"
                f[n] = x
        if errs:
            return None, errs, [], missing

        ti = f.get("total_items")
        f["has_history_before_cutoff"] = f["has_history_before_cutoff"] if f.get("has_history_before_cutoff") is not None else int(ti is None or ti > 0)
        for n in ("fashion_news_frequency", "club_member_status"):
            f[n] = f[n] if f[n] is not None else "Unknown"
        f["preferred_product_group"] = f["preferred_product_group"] or "No prior history"
        f = {n: (np.nan if f[n] is None else f[n]) for n in self.names}
        g = lambda n: f[n]
        warn = [m for bad, m in (
            (g("has_history_before_cutoff") == 0 and g("total_items") > 0, "Has history = 0 but total items > 0."),
            (g("has_history_before_cutoff") == 1 and g("total_items") == 0, "Has history = 1 but total items is 0."),
            (g("unique_articles") > g("total_items"), "Unique articles exceeds total items."),
            (g("purchase_days") > g("total_items"), "Purchase days exceeds total items."),
            (g("items_last_30d") > g("items_last_90d") or g("items_last_90d") > g("items_last_180d"), "Items in the last 30/90/180 days should not decrease with a longer window."),
            (g("customer_tenure_days") < g("recency_days"), "Tenure is shorter than days since last purchase.")) if bad]
        if len(missing) > len(self.names) / 2:
            warn.append(f"{len(missing)} of {len(self.names)} fields were left blank and treated as missing; the prediction is unreliable.")
        elif missing:
            warn.append(f"{len(missing)} blank field(s) treated as missing.")
        return f, errs, warn, missing

    def example(self):
        r = self.sample.sample(1, random_state=int(np.random.randint(1_000_000))).iloc[0]
        return {n: (None if (isinstance(r[n], float) and np.isnan(r[n])) else (r[n].item() if hasattr(r[n], "item") else r[n])) for n in self.names}


def recommend(f, p):
    t, out = tier(p), []
    add = lambda pr, title, detail: out.append({"priority": pr, "title": title, "detail": detail})
    pref = f["preferred_product_group"]
    if not f["has_history_before_cutoff"]:
        add("high", "Welcome / first-purchase series", "No purchases on record: send a welcome offer and best sellers to trigger a first order.")
    else:
        if f["recency_days"] >= 90:
            add("high", "Win-back campaign", f"Last purchase {int(f['recency_days'])} days ago. Send a personalised win-back email with a time-limited offer on {pref}.")
        elif t == "high" and pd.notna(f["recency_days"]):
            add("medium", "Proactive check-in", f"Flagged high risk but still recently active ({int(f['recency_days'])} days). Show new arrivals in {pref} before activity drops.")
        if f["spend_change_30d"] < 0 or f["items_change_30d"] < 0:
            add("medium", "Re-engage declining activity", "Spend/items fell versus the previous 30 days: offer a loyalty reward or free shipping on the next order.")
        if f["total_spend"] >= 2 and t != "low":
            add("high", "VIP retention", "High lifetime spend: assign a personal outreach (early access, exclusive perks) rather than a generic discount.")
        if f["avg_basket_size"] <= 1.5 and f["purchase_days"] >= 3:
            add("low", "Raise basket size", "Repeat buyer with small baskets: bundle offers on complementary items.")
    if f["fashion_news_frequency"] in ("NONE", "Unknown"):
        add("medium", "Grow the owned channel", "Not on the fashion-news list: invite to subscribe with a first-order incentive so future campaigns can reach them.")
    if f["club_member_status"] in ("PRE-CREATE", "LEFT CLUB", "Unknown"):
        add("low", "Re-activate club membership", f"Membership status is {f['club_member_status']}: promote club benefits at next touchpoint.")
    if not out:
        add("low", "Maintain and reward", "Low risk and healthy activity: keep regular recommendations flowing and consider loyalty perks.")
    return sorted(out, key=lambda r: {"high": 0, "medium": 1, "low": 2}[r["priority"]])


def svc():
    if "svc" not in state:
        raise HTTPException(503, f"Churn module unavailable: {state.get('error', 'not loaded')}")
    return state["svc"]


@router.get("/status")
def status():
    return {"ready": "svc" in state, "error": state.get("error")}


@router.get("/summary")
def summary():
    return svc().summary()


@router.get("/customers")
def search(q: str = "", limit: int = Query(10, ge=1, le=50)):
    ids = svc().cust["customer_id"]
    return ids[ids.str.startswith(q.strip().lower())].head(limit).tolist()


@router.get("/customer/{customer_id}")
def customer(customer_id: str):
    try:
        return svc().customer(customer_id.strip().lower())
    except KeyError:
        raise HTTPException(404, "unknown customer_id")


@router.get("/schema")
def schema():
    return {"fields": svc().schema}


@router.get("/example")
def example():
    return {"values": svc().example()}


@router.post("/predict")
def predict(body: dict = Body(...)):
    values = body.get("values")
    if not isinstance(values, dict):
        raise HTTPException(422, detail={"errors": {"_": "body must be {\"values\": {...}}"}})
    f, errs, warn, missing = svc().validate(values)
    if errs:
        raise HTTPException(422, detail={"errors": errs})
    return {**svc().result(f), "warnings": warn, "missing": missing}
