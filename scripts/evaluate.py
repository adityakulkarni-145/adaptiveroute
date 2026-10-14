"""Offline evaluation: run both models once per example, cache, then simulate every router setting.

Router training and threshold/cutoff tuning use SST-2 validation first half (see build_router.py).
All reported numbers come from SST-2 validation second half and the IMDB out-of-domain sample.
"""
import json
import time
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ARTIFACTS, CACHE, RESULTS, load_imdb_ood, load_sst2

THRESHOLDS = np.round(np.arange(0.50, 1.0001, 0.01), 2)
CUTOFFS = np.round(np.arange(0.0, 1.0001, 0.01), 2)
ACC_TOLERANCE = 0.01  # tuning rule: lowest escalation rate within 1 pp of the best tuning accuracy
DATASETS = {"sst2": "SST-2 validation, second half (in-domain)", "imdb": "IMDB test sample, first 1000 chars (out-of-domain)"}


def _run_models(texts, y):
    from app.models import LargeModel, SmallModel

    small, large = SmallModel(), LargeModel()
    for _ in range(3):  # warm-up
        small.predict(texts[0])
        large.predict(texts[0])
    rows = []
    for i, (t, label) in enumerate(zip(texts, y)):
        s0 = time.perf_counter()
        sp, p_raw = small.predict(t)
        s1 = time.perf_counter()
        lp = large.predict(t)
        s2 = time.perf_counter()
        rows.append({
            "text": t, "y": int(label),
            "small_p": sp.p_pos, "small_p_raw": p_raw, "small_pred": int(sp.p_pos >= 0.5),
            "small_conf": sp.confidence, "small_ms": (s1 - s0) * 1000,
            "large_p": lp.p_pos, "large_pred": int(lp.p_pos >= 0.5), "large_ms": (s2 - s1) * 1000,
        })
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(texts)}", flush=True)
    return pd.DataFrame(rows)


def load_cache() -> dict[str, pd.DataFrame]:
    """Predictions + per-model latency for sst2_tune, sst2 (test half) and imdb. Computed once."""
    CACHE.mkdir(exist_ok=True)
    paths = {k: CACHE / f"preds_{k}.pkl" for k in ("sst2_tune", "sst2", "imdb")}
    if not all(p.exists() for p in paths.values()):
        _, tune, test = load_sst2()
        sources = {"sst2_tune": tune, "sst2": test, "imdb": load_imdb_ood()}
        for k, (texts, y) in sources.items():
            if not paths[k].exists():
                print(f"Running both models on {k} ({len(texts)} examples)")
                _run_models(texts, y).to_pickle(paths[k])
    return {k: pd.read_pickle(p) for k, p in paths.items()}


def simulate(df: pd.DataFrame, escalate: np.ndarray, extra_ms: np.ndarray | float = 0.0) -> dict:
    escalate = np.asarray(escalate, dtype=bool)
    pred = np.where(escalate, df["large_pred"], df["small_pred"])
    lat = df["small_ms"].to_numpy() + extra_ms + escalate * df["large_ms"].to_numpy()
    ratio = df["large_ms"].mean() / df["small_ms"].mean()
    return {
        "accuracy": float((pred == df["y"]).mean()),
        "escalation_rate": float(escalate.mean()),
        "mean_latency_ms": float(lat.mean()),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "rel_cost": float(1 + escalate.mean() * ratio),
    }


def baselines(df: pd.DataFrame) -> dict:
    ratio = df["large_ms"].mean() / df["small_ms"].mean()
    small = simulate(df, np.zeros(len(df), bool))
    large = {
        "accuracy": float((df["large_pred"] == df["y"]).mean()),
        "escalation_rate": 1.0,
        "mean_latency_ms": float(df["large_ms"].mean()),
        "p95_latency_ms": float(np.percentile(df["large_ms"], 95)),
        "rel_cost": float(ratio),
    }
    return {"small_only": small, "large_only": large}


def fixed_sweep(df):
    return [{"param": float(t), **simulate(df, df["small_conf"].to_numpy() < t)} for t in THRESHOLDS]


def learned_sweep(df, scores, score_ms):
    return [{"param": float(c), **simulate(df, scores > c, score_ms)} for c in CUTOFFS]


def select(sweep: list[dict]) -> dict:
    best = max(r["accuracy"] for r in sweep)
    ok = [r for r in sweep if r["accuracy"] >= best - ACC_TOLERANCE]
    return min(ok, key=lambda r: (r["escalation_rate"], r["param"]))


def router_scores(df: pd.DataFrame, router) -> tuple[np.ndarray, np.ndarray]:
    from app.features import router_features

    scores, ms = [], []
    for t, pc, pr in zip(df["text"], df["small_p"], df["small_p_raw"]):
        s0 = time.perf_counter()
        x = np.array([router_features(t, pc, pr)])
        scores.append(router.predict_proba(x)[0, 1])
        ms.append((time.perf_counter() - s0) * 1000)
    return np.array(scores), np.array(ms)


def plot_tradeoff(out: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=120)
    for ax, (key, d) in zip(axes, out.items()):
        for name, sweep, color in [("Fixed router (threshold sweep)", d["fixed_sweep"], "#2a78d6"),
                                   ("Learned router (cutoff sweep)", d["learned_sweep"], "#eb6834")]:
            pts = sorted({(r["mean_latency_ms"], r["accuracy"]) for r in sweep})
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=2, color=color, label=name)
        for tag, color in [("fixed_tuned", "#2a78d6"), ("learned_tuned", "#eb6834")]:
            r = d[tag]
            ax.scatter(r["mean_latency_ms"], r["accuracy"], s=70, facecolor="white", edgecolor=color, lw=2, zorder=4)
        for tag, marker, label in [("small_only", "s", "Small only"), ("large_only", "D", "Large only")]:
            r = d[tag]
            ax.scatter(r["mean_latency_ms"], r["accuracy"], marker=marker, s=60, color="#3d3d3a", zorder=5)
            ax.annotate(label, (r["mean_latency_ms"], r["accuracy"]), textcoords="offset points",
                        xytext=(6, -12), fontsize=9, color="#3d3d3a")
        ax.set(title=DATASETS[key], xlabel="Mean latency per request (ms, CPU)", ylabel="Accuracy")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Accuracy vs latency (open circles: settings tuned on SST-2 val first half)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "tradeoff.png")


def error_analysis(df: pd.DataFrame, fixed_threshold: float):
    from app.features import CONTRAST, NEGATIONS, tokenize

    wrong = df[df["small_pred"] != df["y"]].sort_values("small_conf", ascending=False).head(15)
    names = ["negative", "positive"]
    lines = ["# Error analysis: most confident wrong small-model predictions", "",
             "Source: SST-2 validation second half. Confidence is the calibrated small-model confidence.", "",
             "| # | Text | True | Small (conf) | Large | Negation | Contrast |", "|---|---|---|---|---|---|---|"]
    neg = con = fixed_large = escalated = 0
    for i, r in enumerate(wrong.itertuples(), 1):
        toks = tokenize(r.text)
        has_neg = any(t in NEGATIONS or t.endswith("n't") for t in toks)
        has_con = any(t in CONTRAST for t in toks)
        neg += has_neg
        con += has_con
        fixed_large += r.large_pred == r.y
        escalated += r.small_conf < fixed_threshold
        text = r.text.replace("|", "\\|")
        lines.append(f"| {i} | {text} | {names[r.y]} | {names[r.small_pred]} ({r.small_conf:.3f}) | "
                     f"{names[r.large_pred]} | {'yes' if has_neg else ''} | {'yes' if has_con else ''} |")
    n = len(wrong)
    lines += ["", "## Measured summary", "",
              f"- Small-model errors on this split: {int((df['small_pred'] != df['y']).sum())} of {len(df)}.",
              f"- Of these {n} most confident errors: {neg} contain a negation word, {con} contain but/however/although.",
              f"- The large model gets {fixed_large} of {n} right.",
              f"- The tuned fixed router (threshold {fixed_threshold:.2f}) escalates {escalated} of {n}; "
              "any not escalated are answered wrongly by the small model with high confidence.", ""]
    (RESULTS / "errors.md").write_text("\n".join(lines))


def main():
    cache = load_cache()
    bundle = joblib.load(ARTIFACTS / "router.joblib")
    router = bundle["model"]

    tune = cache["sst2_tune"]
    tuned_threshold = select(fixed_sweep(tune))["param"]
    tuned_cutoff = float(bundle["cutoff"])
    print(f"Tuned on SST-2 val first half: threshold={tuned_threshold}, cutoff={tuned_cutoff}")

    out, rows = {}, []
    for key in DATASETS:
        df = cache[key]
        scores, score_ms = router_scores(df, router)
        d = baselines(df)
        d["fixed_sweep"], d["learned_sweep"] = fixed_sweep(df), learned_sweep(df, scores, score_ms)
        d["fixed_default"] = {"param": 0.85, **simulate(df, df["small_conf"].to_numpy() < 0.85)}
        d["fixed_tuned"] = {"param": tuned_threshold, **simulate(df, df["small_conf"].to_numpy() < tuned_threshold)}
        d["learned_tuned"] = {"param": tuned_cutoff, **simulate(df, scores > tuned_cutoff, score_ms)}
        d["n"] = len(df)
        d["latency_ratio"] = float(df["large_ms"].mean() / df["small_ms"].mean())
        out[key] = d
        for method in ("small_only", "large_only", "fixed_default", "fixed_tuned", "learned_tuned"):
            rows.append({"dataset": key, "method": method, "param": d[method].get("param"), **d[method]})
        for method in ("fixed_sweep", "learned_sweep"):
            rows += [{"dataset": key, "method": method, **r} for r in d[method]]

    pd.DataFrame(rows).round(5).to_csv(RESULTS / "results.csv", index=False)
    plot_tradeoff(out)
    error_analysis(cache["sst2"], tuned_threshold)

    def compact(sweep):
        return [[r["param"], round(r["accuracy"], 4), round(r["mean_latency_ms"], 3), round(r["escalation_rate"], 4)]
                for r in sweep]

    def summary(r):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "large_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "tuning": {"split": "SST-2 validation first half", "fixed_threshold": tuned_threshold,
                   "learned_cutoff": tuned_cutoff, "rule": "lowest escalation within 1 pp of best tuning accuracy",
                   "router_cv_auc": bundle.get("cv_auc")},
        "calibration": json.loads((ARTIFACTS / "calibration.json").read_text()),
        "datasets": {},
    }
    for key, d in out.items():
        routed = max(("fixed_tuned", "learned_tuned"), key=lambda m: (d[m]["accuracy"], -d[m]["rel_cost"]))
        payload["datasets"][key] = {
            "title": DATASETS[key], "n": d["n"], "latency_ratio": round(d["latency_ratio"], 2),
            **{m: summary(d[m]) for m in ("small_only", "large_only", "fixed_default", "fixed_tuned", "learned_tuned")},
            "best_routed": routed,
            "sweep_columns": ["param", "accuracy", "mean_latency_ms", "escalation_rate"],
            "fixed_sweep": compact(d["fixed_sweep"]), "learned_sweep": compact(d["learned_sweep"]),
        }
    (ARTIFACTS / "results.json").write_text(json.dumps(payload, indent=1))

    for key, d in out.items():
        print(f"\n{DATASETS[key]} (n={d['n']}, large/small latency ratio {d['latency_ratio']:.1f})")
        for m in ("small_only", "large_only", "fixed_default", "fixed_tuned", "learned_tuned"):
            r = d[m]
            print(f"  {m:14s} param={str(r.get('param', '')):5s} acc={r['accuracy']:.4f} esc={r['escalation_rate']:.3f} "
                  f"mean={r['mean_latency_ms']:.2f}ms p95={r['p95_latency_ms']:.2f}ms cost={r['rel_cost']:.2f}")


if __name__ == "__main__":
    main()
