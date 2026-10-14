"""Train TF-IDF + LogisticRegression on 90% of SST-2 train, sigmoid-calibrate on the held-out 10%."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common import ARTIFACTS, RESULTS, SEED, ece, load_sst2, reliability_points


def main():
    (X, y), _, (X_test, y_test) = load_sst2()
    X_fit, X_cal, y_fit, y_cal = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)

    base = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    base.fit(X_fit, y_fit)
    base.named_steps["tfidf"].stop_words_ = None  # pruned-term set is not needed at inference and is large
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit").fit(X_cal, y_cal)

    report = {"n_fit": len(X_fit), "n_calibration": len(X_cal), "n_eval": len(X_test)}
    for split, (Xs, ys) in {"calibration": (X_cal, y_cal), "sst2_val_second_half": (X_test, y_test)}.items():
        raw, cal = base.predict_proba(Xs)[:, 1], calibrated.predict_proba(Xs)[:, 1]
        report[split] = {
            "accuracy": round(float(((cal >= 0.5) == ys).mean()), 4),
            "ece_before": round(ece(raw, ys), 4),
            "ece_after": round(ece(cal, ys), 4),
        }

    fig, ax = plt.subplots(figsize=(5.2, 5), dpi=120)
    ax.plot([0.5, 1], [0.5, 1], ls="--", color="#8a8984", lw=1, label="Perfect calibration")
    raw, cal = base.predict_proba(X_test)[:, 1], calibrated.predict_proba(X_test)[:, 1]
    r = report["sst2_val_second_half"]
    for p, name, color in [(raw, f"Before (ECE {r['ece_before']:.3f})", "#eb6834"),
                           (cal, f"After sigmoid (ECE {r['ece_after']:.3f})", "#2a78d6")]:
        xs, ys, _ = reliability_points(p, y_test)
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=color, label=name)
    ax.set(xlim=(0.5, 1), ylim=(0.4, 1.0), xlabel="Predicted confidence", ylabel="Observed accuracy",
           title="Small model reliability (SST-2 val, second half)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "reliability.png")

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"calibrated": calibrated, "base": base}, ARTIFACTS / "small_model.joblib", compress=3)
    (ARTIFACTS / "calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
