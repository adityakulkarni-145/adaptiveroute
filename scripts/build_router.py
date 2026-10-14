"""Train the learned router on SST-2 validation first half and tune its cutoff there.

Target: 1 if the large model is right AND the small model is wrong (escalation would help).
The cutoff is tuned on out-of-fold scores so the tuning half is not scored in-sample.
"""
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ARTIFACTS, SEED  # noqa: I001  (sets sys.path for app imports)
from app.features import FEATURE_NAMES, feature_matrix
from evaluate import learned_sweep, load_cache, select


def main():
    tune = load_cache()["sst2_tune"]
    X = feature_matrix(tune["text"], tune["small_p"], tune["small_p_raw"])
    y = ((tune["large_pred"] == tune["y"]) & (tune["small_pred"] != tune["y"])).to_numpy().astype(int)

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    cv_auc = float(roc_auc_score(y, oof))
    cutoff = select(learned_sweep(tune, oof, 0.0))["param"]
    model.fit(X, y)

    coefs = dict(zip(FEATURE_NAMES, np.round(model[-1].coef_[0], 4).tolist()))
    joblib.dump({"model": model, "cutoff": cutoff, "features": FEATURE_NAMES, "cv_auc": round(cv_auc, 4)},
                ARTIFACTS / "router.joblib")
    print(json.dumps({"n": len(y), "positives": int(y.sum()), "cv_auc": round(cv_auc, 4),
                      "cutoff": cutoff, "coefficients": coefs}, indent=2))


if __name__ == "__main__":
    main()
