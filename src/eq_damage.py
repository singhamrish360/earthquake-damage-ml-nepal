"""
Earthquake damage-grade prediction from building attributes
(2015 Gorkha, Nepal earthquake - building structure survey data).

Author : Amrish Singh <singhamrish360@gmail.com>, ORCID 0009-0005-1584-2504
Usage  : python src/eq_damage.py            # full pipeline (download -> EDA -> models -> SHAP)
         python src/eq_damage.py --skip-cv  # faster run without the 5-fold CV block

All numbers written to results/ are produced by this script; nothing is hand-entered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

# Official source (NPC / Kathmandu Living Labs portal) - unreachable from our build machine
# in Oct 2026, so we use a byte-identical public GitHub mirror of the same CSV and pin its hash.
OFFICIAL_URL = "https://eq2015.npc.gov.np/#/download"
MIRROR_URL = "https://github.com/bh555/sc1015_z137_grp6/raw/main/csv_building_structure.zip"
CSV_NAME = "csv_building_structure.csv"
CSV_SHA256 = "f717ac0fe8ff95a9d84e25b4292a3140ef1255ce77bfc011156593579dfca7fd"

SEED = 42
GRADES = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"]

# Columns recorded AFTER the earthquake (or derived from the damage assessment itself).
# Using them would leak the answer, so they are dropped.
LEAKAGE_COLS = ["count_floors_post_eq", "height_ft_post_eq",
                "condition_post_eq", "technical_solution_proposed"]

NUM_COLS = ["count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq"]
CAT_COLS = ["land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
            "other_floor_type", "position", "plan_configuration"]
SUPER_COLS = ["has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone",
              "has_superstructure_stone_flag", "has_superstructure_cement_mortar_stone",
              "has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick",
              "has_superstructure_timber", "has_superstructure_bamboo",
              "has_superstructure_rc_non_engineered", "has_superstructure_rc_engineered",
              "has_superstructure_other"]
STRUCT_FEATURES = NUM_COLS + CAT_COLS + SUPER_COLS          # transferable, building-only
LOC_FEATURES = ["district_id"]                               # proxy for shaking intensity

PALETTE = ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]  # Grade 1 -> 5


# --------------------------------------------------------------------------- data
def download_data(force: bool = False) -> Path:
    import requests
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_RAW / CSV_NAME
    if csv_path.exists() and not force:
        return csv_path
    zpath = DATA_RAW / "csv_building_structure.zip"
    r = requests.get(MIRROR_URL, timeout=300)
    r.raise_for_status()
    zpath.write_bytes(r.content)
    with zipfile.ZipFile(zpath) as z:
        z.extract(CSV_NAME, DATA_RAW)
    return csv_path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_clean(csv_path: Path | None = None, verify_hash: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load the raw CSV and apply documented cleaning steps. Returns (df, cleaning_log)."""
    csv_path = csv_path or download_data()
    if verify_hash:
        got = sha256(csv_path)
        if got != CSV_SHA256:
            raise ValueError(f"SHA-256 mismatch: {got}")
    df = pd.read_csv(csv_path)
    log = {"raw_rows": int(len(df)), "raw_cols": int(df.shape[1])}

    df = df.drop(columns=LEAKAGE_COLS)
    n0 = len(df)
    df = df.dropna(subset=["damage_grade"])
    log["dropped_missing_target"] = int(n0 - len(df))
    n0 = len(df)
    df = df.dropna(subset=["position", "plan_configuration"])
    log["dropped_missing_features"] = int(n0 - len(df))

    # age_building == 999 is a sentinel ("unknown"), not a real age -> NaN
    log["age_999_set_to_nan"] = int((df["age_building"] == 999).sum())
    df.loc[df["age_building"] == 999, "age_building"] = np.nan

    df["other_floor_type"] = df["other_floor_type"].replace("TImber/Bamboo-Mud", "Timber/Bamboo-Mud")
    df["y"] = df["damage_grade"].str.replace("Grade ", "").astype(int) - 1   # 0..4

    for c in CAT_COLS + ["district_id"]:
        df[c] = df[c].astype("category")
    for c in SUPER_COLS:
        df[c] = df[c].astype("int8")
    df["count_floors_pre_eq"] = df["count_floors_pre_eq"].astype("int8")
    df["age_building"] = df["age_building"].astype("float32")
    df["plinth_area_sq_ft"] = df["plinth_area_sq_ft"].astype("float32")
    df["height_ft_pre_eq"] = df["height_ft_pre_eq"].astype("float32")
    log["clean_rows"] = int(len(df))
    return df.reset_index(drop=True), log


# --------------------------------------------------------------------------- EDA
def _stacked(ax, ct: pd.DataFrame, title: str, xlabel: str = "", rotate: int = 0, counts=None):
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(pct))
    for i, g in enumerate(pct.columns):
        ax.bar(range(len(pct)), pct[g].values, bottom=bottom, color=PALETTE[i],
               edgecolor="white", linewidth=0.4, label=GRADES[i])
        bottom += pct[g].values
    labels = [str(x) for x in pct.index]
    if counts is not None:
        labels = [f"{l}\n(n={counts[k]:,})" for l, k in zip(labels, pct.index)]
    ax.set_xticks(range(len(pct)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.set_ylabel("% of buildings"); ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=11); ax.set_xlabel(xlabel)


def make_eda_figures(df: pd.DataFrame) -> dict:
    FIG_DIR.mkdir(exist_ok=True)
    paths = {}

    # 1. damage distribution
    fig, ax = plt.subplots(figsize=(6, 3.6))
    vc = df["y"].value_counts().sort_index()
    ax.bar(GRADES, vc.values, color=PALETTE, edgecolor="k", linewidth=0.5)
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Number of buildings"); ax.set_ylim(0, vc.max() * 1.2)
    ax.set_title(f"Damage grade distribution (n = {len(df):,} buildings)")
    fig.tight_layout(); p = FIG_DIR / "fig1_damage_distribution.png"; fig.savefig(p, dpi=200); plt.close(fig)
    paths["damage_distribution"] = p

    # 2. foundation type + superstructure flags
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 2]})
    ct = pd.crosstab(df["foundation_type"], df["y"])
    order = (ct.div(ct.sum(axis=1), axis=0)[4]).sort_values().index
    _stacked(axes[0], ct.loc[order], "Damage by foundation type", rotate=25,
             counts=df["foundation_type"].value_counts())
    rows, counts = {}, {}
    for c in SUPER_COLS:
        name = c.replace("has_superstructure_", "").replace("_", " ")
        sub = df.loc[df[c] == 1, "y"]
        rows[name] = sub.value_counts().reindex(range(5), fill_value=0)
        counts[name] = len(sub)
    ct2 = pd.DataFrame(rows).T
    order2 = (ct2.div(ct2.sum(axis=1), axis=0)[4]).sort_values().index
    _stacked(axes[1], ct2.loc[order2], "Damage by superstructure material (flags can overlap)",
             rotate=30, counts=counts)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout(); p = FIG_DIR / "fig2_damage_by_foundation_superstructure.png"; fig.savefig(p, dpi=200); plt.close(fig)
    paths["foundation_superstructure"] = p

    # 3. age and number of floors
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = [-1, 5, 10, 20, 30, 50, 80, 1000]
    labels = ["0-5", "6-10", "11-20", "21-30", "31-50", "51-80", ">80"]
    age_bin = pd.cut(df["age_building"], bins=bins, labels=labels)
    ct = pd.crosstab(age_bin, df["y"])
    _stacked(axes[0], ct, "Damage by building age (years)", xlabel="Age at time of earthquake",
             counts=age_bin.value_counts(), rotate=30)
    fl = df["count_floors_pre_eq"].clip(upper=5).map(lambda v: "5+" if v >= 5 else str(v))
    ct = pd.crosstab(fl, df["y"])
    _stacked(axes[1], ct, "Damage by number of storeys (pre-earthquake)", xlabel="Storeys",
             counts=fl.value_counts())
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout(); p = FIG_DIR / "fig3_damage_by_age_floors.png"; fig.savefig(p, dpi=200); plt.close(fig)
    paths["age_floors"] = p
    return paths


# --------------------------------------------------------------------------- models
def make_logreg(max_iter: int = 300):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    pre = ColumnTransformer([
        ("num", num, NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT_COLS),
        ("bin", "passthrough", SUPER_COLS),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=max_iter, C=1.0))])


def make_lgbm(n_estimators: int = 600):
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", n_estimators=n_estimators, learning_rate=0.1,
                              num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                              n_jobs=4, verbose=-1)


def scores(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro")),   # == accuracy for single-label
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "within_one_grade": float(np.mean(np.abs(y_true - y_pred) <= 1)),
            "accuracy": float(accuracy_score(y_true, y_pred))}


def split(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=0.2, stratify=df["y"], random_state=SEED)


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split
    if n >= len(df):
        return df
    sub, _ = train_test_split(df, train_size=n, stratify=df["y"], random_state=seed)
    return sub


def cross_validate_models(train: pd.DataFrame, n_sub: int = 100_000, k: int = 5) -> dict:
    """Stratified k-fold CV on a stratified subsample of the TRAINING set (runtime control)."""
    from sklearn.model_selection import StratifiedKFold
    sub = stratified_subsample(train, n_sub)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    out = {"n_rows": int(len(sub)), "k": k}
    for name, maker, feats in [("logreg", make_logreg, STRUCT_FEATURES),
                               ("lgbm", lambda: make_lgbm(300), STRUCT_FEATURES)]:
        fold = []
        for tr, va in skf.split(sub, sub["y"]):
            m = maker(); m.fit(sub.iloc[tr][feats], sub.iloc[tr]["y"])
            fold.append(scores(sub.iloc[va]["y"], m.predict(sub.iloc[va][feats])))
        out[name] = {met: {"mean": float(np.mean([f[met] for f in fold])),
                           "std": float(np.std([f[met] for f in fold]))} for met in fold[0]}
    return out


def fit_final_models(train: pd.DataFrame, test: pd.DataFrame, n_logreg: int = 200_000) -> dict:
    """Train on the 80 % training split, evaluate once on the 20 % held-out test split."""
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    res, models = {}, {}

    # Logistic regression on a stratified subsample (runtime/memory); more data barely changes a linear model.
    t = time.time()
    lr_train = stratified_subsample(train, n_logreg)
    lr = make_logreg(); lr.fit(lr_train[STRUCT_FEATURES], lr_train["y"])
    p = lr.predict(test[STRUCT_FEATURES])
    res["logreg"] = {**scores(test["y"], p), "train_rows": int(len(lr_train)),
                     "fit_seconds": round(time.time() - t, 1),
                     "confusion": confusion_matrix(test["y"], p).tolist()}
    models["logreg"] = lr

    # LightGBM - building attributes only (main model), early stopping on an internal 10 % validation split
    import lightgbm as lgb
    for key, feats in [("lgbm", STRUCT_FEATURES), ("lgbm_plus_district", STRUCT_FEATURES + LOC_FEATURES)]:
        t = time.time()
        tr, va = train_test_split(train, test_size=0.1, stratify=train["y"], random_state=SEED)
        m = make_lgbm(1000)
        m.fit(tr[feats], tr["y"], eval_set=[(va[feats], va["y"])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict(test[feats])
        res[key] = {**scores(test["y"], p), "train_rows": int(len(tr)),
                    "best_iteration": int(m.best_iteration_), "fit_seconds": round(time.time() - t, 1),
                    "confusion": confusion_matrix(test["y"], p).tolist()}
        models[key] = m

    # Majority-class reference
    maj = int(train["y"].mode()[0])
    res["majority_class"] = scores(test["y"], np.full(len(test), maj))
    return res, models


def plot_confusions(res: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key, title in [(axes[0], "logreg", "Logistic regression"),
                           (axes[1], "lgbm", "LightGBM (building attributes)")]:
        cm = np.array(res[key]["confusion"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j]):,})", ha="center", va="center",
                        fontsize=7, color="white" if cmn[i, j] > 0.55 else "black")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels([f"G{i}" for i in range(1, 6)]); ax.set_yticklabels([f"G{i}" for i in range(1, 6)])
        ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade")
        ax.set_title(f"{title}\nmacro-F1 = {res[key]['macro_f1']:.3f}, micro-F1 = {res[key]['micro_f1']:.3f}",
                     fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.025, label="Row-normalised share (recall)")
    p = FIG_DIR / "fig4_confusion_matrices.png"; fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


# --------------------------------------------------------------------------- interpretability
def pretty(name: str) -> str:
    return (name.replace("has_superstructure_", "super: ").replace("_", " ")
            .replace("count floors pre eq", "storeys").replace("height ft pre eq", "height (ft)")
            .replace("plinth area sq ft", "plinth area (sq ft)").replace("age building", "age (yr)"))


def shap_analysis(model, test: pd.DataFrame, n: int = 4000) -> dict:
    import shap
    X = stratified_subsample(test, n, seed=7)[STRUCT_FEATURES]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)                 # shape (n, features, classes) in recent shap
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[0] == 5:        # older layout (classes, n, features)
        sv = np.transpose(sv, (1, 2, 0))
    mean_abs = np.abs(sv).mean(axis=(0, 2))
    imp = pd.Series(mean_abs, index=STRUCT_FEATURES).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    top = imp.head(15)[::-1]
    ax.barh([pretty(i) for i in top.index], top.values, color="#3b6ea5")
    ax.set_xlabel("mean |SHAP value| (log-odds), averaged over the 5 grades")
    ax.set_title("LightGBM: global feature importance (SHAP)")
    fig.tight_layout(); p1 = FIG_DIR / "fig5_shap_importance.png"; fig.savefig(p1, dpi=200); plt.close(fig)

    # Beeswarm for the Grade-5 (collapse) output
    Xd = X.copy()
    for c in CAT_COLS:
        Xd[c] = Xd[c].cat.codes
    plt.figure()
    shap.summary_plot(sv[:, :, 4], Xd.rename(columns=pretty), max_display=12, show=False, plot_size=(8, 5.5))
    plt.title("SHAP values for predicting Grade 5 (collapse)", fontsize=11)
    plt.tight_layout(); p2 = FIG_DIR / "fig6_shap_grade5_beeswarm.png"; plt.savefig(p2, dpi=200); plt.close()

    # Direction of effect on Grade-5 log-odds for key categorical levels (mean SHAP per level)
    direction = {}
    for c in ["foundation_type", "roof_type", "ground_floor_type", "other_floor_type"]:
        j = STRUCT_FEATURES.index(c)
        direction[c] = (pd.Series(sv[:, j, 4], index=X.index).groupby(X[c], observed=True).mean()
                        .round(3).to_dict())
    for c in ["has_superstructure_mud_mortar_stone", "has_superstructure_rc_engineered",
              "has_superstructure_rc_non_engineered", "has_superstructure_cement_mortar_brick",
              "has_superstructure_timber"]:
        j = STRUCT_FEATURES.index(c)
        direction[c] = (pd.Series(sv[:, j, 4], index=X.index).groupby(X[c]).mean().round(3).to_dict())
    return {"n_rows": int(len(X)), "mean_abs_shap": imp.round(4).to_dict(),
            "grade5_mean_shap_by_level": {k: {str(a): b for a, b in v.items()} for k, v in direction.items()},
            "fig_importance": str(p1), "fig_beeswarm": str(p2)}


def permutation_analysis(model, test: pd.DataFrame, n: int = 6000) -> dict:
    from sklearn.inspection import permutation_importance
    X = stratified_subsample(test, n, seed=11)
    r = permutation_importance(model, X[STRUCT_FEATURES], X["y"], scoring="f1_macro",
                               n_repeats=3, random_state=SEED, n_jobs=1)
    s = pd.DataFrame({"mean": r.importances_mean, "std": r.importances_std},
                     index=STRUCT_FEATURES).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    top = s.head(15)[::-1]
    ax.barh([pretty(i) for i in top.index], top["mean"], xerr=top["std"], color="#c2563a")
    ax.set_xlabel("Drop in macro-F1 when feature is shuffled")
    ax.set_title(f"LightGBM: permutation importance (n = {len(X):,} test rows, 3 repeats)")
    fig.tight_layout(); p = FIG_DIR / "fig7_permutation_importance.png"; fig.savefig(p, dpi=200); plt.close(fig)
    return {"n_rows": int(len(X)), "importance": s.round(4).to_dict(orient="index"), "fig": str(p)}


# --------------------------------------------------------------------------- main
def save_json(obj: dict, name: str = "metrics.json") -> Path:
    RES_DIR.mkdir(exist_ok=True)
    p = RES_DIR / name
    p.write_text(json.dumps(obj, indent=2))
    return p


def main(skip_cv: bool = False):
    t0 = time.time()
    RES_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)
    df, log = load_clean()
    print("cleaning:", log, flush=True)
    figs = make_eda_figures(df)
    train, test = split(df)
    out = {"cleaning": log, "n_train": int(len(train)), "n_test": int(len(test))}
    if not skip_cv:
        out["cv"] = cross_validate_models(train)
        print("cv:", json.dumps(out["cv"], indent=1), flush=True); save_json(out)
    res, models = fit_final_models(train, test)
    out["test"] = res
    save_json(out); print("test:", {k: round(v["macro_f1"], 4) for k, v in res.items()}, flush=True)
    plot_confusions(res)
    out["shap"] = shap_analysis(models["lgbm"], test); save_json(out)
    out["permutation"] = permutation_analysis(models["lgbm"], test)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    save_json(out)
    rows = []
    for k, label in [("majority_class", "Majority class (always Grade 5)"), ("logreg", "Logistic regression"),
                     ("lgbm", "LightGBM - building attributes"), ("lgbm_plus_district", "LightGBM + district_id")]:
        r = res[k]
        rows.append({"model": label, "macro_F1": round(r["macro_f1"], 3), "micro_F1": round(r["micro_f1"], 3),
                     "weighted_F1": round(r["weighted_f1"], 3), "within_1_grade": round(r["within_one_grade"], 3)})
    pd.DataFrame(rows).to_csv(RES_DIR / "test_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"done in {out['runtime_seconds']} s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cv", action="store_true")
    main(**vars(ap.parse_args()))
