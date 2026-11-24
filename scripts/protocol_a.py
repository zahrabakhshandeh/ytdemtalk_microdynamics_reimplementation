"""Protocol A: the released YT-DemTalk train / test split, Stats+PCA+RF, GBDT and AdaBoost (Table 1 of the paper).

Inside the training split: person-disjoint validation part (20%, one StratifiedGroupKFold(5) fold, seed = global).
Pipeline (scaler -> PCA 0.98 -> head) fit on the remaining 80%; temperature (on logit of the head probability) and
Youden threshold fit on the validation part; both frozen and applied to the test split. Nothing is fit on test.

Outputs results/protocol_A/: metrics.json, table1.md, predictions.csv, ablation_test.csv, fig3-fig7 *.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluate as E
from src import features as FT

PAPER = {
    "RF": {"AUROC": 0.953, "AP": 0.961, "F1": 0.851, "Acc": 0.857, "thr": 0.636, "AUROC_CI": [0.912, 0.984],
           "AP_CI": [0.928, 0.986], "recall": 0.771, "specificity": 0.953, "ECE": 0.268, "Brier": 0.168},
    "ADA": {"AUROC": 0.902, "AP": 0.919, "F1": 0.741, "Acc": 0.769, "thr": 0.690},
    "GBDT": {"AUROC": 0.893, "AP": 0.925, "F1": 0.844, "Acc": 0.846, "thr": 0.522},
}
COL = {"RF": "#1f4e79", "GBDT": "#c0392b", "ADA": "#7d8c2e"}
N_SEED_SPREAD = 10

def run_head(head, Xf, yf, Xv, yv, Xt, cfg, seed):
    pipe = E.fit_pipeline(E.make_pipeline(head, cfg, seed), Xf, yf, head)
    pv, pt = pipe.predict_proba(Xv)[:, 1], pipe.predict_proba(Xt)[:, 1]
    T = E.fit_temperature(pv, yv)
    pv_c, pt_c = E.apply_temperature(pv, T), E.apply_temperature(pt, T)
    thr, j = E.youden_threshold(pv_c, yv)
    return {"pipe": pipe, "T": T, "thr": thr, "youden_j_val": j, "pv": pv, "pv_c": pv_c, "pt": pt, "pt_c": pt_c,
            "n_pca": int(pipe.named_steps["pca"].n_components_)}


def fmt_ci(v, ci):
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    seed = cfg["seeds"]["global"]
    nb = cfg["evaluation"]["bootstrap_n"]
    od = ROOT / cfg["paths"]["results_dir"] / "protocol_A"
    od.mkdir(parents=True, exist_ok=True)
    fdir = ROOT / cfg["paths"]["features_dir"]
    clips = pd.read_csv(fdir / "clips.csv")
    info = pd.read_csv(fdir / "clip_info.csv")
    win = pd.read_parquet(fdir / "windows.parquet")
    F, cols = FT.clip_features(win, FT.CHANNELS, cfg["composition"]["eps"], cfg["features"]["min_windows_per_clip"])

    tr = clips[clips.orig_split == "train"].reset_index(drop=True)
    te = clips[clips.orig_split == "test"].reset_index(drop=True)
    fit_idx, val_idx = E.val_split(tr.person_id.to_numpy(), tr.y.to_numpy(), cfg, seed)
    fit, val = tr.iloc[fit_idx], tr.iloc[val_idx]
    assert not set(fit.person_id) & set(val.person_id) and not set(tr.person_id) & set(te.person_id)

    def part(d):
        return {"clips": int(len(d)), "dementia_clips": int(d.y.sum()), "control_clips": int((1 - d.y).sum()),
                "dementia_persons": int(d[d.y == 1].person_id.nunique()),
                "control_persons": int(d[d.y == 0].person_id.nunique()),
                "dementia_person_ids": sorted(d[d.y == 1].person_id.unique().tolist())}
    excl = info[~info.included]
    split_info = {"fit (80% of train)": part(fit), "validation (20% of train)": part(val), "test": part(te),
                  "excluded (< min windows)": excl[["video_id", "label", "person_id", "orig_split", "n_windows"]]
                  .to_dict("records")}

    def run_parts(parts, head, s):
        if parts == FT.CHANNELS:
            Fp, c = F, cols
        else:
            Fp, c = FT.clip_features(win, parts, cfg["composition"]["eps"], cfg["features"]["min_windows_per_clip"])
        X = lambda d: Fp.loc[d.video_id, c].to_numpy()
        return run_head(head, X(fit), fit.y.to_numpy(), X(val), val.y.to_numpy(), X(te), cfg, s)

    yt, gt = te.y.to_numpy(), te.person_id.to_numpy()
    res, metrics, preds = {}, {}, te[["video_id", "person_id", "label", "y"]].copy()
    for head in E.HEADS:
        r = run_parts(FT.CHANNELS, head, seed)
        res[head] = r
        m = E.point_metrics(r["pt_c"], yt, r["thr"], cfg["evaluation"]["ece_bins"])
        m.update(E.bootstrap_ci(r["pt_c"], yt, nb, seed))
        pc = E.bootstrap_ci(r["pt_c"], yt, nb, seed, groups=gt)
        m["AUROC_CI_person_cluster"], m["AP_CI_person_cluster"] = pc["AUROC_CI"], pc["AP_CI"]
        mv = E.point_metrics(r["pv_c"], val.y.to_numpy(), r["thr"], cfg["evaluation"]["ece_bins"])
        m["ECE_uncalibrated"] = E.ece(r["pt"], yt, cfg["evaluation"]["ece_bins"])
        m["Brier_uncalibrated"] = float(np.mean((r["pt"] - yt) ** 2))
        pp = pd.DataFrame({"g": gt, "y": yt, "p": r["pt_c"]}).groupby("g").mean()
        m["person_level_AUROC"] = float(roc_auc_score(pp.y, pp.p))
        m["person_level_n"] = {"dementia": int(pp.y.sum()), "control": int((1 - pp.y).sum())}
        metrics[head] = {"test": m, "validation": {k: mv[k] for k in ["n", "n_pos", "AUROC", "AP", "F1", "Acc"]},
                         "temperature": r["T"], "youden_j_val": r["youden_j_val"], "n_pca_components": r["n_pca"]}
        preds[f"p_raw_{head}"], preds[f"p_cal_{head}"] = r["pt"], r["pt_c"]
        print(head, {k: round(m[k], 3) for k in ["AUROC", "AP", "F1", "Acc", "thr", "recall", "specificity"]},
              "T", round(r["T"], 3), "PCA", r["n_pca"], flush=True)

    spread = []
    for s in range(N_SEED_SPREAD):
        r = run_parts(FT.CHANNELS, "RF", s)
        mm = E.point_metrics(r["pt_c"], yt, r["thr"])
        spread.append({"rf_seed": s, **{k: mm[k] for k in ["AUROC", "AP", "F1", "Acc", "thr"]}})
    spread = pd.DataFrame(spread)
    metrics["RF"]["rf_seed_spread"] = spread.describe().loc[["mean", "std", "min", "max"]].drop(columns="rf_seed").to_dict()


    abl = []
    for drop in [None] + FT.CHANNELS:
        parts = [c for c in FT.CHANNELS if c != drop]
        for s in [seed] + list(range(N_SEED_SPREAD)):
            r = run_parts(parts, "RF", s)
            abl.append({"removed": drop or "none (full)", "rf_seed": s, "primary": s == seed and len(abl) % (N_SEED_SPREAD + 1) == 0,
                        "test_AUROC": roc_auc_score(yt, r["pt_c"]), "val_AUROC": roc_auc_score(val.y, r["pv_c"])})
    abl = pd.DataFrame(abl)
    full = abl[abl.removed == "none (full)"]
    abl_sum = abl.groupby("removed", sort=False).agg(
        test_AUROC_seed42=("test_AUROC", "first"), test_AUROC_mean10=("test_AUROC", lambda x: x.iloc[1:].mean()),
        test_AUROC_sd10=("test_AUROC", lambda x: x.iloc[1:].std()), val_AUROC_seed42=("val_AUROC", "first"))
    abl_sum["drop_vs_full_seed42"] = full.test_AUROC.iloc[0] - abl_sum.test_AUROC_seed42
    abl_sum["drop_vs_full_mean10"] = full.test_AUROC.iloc[1:].mean() - abl_sum.test_AUROC_mean10
    abl_sum.round(4).to_csv(od / "ablation_test.csv")
    metrics["ablation_test"] = abl_sum.round(4).reset_index().to_dict("records")

    metrics["splits"] = split_info
    metrics["paper"] = PAPER
    metrics["settings"] = {"seed": seed, "bootstrap_n": nb, "val_fraction": cfg["calibration"]["val_fraction"],
                           "feature_rule": cfg["cuts"]["feature_rule"], "features": cols,
                           "min_windows_per_clip": cfg["features"]["min_windows_per_clip"]}
    (od / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    preds.to_csv(od / "predictions.csv", index=False)

    # ---- Table 1, paper vs ours
    L = ["# Protocol A (orig_split train/test): Table 1 format, paper vs ours", "",
         f"Test split: {len(te)} clips ({int(yt.sum())} dementia clips / {te[te.y == 1].person_id.nunique()} persons, "
         f"{int((1 - yt).sum())} controls). Validation part of train: {len(val)} clips "
         f"({int(val.y.sum())} dementia / {val[val.y == 1].person_id.nunique()} persons, {int((1 - val.y).sum())} controls). "
         f"Fit part: {len(fit)} clips. Threshold = Youden J on the validation part, after temperature scaling.", "",
         "| Method | | AUROC | AP | F1 | Acc | Thr |", "|---|---|---|---|---|---|---|"]
    for h, name in E.HEADS.items():
        p, m = PAPER[h], metrics[h]["test"]
        L.append(f"| {name} | paper | {p['AUROC']:.3f} | {p['AP']:.3f} | {p['F1']:.3f} | {p['Acc']:.3f} | {p['thr']:.3f} |")
        L.append(f"| | **ours** | **{m['AUROC']:.3f}** | **{m['AP']:.3f}** | **{m['F1']:.3f}** | **{m['Acc']:.3f}** | **{m['thr']:.3f}** |")
    L += ["", "Extended metrics (ours, test split; paper values where reported):", "",
          "| Method | AUROC [95% CI] | AP [95% CI] | AUROC CI, person-cluster | recall | specificity | ECE | Brier | "
          "TN FP FN TP | T | PCA dims | val AUROC |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    p = PAPER["RF"]
    L.append(f"| Stats+PCA+RF (paper) | {fmt_ci(p['AUROC'], p['AUROC_CI'])} | {fmt_ci(p['AP'], p['AP_CI'])} | - | "
             f"{p['recall']:.3f} | {p['specificity']:.3f} | {p['ECE']:.3f} | {p['Brier']:.3f} | (2 FP) | - | - | - |")
    for h, name in E.HEADS.items():
        m, c = metrics[h]["test"], metrics[h]["test"]["confusion"]
        L.append(f"| {name} | {fmt_ci(m['AUROC'], m['AUROC_CI'])} | {fmt_ci(m['AP'], m['AP_CI'])} | "
                 f"[{m['AUROC_CI_person_cluster'][0]:.3f}, {m['AUROC_CI_person_cluster'][1]:.3f}] | {m['recall']:.3f} | "
                 f"{m['specificity']:.3f} | {m['ECE']:.3f} | {m['Brier']:.3f} | {c['tn']} {c['fp']} {c['fn']} {c['tp']} | "
                 f"{metrics[h]['temperature']:.2f} | {metrics[h]['n_pca_components']} | {metrics[h]['validation']['AUROC']:.3f} |")
    sp = metrics["RF"]["rf_seed_spread"]
    L += ["", f"RF run-to-run spread (random_state 0-{N_SEED_SPREAD - 1}, same split/settings): "
          + ", ".join(f"{k} {sp[k]['mean']:.3f} +- {sp[k]['std']:.3f} [{sp[k]['min']:.3f}-{sp[k]['max']:.3f}]"
                      for k in ["AUROC", "AP", "F1", "Acc", "thr"]),
          f"Person-level test AUROC (mean probability per person): " + ", ".join(
              f"{h} {metrics[h]['test']['person_level_AUROC']:.3f}" for h in E.HEADS)
          + f" (n = {metrics['RF']['test']['person_level_n']})", "",
          "Per-channel ablation on TEST (RF; as in the paper's Fig. 7; test-set analysis, not for model selection):", "",
          abl_sum.round(3).to_markdown(), "", "Excluded clips (< min windows): "
          + ", ".join(f"{e['video_id']} ({e['label']}, {e['person_id']}, {e['orig_split']}, {e['n_windows']} win)"
                      for e in split_info["excluded (< min windows)"])]
    (od / "table1.md").write_text("\n".join(L))
    print("\n".join(L))
    figures(res, te, yt, val, abl, abl_sum, metrics, od)


def figures(res, te, yt, val, abl, abl_sum, metrics, od):
    tag = f"protocol A, TEST split (n={len(yt)}: {int(yt.sum())} dementia / {int((1 - yt).sum())} control clips)"
    # ROC
    fig, a = plt.subplots(figsize=(4.8, 4.4))
    for h, name in E.HEADS.items():
        f, t, _ = roc_curve(yt, res[h]["pt_c"])
        a.plot(f, t, color=COL[h], lw=2 if h == "RF" else 1.3, label=f"{name}  AUROC {metrics[h]['test']['AUROC']:.3f}")
    a.plot([0, 1], [0, 1], ":", color="grey")
    a.set(xlabel="false positive rate", ylabel="true positive rate", title=f"ROC (test)\n{tag}")
    a.title.set_fontsize(8)
    a.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(od / "fig3_roc_test.png", dpi=140); plt.close(fig)
    # PR
    fig, a = plt.subplots(figsize=(4.8, 4.4))
    for h, name in E.HEADS.items():
        pr, rc, _ = precision_recall_curve(yt, res[h]["pt_c"])
        a.step(rc, pr, where="post", color=COL[h], lw=2 if h == "RF" else 1.3, label=f"{name}  AP {metrics[h]['test']['AP']:.3f}")
    a.axhline(yt.mean(), ls=":", color="grey", label=f"prevalence {yt.mean():.2f}")
    a.set(xlabel="recall", ylabel="precision", ylim=(0, 1.03), title=f"Precision-recall (test)\n{tag}")
    a.title.set_fontsize(8)
    a.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(od / "fig4_pr_test.png", dpi=140); plt.close(fig)
    # confusion + histograms (RF)
    r, m = res["RF"], metrics["RF"]["test"]
    c = m["confusion"]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.8))
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    axs[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axs[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    axs[0].set_xticks([0, 1], ["pred control", "pred dementia"]); axs[0].set_yticks([0, 1], ["control", "dementia"])
    axs[0].set_title(f"Stats+PCA+RF, thr {r['thr']:.3f} (validation Youden)", fontsize=8)
    bins = np.linspace(0, 1, 21)
    for y, lab, col in [(0, "control", "#1f4e79"), (1, "dementia", "#c0392b")]:
        axs[1].hist(r["pt_c"][yt == y], bins=bins, alpha=0.6, color=col, label=lab)
    axs[1].axvline(r["thr"], color="black", ls="--", lw=1, label="threshold")
    axs[1].set(xlabel="calibrated P(dementia)", ylabel="clips")
    axs[1].legend(fontsize=7)
    fig.suptitle(f"Confusion matrix and posteriors, {tag}", fontsize=8)
    fig.tight_layout(); fig.savefig(od / "fig5_confusion_test.png", dpi=140); plt.close(fig)
    # threshold sweep (RF)
    ths = np.linspace(0, 1, 201)
    rows = [E.point_metrics(r["pt_c"], yt, t) for t in ths]
    fig, a = plt.subplots(figsize=(5.6, 4))
    for k, col in [("Acc", "#333333"), ("F1", "#1f4e79"), ("precision", "#7d8c2e"), ("recall", "#c0392b"),
                   ("specificity", "#8e44ad")]:
        a.plot(ths, [q[k] for q in rows], color=col, label=k, lw=1.5)
    a.axvline(r["thr"], color="black", ls="--", lw=1, label=f"val-chosen thr {r['thr']:.3f}")
    a.set(xlabel="threshold on calibrated P(dementia)", ylabel="metric on test", ylim=(-0.02, 1.05),
          title=f"Threshold sweep, Stats+PCA+RF\n{tag}")
    a.title.set_fontsize(8)
    a.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(od / "fig6_threshold_sweep_test.png", dpi=140); plt.close(fig)
    # ablation
    fig, a = plt.subplots(figsize=(6, 3.8))
    names = abl_sum.index.tolist()
    xs = np.arange(len(names))
    a.bar(xs, abl_sum.test_AUROC_seed42, color=["#555555"] + ["#1f4e79"] * (len(names) - 1), alpha=0.85,
          label="RF seed 42 (primary)")
    a.errorbar(xs, abl_sum.test_AUROC_mean10, yerr=abl_sum.test_AUROC_sd10, fmt="o", color="#c0392b", ms=4,
               capsize=3, label="mean +- SD over RF seeds 0-9")
    a.axhline(0.5, ls=":", color="grey")
    a.set_xticks(xs, ["full"] + [f"- {n}" for n in names[1:]], fontsize=8)
    a.set(ylabel="test AUROC", ylim=(0.3, 1.0), title=f"Per-channel ablation (one channel removed, composition rebuilt)\n{tag}")
    a.title.set_fontsize(8)
    a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(od / "fig7_ablation_test.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
