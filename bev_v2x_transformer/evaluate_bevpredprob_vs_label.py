import json
import os

import numpy as np


PRED_ROOT = "BEVPredProb"
LABEL_ROOT = "BEVLabel_01"
OUT_JSON = "bevpredprob_vs_bevlabel_eval.json"

# pred T1/T2/T3 对齐 label channel 2/3/4
HORIZONS = [("T1", 2), ("T2", 3), ("T3", 4)]
THRESHOLDS = [0.3, 0.5]


def safe_div(a, b):
    return a / b if b else 0.0


def calc_conf(prob, gt, thr):
    pred_bin = prob >= thr
    tp = int(np.logical_and(pred_bin, gt).sum())
    fp = int(np.logical_and(pred_bin, ~gt).sum())
    fn = int(np.logical_and(~pred_bin, gt).sum())
    tn = int(np.logical_and(~pred_bin, ~gt).sum())
    return tp, fp, fn, tn


def summarize_conf(tp, fp, fn, tn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    iou = safe_div(tp, tp + fp + fn)
    acc = safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main():
    pred_dirs = sorted(
        [d for d in os.listdir(PRED_ROOT) if os.path.isdir(os.path.join(PRED_ROOT, d))]
    )
    label_files = sorted([f for f in os.listdir(LABEL_ROOT) if f.endswith(".npy")])
    label_map = {os.path.splitext(f)[0]: f for f in label_files}

    common_names = [d for d in pred_dirs if d in label_map]
    if not common_names:
        raise RuntimeError("No common sample names between BEVPredProb and BEVLabel_01")

    agg = {
        "global": {
            "pairs": 0,
            "pixels": 0,
            "bce_num": 0.0,
            "brier_num": 0.0,
            "conf": {str(t): [0, 0, 0, 0] for t in THRESHOLDS},
        },
        "per_horizon": {
            h: {
                "pairs": 0,
                "pixels": 0,
                "bce_num": 0.0,
                "brier_num": 0.0,
                "conf": {str(t): [0, 0, 0, 0] for t in THRESHOLDS},
            }
            for h, _ in HORIZONS
        },
    }

    all_probs = []
    all_gts = []

    for name in common_names:
        label = np.load(os.path.join(LABEL_ROOT, label_map[name]))
        if label.ndim != 3 or label.shape[0] < 5:
            continue

        preds = {}
        complete = True
        for h, _ in HORIZONS:
            pred_file = os.path.join(PRED_ROOT, name, f"{h}.npy")
            if not os.path.exists(pred_file):
                complete = False
                break
            preds[h] = np.load(pred_file)
        if not complete:
            continue

        for h, ch in HORIZONS:
            prob = preds[h].astype(np.float64)
            gt = (label[ch] > 0.5)
            if prob.shape != gt.shape:
                continue

            p = np.clip(prob, 1e-7, 1.0 - 1e-7)
            bce = -(gt * np.log(p) + (~gt) * np.log(1.0 - p)).mean()
            brier = ((prob - gt.astype(np.float64)) ** 2).mean()
            npx = int(gt.size)

            g = agg["global"]
            g["pairs"] += 1
            g["pixels"] += npx
            g["bce_num"] += float(bce) * npx
            g["brier_num"] += float(brier) * npx

            ph = agg["per_horizon"][h]
            ph["pairs"] += 1
            ph["pixels"] += npx
            ph["bce_num"] += float(bce) * npx
            ph["brier_num"] += float(brier) * npx

            for t in THRESHOLDS:
                tp, fp, fn, tn = calc_conf(prob, gt, t)

                conf_g = g["conf"][str(t)]
                conf_g[0] += tp
                conf_g[1] += fp
                conf_g[2] += fn
                conf_g[3] += tn

                conf_h = ph["conf"][str(t)]
                conf_h[0] += tp
                conf_h[1] += fp
                conf_h[2] += fn
                conf_h[3] += tn

            all_probs.append(prob.reshape(-1))
            all_gts.append(gt.reshape(-1))

    result = {
        "dataset": {
            "n_pred_dirs": len(pred_dirs),
            "n_label_files": len(label_files),
            "n_common_names": len(common_names),
        },
        "global": {
            "n_samples_horizon_pairs": agg["global"]["pairs"],
            "n_pixels": agg["global"]["pixels"],
            "bce": safe_div(agg["global"]["bce_num"], agg["global"]["pixels"]),
            "brier": safe_div(agg["global"]["brier_num"], agg["global"]["pixels"]),
            "metrics@0.3": summarize_conf(*agg["global"]["conf"]["0.3"]),
            "metrics@0.5": summarize_conf(*agg["global"]["conf"]["0.5"]),
        },
        "per_horizon": {},
    }

    for h, _ in HORIZONS:
        item = agg["per_horizon"][h]
        result["per_horizon"][h] = {
            "n_samples": item["pairs"],
            "n_pixels": item["pixels"],
            "bce": safe_div(item["bce_num"], item["pixels"]),
            "brier": safe_div(item["brier_num"], item["pixels"]),
            "metrics@0.3": summarize_conf(*item["conf"]["0.3"]),
            "metrics@0.5": summarize_conf(*item["conf"]["0.5"]),
        }

    if all_probs:
        probs = np.concatenate(all_probs)
        gts = np.concatenate(all_gts)
        best_t = None
        best_f1 = -1.0
        for t in np.arange(0.10, 0.91, 0.05):
            tp, fp, fn, _ = calc_conf(probs, gts, t)
            p = safe_div(tp, tp + fp)
            r = safe_div(tp, tp + fn)
            f1 = safe_div(2 * p * r, p + r)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_t = float(round(float(t), 2))
        result["global"]["best_f1_threshold_scan_0.10_0.90_step0.05"] = {
            "threshold": best_t,
            "f1": best_f1,
        }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"saved: {OUT_JSON}")
    print(f"common samples: {result['dataset']['n_common_names']}")
    print(f"global BCE: {result['global']['bce']:.6f}")
    print(f"global Brier: {result['global']['brier']:.6f}")

    m03 = result["global"]["metrics@0.3"]
    m05 = result["global"]["metrics@0.5"]
    print(
        "global@0.3 F1/IoU/P/R:",
        f"{m03['f1']:.6f}",
        f"{m03['iou']:.6f}",
        f"{m03['precision']:.6f}",
        f"{m03['recall']:.6f}",
    )
    print(
        "global@0.5 F1/IoU/P/R:",
        f"{m05['f1']:.6f}",
        f"{m05['iou']:.6f}",
        f"{m05['precision']:.6f}",
        f"{m05['recall']:.6f}",
    )

    best = result["global"].get("best_f1_threshold_scan_0.10_0.90_step0.05")
    if best:
        print(
            "best F1 threshold scan:",
            f"threshold={best['threshold']}, f1={best['f1']:.6f}",
        )

    print(
        "per-horizon F1@0.3:",
        {
            k: round(result["per_horizon"][k]["metrics@0.3"]["f1"], 6)
            for k in ["T1", "T2", "T3"]
        },
    )


if __name__ == "__main__":
    main()
