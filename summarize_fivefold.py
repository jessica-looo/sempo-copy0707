import sys
from pathlib import Path

import numpy as np

from utils.metrics import (
    MAE, MSE, pearson_corr, smooth_corr, a3_corr,
    trend_corr, direction_accuracy, lag_aware_corr,
    peak_event_metrics,
)


def summarize(root):
    root = Path(root)
    batches = []
    seen = set()

    for fold in range(5):
        with np.load(root / f"fold_{fold}" / "results.npz") as data:
            batch = {
                k: data[k]
                for k in ("preds", "trues", "sites", "years", "clusters")
            }

        sites = set(batch["sites"].tolist())
        expected = 6 if fold == 0 else 5
        if len(sites) != expected or seen & sites:
            raise ValueError(f"fold {fold} 站点数量错误或跨折重复")
        seen.update(sites)

        pred, true = batch["preds"], batch["trues"]
        if pred.shape != true.shape or pred.ndim != 3:
            raise ValueError(f"fold {fold} 预测维度错误")
        if any(len(v) != len(pred) for v in batch.values()):
            raise ValueError(f"fold {fold} 样本字段长度不一致")
        if not np.isfinite(pred).all() or not np.isfinite(true).all():
            raise ValueError(f"fold {fold} 含非有限预测或真值")

        batches.append(batch)

    if len(seen) != 26:
        raise ValueError("测试结果未覆盖 26 个站点")

    merged = {
        k: np.concatenate([b[k] for b in batches], axis=0)
        for k in batches[0]
    }
    pred, true = merged["preds"], merged["trues"]

    lag_corr, best_lag = lag_aware_corr(pred, true, max_lag=2)
    recall, precision, f1 = peak_event_metrics(
        pred, true, top_ratio=0.1, lag_window=2
    )
    scores = {
        "mse": MSE(pred, true),
        "mae": MAE(pred, true),
        "corr": pearson_corr(pred, true),
        "corr_smooth_3": smooth_corr(pred, true, window=3),
        "a3_corr": a3_corr(pred, true),
        "trend_corr": trend_corr(pred, true),
        "direction_acc": direction_accuracy(pred, true),
        "lag_corr": lag_corr,
        "best_lag": best_lag,
        "peak_recall_10_lag2": recall,
        "peak_precision_10_lag2": precision,
        "peak_f1_10_lag2": f1,
    }

    text = (
        f"mse: {scores['mse']:.4f}, mae: {scores['mae']:.4f}\n"
        f"corr: {scores['corr']:.4f}\n"
        f"corr_smooth_3: {scores['corr_smooth_3']:.4f}\n"
        f"a3_corr: {scores['a3_corr']:.4f}\n"
        f"trend_corr: {scores['trend_corr']:.4f}\n"
        f"direction_acc: {scores['direction_acc'] * 100:.1f}%\n"
        f"lag_corr@±2: {lag_corr:.4f}, best_lag: {best_lag}\n"
        f"PeakRecall@10%_±2: {recall:.4f}\n"
        f"PeakPrecision@10%_±2: {precision:.4f}\n"
        f"PeakF1@10%_±2: {f1:.4f}"
    )
    print(text)
    (root / "global_metrics.txt").write_text(text + "\n", encoding="utf-8")
    np.savez(root / "global_results.npz", **merged, **scores)


if __name__ == "__main__":
    summarize(sys.argv[1])