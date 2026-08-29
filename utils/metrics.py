import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def _as_3d(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)

    if pred.ndim == 2:
        pred = pred[..., None]
        true = true[..., None]

    return pred, true


def _nanmean_or_nan(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return np.nanmean(values)


def pearson_corr(pred, true, eps=1e-8):
    pred, true = _as_3d(pred, true)

    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    true_centered = true - true.mean(axis=1, keepdims=True)

    numerator = (pred_centered * true_centered).sum(axis=1)
    denominator = np.sqrt((pred_centered ** 2).sum(axis=1) * (true_centered ** 2).sum(axis=1))

    corr = numerator / (denominator + eps)
    corr = np.where(denominator > eps, corr, np.nan)
    return _nanmean_or_nan(corr)


def trend_corr(pred, true, eps=1e-8):
    pred, true = _as_3d(pred, true)

    if pred.shape[1] < 2:
        return np.nan

    pred_diff = np.diff(pred, axis=1)
    true_diff = np.diff(true, axis=1)
    return pearson_corr(pred_diff, true_diff, eps=eps)


def direction_accuracy(pred, true, eps=1e-8):
    pred, true = _as_3d(pred, true)

    if pred.shape[1] < 2:
        return np.nan

    pred_diff = np.diff(pred, axis=1)
    true_diff = np.diff(true, axis=1)

    pred_sign = np.sign(np.where(np.abs(pred_diff) < eps, 0.0, pred_diff))
    true_sign = np.sign(np.where(np.abs(true_diff) < eps, 0.0, true_diff))

    return np.mean(pred_sign == true_sign)


def lag_aware_corr(pred, true, max_lag=2, eps=1e-8):
    """
    Returns the best Pearson correlation after shifting pred against true.
    best_lag < 0 means pred leads true; best_lag > 0 means pred lags true.
    """
    pred, true = _as_3d(pred, true)

    best_corr = -np.inf
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            shifted_pred = pred[:, :lag, :]
            shifted_true = true[:, -lag:, :]
        elif lag > 0:
            shifted_pred = pred[:, lag:, :]
            shifted_true = true[:, :-lag, :]
        else:
            shifted_pred = pred
            shifted_true = true

        if shifted_pred.shape[1] < 2:
            continue

        corr = pearson_corr(shifted_pred, shifted_true, eps=eps)
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    if best_corr == -np.inf:
        return np.nan, 0

    return best_corr, best_lag


def _moving_average_same(x, window):
    x = np.asarray(x, dtype=np.float64)

    if window <= 1 or x.shape[1] < 2:
        return x

    window = min(int(window), x.shape[1])
    left = window // 2
    right = window - 1 - left

    x_pad = np.pad(x, ((0, 0), (left, right), (0, 0)), mode='edge')
    cumsum = np.cumsum(x_pad, axis=1)
    cumsum = np.concatenate([np.zeros_like(cumsum[:, :1, :]), cumsum], axis=1)
    return (cumsum[:, window:, :] - cumsum[:, :-window, :]) / window


def smooth_corr(pred, true, window=3, eps=1e-8):
    pred, true = _as_3d(pred, true)

    pred_smooth = _moving_average_same(pred, window)
    true_smooth = _moving_average_same(true, window)

    return pearson_corr(pred_smooth, true_smooth, eps=eps)


def _swt_haar_a3_np(x, levels=3):
    x = np.asarray(x, dtype=np.float64)
    approx = x

    for level in range(levels):
        shift = 2 ** level
        shifted = np.roll(approx, shift=-shift, axis=1)
        approx = 0.5 * (approx + shifted)

    return approx


def a3_corr(pred, true, eps=1e-8):
    pred, true = _as_3d(pred, true)

    if pred.shape[1] < 2:
        return np.nan

    pred_a3 = _swt_haar_a3_np(pred, levels=3)
    true_a3 = _swt_haar_a3_np(true, levels=3)

    return pearson_corr(pred_a3, true_a3, eps=eps)


def peak_event_metrics(pred, true, top_ratio=0.1, lag_window=2):
    """
    Event-level peak matching on first differences.

    Peaks are the top `top_ratio` absolute changes in each sample/channel.
    A true peak is matched if an unmatched predicted peak appears within
    +/- `lag_window` time steps.
    """
    pred, true = _as_3d(pred, true)

    if pred.shape[1] < 2:
        return np.nan, np.nan, np.nan

    pred_diff = np.diff(pred, axis=1)
    true_diff = np.diff(true, axis=1)

    total_true = 0
    total_pred = 0
    total_matched = 0

    batch_size, diff_len, n_vars = true_diff.shape
    k = max(1, int(np.ceil(diff_len * float(top_ratio))))

    for b in range(batch_size):
        for c in range(n_vars):
            true_mag = np.abs(true_diff[b, :, c])
            pred_mag = np.abs(pred_diff[b, :, c])

            if np.all(~np.isfinite(true_mag)) or np.all(~np.isfinite(pred_mag)):
                continue

            true_idx = np.argsort(true_mag)[-k:]
            pred_idx = np.argsort(pred_mag)[-k:]

            true_idx = true_idx[np.argsort(-true_mag[true_idx])]
            pred_set = set(int(i) for i in pred_idx)

            total_true += len(true_idx)
            total_pred += len(pred_idx)

            for ti in true_idx:
                candidates = [
                    pi for pi in pred_set
                    if abs(int(pi) - int(ti)) <= lag_window
                ]

                if not candidates:
                    continue

                best_pi = min(
                    candidates,
                    key=lambda pi: (abs(int(pi) - int(ti)), -pred_mag[int(pi)])
                )
                pred_set.remove(best_pi)
                total_matched += 1

    recall = total_matched / total_true if total_true > 0 else np.nan
    precision = total_matched / total_pred if total_pred > 0 else np.nan

    if np.isnan(recall) or np.isnan(precision) or (recall + precision) == 0:
        f1 = np.nan if np.isnan(recall) or np.isnan(precision) else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return recall, precision, f1


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
