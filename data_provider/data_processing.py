import os
import warnings
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import pywt
from scipy.stats import median_abs_deviation
from scipy.stats import spearmanr
from scipy.signal import detrend

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. Name cleaning / loading maps
# ------------------------------------------------------------

def clean_site_name(name):
    """
    Unified site-name cleaning.

    Use this for:
    - csv column names
    - cluster_result site names
    - data_sources.csv site names
    """
    return (
        str(name)
        .strip()
        .strip('"')
        .strip("'")
        .replace('  ', ' ')
    )


def clean_dataframe_columns(df):
    """
    Clean all column names in a dataframe.
    """
    return df.rename(columns={c: clean_site_name(c) for c in df.columns})


def load_cluster_map(cluster_result_path, cluster_col='Level_0'):
    """
    Load cluster result csv and return:
        {clean_site_name: cluster_id}
    """
    cluster_df = pd.read_csv(cluster_result_path)

    site_col = cluster_df.columns[0]
    cluster_df[site_col] = cluster_df[site_col].apply(clean_site_name)

    if cluster_col not in cluster_df.columns:
        raise ValueError(
            f"cluster_col={cluster_col} not found in {cluster_result_path}. "
            f"Available columns: {cluster_df.columns.tolist()}"
        )

    return cluster_df.set_index(site_col)[cluster_col].to_dict()


def load_site_meta(site_meta_path):
    """
    Load data_sources.csv and return:
        {clean_site_name: {'Latitude': ..., 'Longitude': ..., 'Elevation': ...}}
    """
    df_meta = pd.read_csv(site_meta_path)
    df_meta['site_name_clean'] = df_meta['site'].apply(clean_site_name)

    required_cols = ['Latitude', 'Longitude', 'Elevation']
    for col in required_cols:
        if col not in df_meta.columns:
            raise ValueError(f"{col} not found in site metadata: {site_meta_path}")

    return df_meta.set_index('site_name_clean')[required_cols].to_dict('index')


# ------------------------------------------------------------
# 2. Static feature
# ------------------------------------------------------------

def get_static_raw(site, site_meta_dict):
    """
    Return raw static feature:
        [Latitude, sin(Longitude), cos(Longitude), Elevation]

    If metadata is missing, use zeros.
    """
    clean_s = clean_site_name(site)
    meta = site_meta_dict.get(
        clean_s,
        {'Latitude': 0.0, 'Longitude': 0.0, 'Elevation': 0.0}
    )

    lat = float(meta.get('Latitude', 0.0))
    lon = float(meta.get('Longitude', 0.0))
    elev = float(meta.get('Elevation', 0.0))

    lon_rad = np.radians(lon)
    return np.array([lat, np.sin(lon_rad), np.cos(lon_rad), elev], dtype=np.float32)


def fit_static_scaler(train_sites, site_meta_dict):
    """
    Fit static scaler only on train_sites.
    """
    scaler = StandardScaler()

    static_list = []
    for s in train_sites:
        static_list.append(get_static_raw(s, site_meta_dict))

    if len(static_list) > 0:
        scaler.fit(np.asarray(static_list, dtype=np.float32))
    else:
        # fallback: fit on one zero vector to avoid NotFittedError
        scaler.fit(np.zeros((1, 4), dtype=np.float32))

    return scaler


def get_static_scaled(site, site_meta_dict, static_scaler):
    """
    Return scaled static feature with shape [4].
    """
    static_raw = get_static_raw(site, site_meta_dict).reshape(1, -1)
    return static_scaler.transform(static_raw)[0].astype(np.float32)


# ------------------------------------------------------------
# 3. PyWT denoising
# ------------------------------------------------------------

def swt_denoise_single_series(series, wavelet='db2', level=1):
    """
    Denoise one pandas Series by SWT.

    Only non-NaN values are denoised.
    Return clean values corresponding to valid non-NaN positions.
    """
    valid_data = series.dropna().values
    n = len(valid_data)

    if n == 0:
        return valid_data

    # SWT requires length divisible by 2 for level=1.
    pad_width = 1 if n % 2 != 0 else 0
    if pad_width > 0:
        x = np.pad(valid_data, (0, pad_width), mode='symmetric')
    else:
        x = valid_data.copy()

    if pywt.swt_max_level(len(x)) < level:
        return valid_data

    coeffs = pywt.swt(x, wavelet, level=level, start_level=0)
    cA, cD = coeffs[0]

    sigma = median_abs_deviation(cD, scale='normal')
    if sigma > 0:
        threshold = sigma * np.sqrt(2 * np.log(n))
        cD_clean = pywt.threshold(cD, value=threshold, mode='soft')
    else:
        cD_clean = cD

    clean_data = pywt.iswt([(cA, cD_clean)], wavelet)

    if pad_width > 0:
        clean_data = clean_data[:-pad_width]

    return clean_data


def apply_pywt_to_dataframe(df, sites, wavelet='db2', level=1, verbose=True):
    """
    Apply SWT denoising to selected site columns.
    """
    df_copy = df.copy()

    if verbose:
        print("=" * 60)
        print(f"Apply pywt denoising (wavelet={wavelet}, level={level})...")
        print("=" * 60)

    for site in sites:
        if site not in df_copy.columns:
            continue

        series = df_copy[site]
        valid_idx = series.dropna().index
        clean_vals = swt_denoise_single_series(series, wavelet=wavelet, level=level)

        if len(valid_idx) == len(clean_vals):
            df_copy.loc[valid_idx, site] = clean_vals

    if verbose:
        print(f"pywt done! Processed {len(sites)} sites.")
        print("=" * 60)

    return df_copy


# ------------------------------------------------------------
# 4. Site filtering / splitting / logging
# ------------------------------------------------------------

def get_common_sites(df_a, df_b=None, year_col='year'):
    """
    Get common site columns.

    If df_b is None:
        return all columns except year_col.
    Else:
        return intersection of site columns from df_a and df_b.
    """
    sites_a = set(df_a.columns) - {year_col}

    if df_b is None:
        return sorted(sites_a)

    sites_b = set(df_b.columns) - {year_col}
    return sorted(sites_a & sites_b)


def filter_valid_sites_single(df, sites, seq_len, pred_len, year_col='year'):
    """
    For single-variable dataset.
    A site is valid if its non-NaN length >= seq_len + pred_len.
    """
    valid_sites = []
    invalid_sites = []

    min_len = seq_len + pred_len

    for site in sites:
        length = len(df[[year_col, site]].dropna())
        if length >= min_len:
            valid_sites.append(site)
        else:
            invalid_sites.append(f"{site}({length}年)")

    return valid_sites, invalid_sites


def filter_valid_sites_pair(df_x, df_y, sites, seq_len, pred_len=0, year_col='year'):
    """
    For cross-variable or multi-source datasets.
    A site is valid if both x/y have enough non-NaN values.

    Use pred_len=0 for same-time cross-variable windows.
    Use pred_len>0 for forecasting-style windows.
    """
    valid_sites = []
    invalid_sites = []

    min_len = seq_len + pred_len

    for site in sites:
        len_x = len(df_x[[year_col, site]].dropna())
        len_y = len(df_y[[year_col, site]].dropna())

        if len_x >= min_len and len_y >= min_len:
            valid_sites.append(site)
        else:
            invalid_sites.append(f"{site}(x={len_x}, y={len_y})")

    return valid_sites, invalid_sites


def _site_columns_from_csv(path, year_col='year'):
    if path is None or not os.path.exists(path):
        return None

    df = pd.read_csv(path, nrows=1)
    df = clean_dataframe_columns(df)
    return set(df.columns) - {year_col}


def _first_existing_path(root_path, relative_candidates):
    search_roots = []
    cur = os.path.abspath(root_path)

    for _ in range(3):
        search_roots.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    for search_root in search_roots:
        for rel_path in relative_candidates:
            path = os.path.join(search_root, rel_path)
            if os.path.exists(path):
                return path

    return None


def build_preferred_test_site_ranks(root_path, year_col='year', verbose=False):
    """
    Build test-site priority ranks from cross-variable site overlap.

    Rank meaning:
        0: common to accum_100y, d18O and chem
        1: common to accum_100y + d18O, or chem + d18O

    The split function still keeps the per-cluster train/val/test counts.
    These ranks only decide which sites are consumed first by test slots.
    """
    accum_path = _first_existing_path(
        root_path,
        [
            'accum/accum_100y.csv',
            'accum_100y.csv',
            'accum/accum.csv',
            'accum.csv',
        ]
    )
    d18o_path = _first_existing_path(
        root_path,
        [
            'd18O/d18O_clean.csv',
            'd18O_clean.csv',
            'd180/d180_clean.csv',
            'd180_clean.csv',
            'd18O/merged_d18O.csv',
            'merged_d18O.csv',
        ]
    )
    chem_path = _first_existing_path(
        root_path,
        [
            'chem/chem_clean.csv',
            'chem_clean.csv',
            'chem/chem.csv',
            'chem.csv',
            'chem/merged_chem.csv',
            'merged_chem.csv',
        ]
    )

    accum_sites = _site_columns_from_csv(accum_path, year_col=year_col)
    d18o_sites = _site_columns_from_csv(d18o_path, year_col=year_col)
    chem_sites = _site_columns_from_csv(chem_path, year_col=year_col)

    if accum_sites is None or d18o_sites is None or chem_sites is None:
        if verbose:
            print(
                "Test priority overlap skipped: "
                f"accum_path={accum_path}, d18O_path={d18o_path}, chem_path={chem_path}"
            )
        return {}

    triple_common = accum_sites & d18o_sites & chem_sites
    d18o_pair_common = ((accum_sites & d18o_sites) | (chem_sites & d18o_sites)) - triple_common

    ranks = {}
    for site in triple_common:
        ranks[clean_site_name(site)] = 0
    for site in d18o_pair_common:
        ranks[clean_site_name(site)] = 1

    if verbose:
        print(
            "Test priority overlap: "
            f"rank0(accum+d18O+chem)={len(triple_common)}, "
            f"rank1(d18O pair)={len(d18o_pair_common)}"
        )

    return ranks


def split_sites_by_cluster(
    valid_sites,
    cluster_map,
    seed=2030,
    train_ratio=0.6,
    filter_cid=None,
    preferred_test_site_ranks=None,
    fold_id=None
):
    if fold_id is not None:
        parts = split_sites_fivefold(
            valid_sites, cluster_map, fold_id, seed
        )
        # 保留原含义：先划分，再按簇过滤。
        if filter_cid is not None:
            parts = tuple(
                [s for s in part
                 if cluster_map.get(clean_site_name(s), -1) == filter_cid]
                for part in parts
            )
        return parts

    # 下面保留原来的划分代码。
    rng = np.random.RandomState(seed)
    # preferred_test_site_ranks = preferred_test_site_ranks or {}
    preferred_test_site_ranks = {}

    cluster_dict = {}
    for s in valid_sites:
        cid = cluster_map.get(clean_site_name(s), -1)
        cluster_dict.setdefault(cid, []).append(s)

    train_sites, val_sites, test_sites = [], [], []

    for cid, c_sites in cluster_dict.items():
        c_sites = list(c_sites)
        rng.shuffle(c_sites)

        def select_test_sites(shuffled_sites, n_test):
            priority_sites = [
                s for s in shuffled_sites
                if preferred_test_site_ranks.get(clean_site_name(s), 2) < 2
            ]
            priority_sites = sorted(
                priority_sites,
                key=lambda s: preferred_test_site_ranks.get(clean_site_name(s), 2)
            )

            selected = priority_sites[:n_test]
            selected_set = set(selected)

            if len(selected) < n_test:
                neutral_sites = [s for s in shuffled_sites if s not in selected_set]
                selected.extend(neutral_sites[-(n_test - len(selected)):])

            return selected

        n = len(c_sites)

        if n == 1:
            train_sites.extend(c_sites)

        elif n == 2:
            selected_test = select_test_sites(c_sites, 1)
            selected_test_set = set(selected_test)
            selected_train = [s for s in c_sites if s not in selected_test_set]

            train_sites.extend(selected_train)
            test_sites.extend(selected_test)

        else:
            n_train = int(n * train_ratio)
            n_rem = n - n_train
            n_val = n_rem // 2
            n_test = n_rem - n_val

            selected_test = select_test_sites(c_sites, n_test)
            selected_test_set = set(selected_test)
            remaining_sites = [s for s in c_sites if s not in selected_test_set]

            train_sites.extend(remaining_sites[:n_train])
            val_sites.extend(remaining_sites[n_train:n_train + n_val])
            test_sites.extend(selected_test)

    if filter_cid is not None:
        train_sites = [
            s for s in train_sites
            if cluster_map.get(clean_site_name(s), -1) == filter_cid
        ]
        val_sites = [
            s for s in val_sites
            if cluster_map.get(clean_site_name(s), -1) == filter_cid
        ]
        test_sites = [
            s for s in test_sites
            if cluster_map.get(clean_site_name(s), -1) == filter_cid
        ]

    return train_sites, val_sites, test_sites


def select_current_sites(set_type, train_sites, val_sites, test_sites):
    """
    set_type:
        0 train
        1 val
        2 test
    """
    if set_type == 0:
        return train_sites
    elif set_type == 1:
        return val_sites
    else:
        return test_sites


def count_sites_by_cluster(sites, cluster_map):
    """
    Count number of sites per cluster.
    """
    counter = {}

    for s in sites:
        cid = cluster_map.get(clean_site_name(s), -1)
        counter[cid] = counter.get(cid, 0) + 1

    return counter


def log_site_split(
    train_sites,
    val_sites,
    test_sites,
    cluster_map,
    mode='cluster',
    invalid_sites=None,
):
    """
    Unified split logging.
    """
    if invalid_sites is None:
        invalid_sites = []

    print("=" * 60)
    print(f"数据划分模式: {mode}")

    if len(invalid_sites) > 0:
        print(f"剔除的站点: {invalid_sites}")

    print("-" * 60)
    print(f"Train 站点分配 ({len(train_sites)}个): {train_sites}")
    print(f"Val 站点分配 ({len(val_sites)}个): {val_sites}")
    print(f"Test 站点分配 ({len(test_sites)}个): {test_sites}")

    train_cnt = count_sites_by_cluster(train_sites, cluster_map)
    val_cnt = count_sites_by_cluster(val_sites, cluster_map)
    test_cnt = count_sites_by_cluster(test_sites, cluster_map)

    all_cids = sorted(set(train_cnt) | set(val_cnt) | set(test_cnt))

    print(f"{'簇ID':<12} {'Train':>6} {'Val':>6} {'Test':>6} {'合计':>6}")
    print("-" * 44)

    for cid in all_cids:
        t = train_cnt.get(cid, 0)
        v = val_cnt.get(cid, 0)
        te = test_cnt.get(cid, 0)
        label = f"Cluster {cid}" if cid != -1 else "未匹配"
        print(f"{label:<12} {t:>6} {v:>6} {te:>6} {t + v + te:>6}")

    print("-" * 44)
    print(
        f"{'合计':<12} {len(train_sites):>6} {len(val_sites):>6} "
        f"{len(test_sites):>6} {len(train_sites) + len(val_sites) + len(test_sites):>6}"
    )
    print("=" * 60)

def split_sites_fivefold(valid_sites, cluster_map, fold_id, seed=2030):
    import random

    quotas = {
        (6, 12, 8): [(2, 2, 2), (1, 3, 1), (1, 3, 1),
                     (1, 2, 2), (1, 2, 2)],
        (8, 6, 12): [(2, 2, 2), (1, 1, 3), (1, 1, 3),
                     (2, 1, 2), (2, 1, 2)],
    }

    if fold_id not in range(5):
        raise ValueError("fold_id 必须为 0～4")

    sites = list(valid_sites)
    if len(sites) != 26 or len(set(sites)) != 26:
        raise ValueError("五折模式要求 26 个不重复的有效站点")

    pools = [
        sorted(s for s in sites
               if cluster_map.get(clean_site_name(s), -1) == cid)
        for cid in range(3)
    ]
    counts = tuple(map(len, pools))
    if counts not in quotas:
        raise ValueError(f"不支持的簇人数：{counts}")

    groups = [[] for _ in range(5)]
    rng = random.Random(seed)
    for cid, pool in enumerate(pools):
        rng.shuffle(pool)
        start = 0
        for group, quota in zip(groups, quotas[counts]):
            end = start + quota[cid]
            group.extend(pool[start:end])
            start = end

    val_id = (fold_id + 1) % 5
    train = [
        site
        for i, group in enumerate(groups)
        if i not in (fold_id, val_id)
        for site in group
    ]
    return train, groups[val_id], groups[fold_id]

# ------------------------------------------------------------
# 5. Scaling
# ------------------------------------------------------------

def fit_global_scaler_from_sites(df, train_sites, year_col='year'):
    """
    Fit one StandardScaler using all train-site values stacked together.
    Suitable for single-variable accum dataset.
    """
    scaler = StandardScaler()

    train_data = []
    for site in train_sites:
        vals = df[[site]].dropna().values
        if len(vals) > 0:
            train_data.append(vals)

    if len(train_data) > 0:
        scaler.fit(np.vstack(train_data))
    else:
        scaler.fit(np.zeros((1, 1)))

    return scaler

def fit_crossvar_scalers_from_support(
    df_x,
    df_y,
    sites,
    seq_len,
    support_ratio,
    year_col='year',
):
    """
    只使用所有指定站点的 support 段拟合 x/y scaler。

    每个站点：
        1. 按共同年份对齐 x/y；
        2. 取前 support_ratio 比例；
        3. query 段不参与 scaler.fit。
    """
    x_support_values = []
    y_support_values = []

    support_len_map = {}

    for site in sites:
        _, x_vals, y_vals = align_two_site_series(
            df_x=df_x,
            df_y=df_y,
            site=site,
            year_col=year_col,
        )

        total_len = len(x_vals)

        if support_ratio is None:
            support_len = min(100, total_len)
        else:
            support_len = int(total_len * float(support_ratio))
            support_len = max(int(seq_len), support_len)
            support_len = min(support_len, total_len)

        support_len_map[site] = support_len

        if support_len > 0:
            x_support_values.append(x_vals[:support_len])
            y_support_values.append(y_vals[:support_len])

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    if x_support_values:
        x_scaler.fit(np.vstack(x_support_values))
    else:
        x_scaler.fit(np.zeros((1, 1), dtype=np.float32))

    if y_support_values:
        y_scaler.fit(np.vstack(y_support_values))
    else:
        y_scaler.fit(np.zeros((1, 1), dtype=np.float32))

    return x_scaler, y_scaler, support_len_map

def transform_site_series(df_site, site, scaler=None, scale=True):
    """
    Extract one site value column and optionally scale it.
    Return shape [T, 1].
    """
    vals = df_site[[site]].values.astype(np.float32)

    if scale and scaler is not None:
        vals = scaler.transform(vals)

    return vals.astype(np.float32)


def fit_columnwise_scaler(df, train_sites):
    """
    Fit StandardScaler on a matrix with columns = train sites.

    This is useful for Dataset_IceCore_MS where current code scales:
        df_accum_pywt[train_sites]
        df_chem_pywt[train_sites]
    """
    scaler = StandardScaler()

    if len(train_sites) == 0:
        scaler.fit(np.zeros((1, 1)))
        return scaler

    train_vals = df[train_sites].dropna().values

    if train_vals.size == 0:
        scaler.fit(np.zeros((1, len(train_sites))))
    else:
        scaler.fit(train_vals)

    return scaler


def transform_dataframe_columnwise(df, sites, train_sites, scaler):
    """
    Transform dataframe columns.

    For train_sites:
        use fitted columnwise scaler.
    For non-train sites:
        use average train mean/std as fallback.
    """
    df_scaled = df.copy()

    if len(train_sites) > 0:
        df_scaled[train_sites] = scaler.transform(df_scaled[train_sites])

    non_train_sites = [s for s in sites if s not in train_sites]

    fallback_mean = float(np.mean(scaler.mean_))
    fallback_std = float(np.mean(scaler.scale_)) + 1e-6

    for col in non_train_sites:
        df_scaled[col] = (df_scaled[col] - fallback_mean) / fallback_std

    return df_scaled


# ------------------------------------------------------------
# 6. Year stamp / alignment
# ------------------------------------------------------------

def make_year_stamp(years):
    """
    Build normalized year stamp.

    Return shape [T, 1].
    """
    years = np.asarray(years, dtype=np.float32)

    mean = years.mean()
    std = years.std()

    if std < 1e-6:
        std = 1.0

    year_norm = (years - mean) / std
    return year_norm.reshape(-1, 1).astype(np.float32)


def align_two_site_series(df_x, df_y, site, year_col='year'):
    """
    Align x/y by common years.

    Return:
        years, x_vals, y_vals
    where:
        years shape [T]
        x_vals shape [T, 1]
        y_vals shape [T, 1]
    """
    df_site_x = df_x[[year_col, site]].dropna()
    df_site_y = df_y[[year_col, site]].dropna()

    df_site_x = df_site_x.set_index(year_col)
    df_site_y = df_site_y.set_index(year_col)

    common_years = df_site_x.index.intersection(df_site_y.index)

    df_site_x = df_site_x.loc[common_years]
    df_site_y = df_site_y.loc[common_years]

    years = common_years.values.astype(np.float32)
    x_vals = df_site_x[[site]].values.astype(np.float32)
    y_vals = df_site_y[[site]].values.astype(np.float32)

    return years, x_vals, y_vals


def concat_multivariate_site(df_list, site, year_col='year'):
    """
    Build multivariate data for one site from several dataframes.

    Example:
        data_vals = concat_multivariate_site(
            [df_accum_pywt, df_chem_pywt],
            site
        )

    Return:
        years, data_vals

    data_vals shape:
        [T, num_variables]
    """
    indexed = []

    for df in df_list:
        tmp = df[[year_col, site]].dropna().set_index(year_col)
        indexed.append(tmp)

    common_years = indexed[0].index
    for tmp in indexed[1:]:
        common_years = common_years.intersection(tmp.index)

    vals = []
    for tmp in indexed:
        vals.append(tmp.loc[common_years, site].values.reshape(-1, 1))

    data_vals = np.concatenate(vals, axis=1).astype(np.float32)
    years = common_years.values.astype(np.float32)

    return years, data_vals


# ------------------------------------------------------------
# 7. Correlation / local statistics
# ------------------------------------------------------------

def compute_detrended_spearman(x, y, min_len=30):
    """
    Compute detrended Spearman correlation.
    Return 0.0 if data is too short or invalid.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    n = min(len(x), len(y))
    if n < min_len:
        return 0.0

    x = x[:n]
    y = y[:n]

    try:
        x_dt = detrend(x, type='linear')
        y_dt = detrend(y, type='linear')
        rho, _ = spearmanr(x_dt, y_dt)

        if np.isnan(rho):
            return 0.0

        return float(rho)

    except Exception:
        return 0.0


def compute_local_stats(series):
    """
    Compute:
        local_mean
        local_std
        low_freq_ratio

    Used in maml_static_feat.
    """
    x = np.asarray(series).flatten().astype(np.float32)

    if len(x) == 0:
        return 0.0, 1.0, 0.0

    local_mean = float(np.mean(x))
    local_std = float(np.std(x) + 1e-6)

    centered = x - local_mean
    fft_power = np.abs(np.fft.rfft(centered)) ** 2

    if len(fft_power) <= 1:
        local_f0 = 0.0
    else:
        split = max(1, len(fft_power) // 5)
        low_power = fft_power[:split].sum()
        high_power = fft_power[split:].sum()
        local_f0 = float(low_power / (high_power + 1e-8))

    return local_mean, local_std, local_f0


# ------------------------------------------------------------
# 8. Prototype features
# ------------------------------------------------------------

def compute_support_lengths(data_list, seq_len, pred_len=0, support_ratio=None, default_support_len=100):
    """
    Compute per-site support length.

    For forecasting:
        pred_len should be > 0.

    For same-time cross-variable:
        pred_len can be 0.
    """
    support_len_list = []
    min_required = seq_len + pred_len

    for data in data_list:
        total = len(data)

        if support_ratio is not None:
            sl = int(total * support_ratio)

            # support should be able to generate at least one window
            sl = max(sl, min_required)

            # query should also have some remaining length
            upper = total - min_required
            if upper > min_required:
                sl = min(sl, upper)

            sl = max(sl, min_required)

        else:
            sl = default_support_len

        # final protection
        sl = min(sl, total)
        support_len_list.append(sl)

    support_len = min(support_len_list) if len(support_len_list) > 0 else default_support_len

    return support_len_list, support_len


def build_cluster_prototypes(
    train_sites,
    df,
    cluster_map,
    support_len,
    target_cids=(0, 1, 2),
    year_col='year',
):
    """
    Build average prototype sequence for each cluster.

    df should contain:
        year column + site columns

    Return:
        {cid: np.array shape [support_len]}
    """
    prototypes = {cid: [] for cid in target_cids}

    for site in train_sites:
        cid = cluster_map.get(clean_site_name(site), -1)

        if cid not in prototypes:
            continue

        vals = df[[year_col, site]].dropna()[site].values.astype(np.float32)

        if len(vals) >= support_len:
            prototypes[cid].append(vals[:support_len])

    for cid in prototypes:
        if len(prototypes[cid]) > 0:
            prototypes[cid] = np.mean(prototypes[cid], axis=0)
        else:
            prototypes[cid] = np.zeros(support_len, dtype=np.float32)

    return prototypes


def compute_prototype_similarity(series, prototypes, support_len, target_cids=(0, 1, 2)):
    """
    Compute Pearson similarity between current site support series and each prototype.
    """
    x = np.asarray(series).flatten().astype(np.float32)

    if len(x) < support_len:
        return [0.0 for _ in target_cids]

    x = x[:support_len]

    sims = []

    for cid in target_cids:
        proto = np.asarray(prototypes.get(cid, np.zeros(support_len))).flatten()

        if len(proto) < support_len:
            sims.append(0.0)
            continue

        proto = proto[:support_len]

        if np.std(x) < 1e-6 or np.std(proto) < 1e-6:
            sims.append(0.0)
            continue

        sim = np.corrcoef(x, proto)[0, 1]
        sims.append(float(sim) if not np.isnan(sim) else 0.0)

    return sims


def load_encoder_proto_static(encoder_proto_path):
    """
    Load offline encoder-prototype cosine similarity.

    Expected npz fields:
        site_names: [N] clean site names
        site_sim:   [N, K] cosine similarity to train-set encoder prototypes

    Return:
        sim_map: {clean_site_name: np.array [K]}
        label_map: {clean_site_name: int prototype_cluster_label}
        proto_dim: K
        metadata: dict
    """
    if encoder_proto_path is None:
        return {}, {}, 0, {}

    if not os.path.exists(encoder_proto_path):
        raise FileNotFoundError(f"encoder_proto_path not found: {encoder_proto_path}")

    data = np.load(encoder_proto_path, allow_pickle=True)
    if 'site_names' not in data or 'site_sim' not in data:
        raise ValueError(
            f"{encoder_proto_path} must contain 'site_names' and 'site_sim'. "
            f"Available keys: {list(data.keys())}"
        )

    site_names = [clean_site_name(s) for s in data['site_names'].tolist()]
    site_sim = np.asarray(data['site_sim'], dtype=np.float32)

    if site_sim.ndim != 2:
        raise ValueError(f"site_sim must be 2D [N, K], got shape={site_sim.shape}")

    if len(site_names) != site_sim.shape[0]:
        raise ValueError(
            f"site_names length ({len(site_names)}) does not match site_sim rows ({site_sim.shape[0]})"
        )

    metadata = {}
    if 'metadata_json' in data:
        raw = data['metadata_json']
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        try:
            metadata = json.loads(str(raw))
        except json.JSONDecodeError:
            metadata = {'metadata_json': str(raw)}

    sim_map = {
        site: site_sim[i].astype(np.float32)
        for i, site in enumerate(site_names)
    }

    label_map = {}
    if 'site_cluster_label' in data:
        labels = np.asarray(data['site_cluster_label'], dtype=np.int64).reshape(-1)
        if len(labels) != len(site_names):
            raise ValueError(
                f"site_cluster_label length ({len(labels)}) does not match "
                f"site_names length ({len(site_names)})"
            )
        label_map = {
            site: int(labels[i])
            for i, site in enumerate(site_names)
        }

    return sim_map, label_map, int(site_sim.shape[1]), metadata


def append_encoder_proto_static(
    maml_static_feat,
    site_name,
    proto_sim_map,
    proto_dim,
    proto_label_map=None,
    include_proto_onehot=False,
):
    """
    Append offline encoder prototype cosine similarity and cluster one-hot.
    Missing sites receive zeros so val/test can still run.
    """
    base = np.asarray(maml_static_feat, dtype=np.float32).flatten()

    if proto_dim <= 0:
        return base

    clean_s = clean_site_name(site_name)
    sim = proto_sim_map.get(clean_s)
    if sim is None:
        sim = np.zeros(proto_dim, dtype=np.float32)
    else:
        sim = np.asarray(sim, dtype=np.float32).flatten()
        if sim.shape[0] != proto_dim:
            fixed = np.zeros(proto_dim, dtype=np.float32)
            take = min(proto_dim, sim.shape[0])
            fixed[:take] = sim[:take]
            sim = fixed

    extra = [base, sim]

    if include_proto_onehot:
        onehot = np.zeros(proto_dim, dtype=np.float32)
        if proto_label_map is None:
            label = None
        else:
            label = proto_label_map.get(clean_s)
        if label is not None and 0 <= int(label) < proto_dim:
            onehot[int(label)] = 1.0
        extra.append(onehot)

    return np.concatenate(extra, axis=0).astype(np.float32)


# ------------------------------------------------------------
# 9. Residual prediction
# ------------------------------------------------------------

def build_cluster_time_means(
    train_sites,
    df,
    cluster_map,
    target_cids=(0, 1, 2),
    year_col='year',
):
    """
    Build cluster mean time series from train_sites.

    Assumes all selected sites are aligned in the dataframe.
    """
    cluster_time_means = {}

    df_indexed = df.set_index(year_col)

    for cid in target_cids:
        c_sites = [
            s for s in train_sites
            if cluster_map.get(clean_site_name(s), -1) == cid
        ]

        c_sites = [s for s in c_sites if s in df_indexed.columns]

        if len(c_sites) == 0:
            continue

        stack_data = np.stack(
            [df_indexed[s].values.astype(np.float32) for s in c_sites],
            axis=0
        )

        cluster_time_means[cid] = stack_data.mean(axis=0)

    return cluster_time_means


def apply_residual(data_x, data_y, cluster_id, cluster_mean_x=None, cluster_mean_y=None):
    """
    Apply residual prediction.

    If cluster_mean_y is None, use cluster_mean_x for both x/y.
    """
    total_len = len(data_x)

    if cluster_mean_x is None:
        mean_x = np.zeros_like(data_x, dtype=np.float32)
    else:
        mean_x = np.asarray(cluster_mean_x, dtype=np.float32)
        if mean_x.ndim == 1:
            mean_x = mean_x.reshape(-1, 1)
        mean_x = mean_x[:total_len]

    if cluster_mean_y is None:
        mean_y = mean_x
    else:
        mean_y = np.asarray(cluster_mean_y, dtype=np.float32)
        if mean_y.ndim == 1:
            mean_y = mean_y.reshape(-1, 1)
        mean_y = mean_y[:total_len]

    data_x_model = data_x - mean_x
    data_y_model = data_y - mean_y

    return data_x_model, data_y_model, mean_x, mean_y


# ------------------------------------------------------------
# 10. Window builders
# ------------------------------------------------------------

def build_forecast_windows(
    data_x,
    data_y,
    data_stamp,
    years,
    seq_len,
    label_len,
    pred_len,
    support_len,
    stride=1,
):
    """
    Forecast-style slicing.

    x:
        [s_begin : s_begin + seq_len]

    y:
        [r_begin : r_begin + label_len + pred_len]

    query starts from:
        support_len - seq_len

    This matches Dataset_IceCore_Accum_Site / Dataset_IceCore_MS style.
    """
    total_len = len(data_x)
    window_size = seq_len + pred_len

    support_x, support_y = [], []
    support_x_mark, support_y_mark = [], []

    for i in range(0, support_len - window_size + 1, stride):
        s_begin = i
        s_end = s_begin + seq_len

        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len

        support_x.append(data_x[s_begin:s_end])
        support_y.append(data_y[r_begin:r_end])
        support_x_mark.append(data_stamp[s_begin:s_end])
        support_y_mark.append(data_stamp[r_begin:r_end])

    query_x, query_y = [], []
    query_x_mark, query_y_mark = [], []
    query_years = []

    q_start = support_len - seq_len

    for i in range(q_start, total_len - window_size + 1, stride):
        s_begin = i
        s_end = s_begin + seq_len

        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len

        query_x.append(data_x[s_begin:s_end])
        query_y.append(data_y[r_begin:r_end])
        query_x_mark.append(data_stamp[s_begin:s_end])
        query_y_mark.append(data_stamp[r_begin:r_end])

        if years is not None:
            query_years.append(years[r_begin + label_len:r_end])

    return {
        'support_x': support_x,
        'support_y': support_y,
        'support_x_mark': support_x_mark,
        'support_y_mark': support_y_mark,
        'query_x': query_x,
        'query_y': query_y,
        'query_x_mark': query_x_mark,
        'query_y_mark': query_y_mark,
        'query_years': query_years,
    }


def build_same_time_windows(
    data_x,
    data_y,
    data_stamp,
    years,
    seq_len,
    support_len,
    stride=1,
):
    """
    Same-time cross-variable slicing.

    x:
        [s_begin : s_begin + seq_len]

    y:
        [s_begin : s_begin + seq_len]

    query starts from:
        support_len

    This matches Dataset_IceCore_CrossVar style.
    """
    total_len = len(data_x)
    window_size = seq_len

    support_x, support_y = [], []
    support_x_mark, support_y_mark = [], []

    for i in range(0, support_len - window_size + 1, stride):
        s_begin = i
        s_end = s_begin + seq_len

        support_x.append(data_x[s_begin:s_end])
        support_y.append(data_y[s_begin:s_end])
        support_x_mark.append(data_stamp[s_begin:s_end])
        support_y_mark.append(data_stamp[s_begin:s_end])

    query_x, query_y = [], []
    query_x_mark, query_y_mark = [], []
    query_years = []

    q_start = support_len

    for i in range(q_start, total_len - window_size + 1, stride):
        s_begin = i
        s_end = s_begin + seq_len

        query_x.append(data_x[s_begin:s_end])
        query_y.append(data_y[s_begin:s_end])
        query_x_mark.append(data_stamp[s_begin:s_end])
        query_y_mark.append(data_stamp[s_begin:s_end])

        if years is not None:
            query_years.append(years[s_begin])

    return {
        'support_x': support_x,
        'support_y': support_y,
        'support_x_mark': support_x_mark,
        'support_y_mark': support_y_mark,
        'query_x': query_x,
        'query_y': query_y,
        'query_x_mark': query_x_mark,
        'query_y_mark': query_y_mark,
        'query_years': query_years,
    }


def build_mean_windows_forecast(
    mean_x,
    mean_y,
    seq_len,
    label_len,
    pred_len,
    support_len,
    total_len,
    stride=1,
):
    """
    Build residual mean windows for forecast-style dataset.
    Return:
        query_x_mean
        query_y_mean
    """
    window_size = seq_len + pred_len

    query_x_mean = []
    query_y_mean = []

    q_start = support_len - seq_len

    for i in range(q_start, total_len - window_size + 1, stride):
        s_begin = i
        s_end = s_begin + seq_len

        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len

        query_x_mean.append(mean_x[s_begin:s_end])
        query_y_mean.append(mean_y[r_begin + label_len:r_end])

    return query_x_mean, query_y_mean


def build_mean_windows_same_time(
    mean_x,
    mean_y,
    seq_len,
    support_len,
    total_len,
    stride=1,
):
    """
    Build residual mean windows for same-time cross-variable dataset.
    """
    query_x_mean = []
    query_y_mean = []

    q_start = support_len

    for i in range(q_start, total_len - seq_len + 1, stride):
        s_begin = i
        s_end = s_begin + seq_len

        query_x_mean.append(mean_x[s_begin:s_end])
        query_y_mean.append(mean_y[s_begin:s_end])

    return query_x_mean, query_y_mean


# ------------------------------------------------------------
# 11. MAML static feature builder
# ------------------------------------------------------------

def build_maml_static_feature(
    cluster_id,
    static_feat,
    local_series,
    prototypes=None,
    support_len=None,
    include_cluster_onehot=True,
    include_prototype=True,
    include_local_stats=True,
    target_cids=(0, 1, 2),
):
    """
    Build static feature used by MAML-style dataset.

    Components:
    - cluster one-hot, optional -- dim = 3
    - prototype similarity, optional -- dim = 3
    - static_feat, usually geo or geo + hist_corr -- dim = 4/5
    - local stats: mean/std/low_freq_ratio, optional -- dim = 3
    """
    features = []

    if include_cluster_onehot:
        one_hot = [1.0 if cluster_id == cid else 0.0 for cid in target_cids]
        features.extend(one_hot)

    if include_prototype and prototypes is not None and support_len is not None:
        sim_feats = compute_prototype_similarity(
            local_series,
            prototypes,
            support_len,
            target_cids=target_cids,
        )
        features.extend(sim_feats)

    features.extend(np.asarray(static_feat).flatten().tolist())

    if include_local_stats:
        local_mean, local_std, local_f0 = compute_local_stats(local_series)
        features.extend([local_mean, local_std, local_f0])

    return np.asarray(features, dtype=np.float32)


# ------------------------------------------------------------
# 12. Tensor conversion
# ------------------------------------------------------------

def to_float_tensor(x):
    """
    Convert list/array to torch.float32 tensor.
    """
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def pack_maml_return(
    windows,
    maml_static_feat,
    query_x_mean=None,
    query_y_mean=None,
):
    """
    Return the standard 12-item tuple currently used by your MAML datasets.

    Return order:
        support_x
        support_y
        support_x_mark
        support_y_mark
        query_x
        query_y
        query_x_mark
        query_y_mark
        maml_static_feat
        query_years
        query_x_mean
        query_y_mean
    """
    if query_x_mean is None:
        query_x_mean = []

    if query_y_mean is None:
        query_y_mean = []

    return (
        to_float_tensor(windows['support_x']),
        to_float_tensor(windows['support_y']),
        to_float_tensor(windows['support_x_mark']),
        to_float_tensor(windows['support_y_mark']),
        to_float_tensor(windows['query_x']),
        to_float_tensor(windows['query_y']),
        to_float_tensor(windows['query_x_mark']),
        to_float_tensor(windows['query_y_mark']),
        to_float_tensor(maml_static_feat),
        to_float_tensor(windows['query_years']),
        to_float_tensor(query_x_mean),
        to_float_tensor(query_y_mean),
    )
