import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_provider.data_processing import (
    align_two_site_series,
    apply_pywt_to_dataframe,
    clean_dataframe_columns,
    clean_site_name,
    compute_detrended_spearman,
    filter_valid_sites_pair,
    fit_global_scaler_from_sites,
    fit_static_scaler,
    get_common_sites,
    get_static_scaled,
    load_site_meta,
    make_year_stamp,
)


class Dataset_IceCore_RelationPretrain(Dataset):
    """
    CrossVar relation pretraining dataset.

    One sample is two history windows from the same site:
        x1/y1 and x2/y2.

    It intentionally does not use cid or cluster split. The only static feature
    is scaled lat/lon representation from the existing geo scaler.
    """

    def __init__(
        self,
        root_path,
        x_data_path,
        y_data_path,
        seq_len,
        samples_per_site=8,
        support_ratio=None,
        apply_pywt=False,
        site_meta_path='dataset/icecore/data_sources.csv',
        scale=True,
        verbose=True,
    ):
        self.root_path = root_path
        self.x_data_path = x_data_path
        self.y_data_path = y_data_path
        self.seq_len = int(seq_len)
        self.samples_per_site = int(samples_per_site)
        self.support_ratio = support_ratio
        self.apply_pywt = apply_pywt
        self.site_meta_path = site_meta_path
        self.scale = scale
        self.verbose = verbose

        self.__read_data__()

    def __read_data__(self):
        x_file = os.path.join(self.root_path, self.x_data_path)
        y_file = os.path.join(self.root_path, self.y_data_path)

        df_x_raw = clean_dataframe_columns(pd.read_csv(x_file))
        df_y_raw = clean_dataframe_columns(pd.read_csv(y_file))
        df_x_raw = df_x_raw.sort_values(by='year', ascending=True).reset_index(drop=True)
        df_y_raw = df_y_raw.sort_values(by='year', ascending=True).reset_index(drop=True)

        common_sites = get_common_sites(df_x_raw, df_y_raw, year_col='year')

        if self.apply_pywt:
            df_x_data = apply_pywt_to_dataframe(
                df_x_raw,
                sites=common_sites,
                wavelet='db2',
                level=1,
                verbose=self.verbose,
            )
            df_y_data = apply_pywt_to_dataframe(
                df_y_raw,
                sites=common_sites,
                wavelet='db2',
                level=1,
                verbose=False,
            )
        else:
            df_x_data = df_x_raw.copy()
            df_y_data = df_y_raw.copy()

        valid_sites, invalid_sites = filter_valid_sites_pair(
            df_x=df_x_data,
            df_y=df_y_data,
            sites=common_sites,
            seq_len=self.seq_len,
            pred_len=0,
            year_col='year',
        )

        self.site_meta_dict = load_site_meta(self.site_meta_path)
        self.x_scaler = fit_global_scaler_from_sites(df_x_data, valid_sites, year_col='year') if self.scale else None
        self.y_scaler = fit_global_scaler_from_sites(df_y_data, valid_sites, year_col='year') if self.scale else None
        self.static_scaler = fit_static_scaler(valid_sites, self.site_meta_dict)

        df_x_scaled = df_x_data.copy()
        df_y_scaled = df_y_data.copy()
        if self.scale:
            for site in valid_sites:
                if site in df_x_scaled.columns:
                    mask_x = df_x_scaled[site].notna()
                    vals_x = df_x_scaled.loc[mask_x, [site]].values.astype(np.float32)
                    if len(vals_x) > 0:
                        df_x_scaled.loc[mask_x, site] = self.x_scaler.transform(vals_x).flatten()
                if site in df_y_scaled.columns:
                    mask_y = df_y_scaled[site].notna()
                    vals_y = df_y_scaled.loc[mask_y, [site]].values.astype(np.float32)
                    if len(vals_y) > 0:
                        df_y_scaled.loc[mask_y, site] = self.y_scaler.transform(vals_y).flatten()

        self.site_names = []
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []
        self.static_feat_list = []

        for site in valid_sites:
            years, x_vals, y_vals = align_two_site_series(df_x_scaled, df_y_scaled, site, year_col='year')
            _, x_raw, y_raw = align_two_site_series(df_x_data, df_y_data, site, year_col='year')

            history_len = self._history_len(len(x_vals))
            if history_len < self.seq_len:
                continue

            static_scaled = get_static_scaled(site, self.site_meta_dict, self.static_scaler)
            # Geo only: scaled latitude plus sin/cos longitude.
            # This keeps the first pretrain round independent from cid-derived fields.
            static_feat = static_scaled[:3].astype(np.float32)
            _ = compute_detrended_spearman(x_raw, y_raw, min_len=30)

            self.site_names.append(clean_site_name(site))
            self.data_x_list.append(x_vals[:history_len].astype(np.float32))
            self.data_y_list.append(y_vals[:history_len].astype(np.float32))
            self.data_stamp_list.append(make_year_stamp(years[:history_len]).astype(np.float32))
            self.static_feat_list.append(static_feat)

        self.num_sites = len(self.site_names)
        if self.verbose:
            print(
                f"Relation pretrain sites: {self.num_sites}; "
                f"invalid sites before history cutoff: {len(invalid_sites)}"
            )

    def _history_len(self, total_len):
        if self.support_ratio is None:
            return total_len
        history_len = int(total_len * float(self.support_ratio))
        history_len = max(history_len, self.seq_len)
        return min(history_len, total_len)

    def _sample_window(self, site_id, rng):
        x = self.data_x_list[site_id]
        y = self.data_y_list[site_id]
        stamp = self.data_stamp_list[site_id]
        max_start = len(x) - self.seq_len
        start = int(rng.randint(0, max_start + 1)) if max_start > 0 else 0
        end = start + self.seq_len
        return x[start:end], y[start:end], stamp[start:end]

    def __getitem__(self, index):
        site_id = index % self.num_sites
        rng = np.random.RandomState((index + 1) * 1009 + np.random.randint(0, 1000000))

        x1, y1, m1 = self._sample_window(site_id, rng)
        x2, y2, m2 = self._sample_window(site_id, rng)

        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(y1, dtype=torch.float32),
            torch.tensor(m1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(y2, dtype=torch.float32),
            torch.tensor(m2, dtype=torch.float32),
            torch.tensor(self.static_feat_list[site_id], dtype=torch.float32),
            torch.tensor(site_id, dtype=torch.long),
        )

    def __len__(self):
        return self.num_sites * max(1, self.samples_per_site)
