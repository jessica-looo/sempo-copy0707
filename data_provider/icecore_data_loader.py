import os
import warnings
from collections import defaultdict
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
from data_provider.data_processing import *
from data_provider.icecore_relation_pretrain_loader import Dataset_IceCore_RelationPretrain

warnings.filterwarnings('ignore')

class Dataset_IceCore_S(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='antarctic_aligned.csv',
                 target='snow', scale=True, timeenc=0, freq='y',
                 percent=100, task_name='long_term_forecast', is_pretraining=0,
                 apply_pywt=False,
                 site_meta_path='dataset/icecore/data_sources.csv',
                 use_spectral_features=1,
                 cluster_result_path='cluster_results_cosine_zscore.csv',
                 residual_prediction=False,
                 pred_seperate=None,
                 support_ratio=None,
                 filter_cid=None,
                 use_encoder_proto_static=False,
                 encoder_proto_path=None,
                 use_old_cluster_onehot=False,
                 use_pretrain_cluster_onehot=False,
                 fold_id=None,
                 verbose=True):

        # =========================
        # Basic config
        # =========================
        if size is None:
            self.seq_len = 64
            self.label_len = 32
            self.pred_len = 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.site_meta_path = site_meta_path
        self.cluster_result_path = cluster_result_path
        self.fold_id = fold_id

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.task_name = task_name
        self.is_pretraining = is_pretraining

        self.apply_pywt = apply_pywt
        self.residual_prediction = residual_prediction
        self.support_ratio = support_ratio
        self.filter_cid = filter_cid
        self.verbose = verbose
        self.use_encoder_proto_static = use_encoder_proto_static
        self.encoder_proto_path = encoder_proto_path

        # 保留参数，但先不使用，避免破坏外部 argparse / data_factory
        self.use_spectral_features = use_spectral_features
        self.pred_seperate = pred_seperate
        self.pred_seperate_list = None
        self.encoder_proto_sim_map = {}
        self.encoder_proto_label_map = {}
        self.encoder_proto_dim = 0
        self.encoder_proto_metadata = {}
        if self.use_encoder_proto_static:
            self.encoder_proto_sim_map, self.encoder_proto_label_map, self.encoder_proto_dim, self.encoder_proto_metadata = load_encoder_proto_static(
                self.encoder_proto_path
            )

        # 如果后面还想启用 pred_seperate，可以再打开
        # if pred_seperate is not None:
        #     if isinstance(pred_seperate, str):
        #         self.pred_seperate_list = [int(x) for x in pred_seperate.strip().split()]
        #     elif isinstance(pred_seperate, list):
        #         self.pred_seperate_list = [int(x) for x in pred_seperate]
        #     else:
        #         self.pred_seperate_list = [int(pred_seperate)]

        self.__read_data__()

    def __read_data__(self):
        # =========================
        # 1. Load files
        # =========================
        data_file = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(data_file)

        df_raw = clean_dataframe_columns(df_raw)
        df_raw = df_raw.sort_values(by='year', ascending=True).reset_index(drop=True)

        self.cluster_map = load_cluster_map(self.cluster_result_path)
        self.site_meta_dict = load_site_meta(self.site_meta_path)

        all_sites = [col for col in df_raw.columns if col != 'year']

        # =========================
        # 2. Optional PyWT denoising
        # =========================
        if self.apply_pywt:
            df_data = apply_pywt_to_dataframe(
                df_raw,
                sites=all_sites,
                wavelet='db2',
                level=1,
                verbose=(self.verbose and self.set_type == 0)
            )
        else:
            df_data = df_raw.copy()

        self.df_data = df_data

        # =========================
        # 3. Filter valid sites
        # =========================
        valid_sites, invalid_sites = filter_valid_sites_single(
            df_data,
            sites=all_sites,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            year_col='year'
        )

        # =========================
        # 4. Split train / val / test
        # =========================
        preferred_test_site_ranks = build_preferred_test_site_ranks(
            root_path=self.root_path,
            year_col='year',
            verbose=(self.verbose and self.set_type == 0)
        )

        train_sites, val_sites, test_sites = split_sites_by_cluster(
            valid_sites=valid_sites,
            cluster_map=self.cluster_map,
            seed=2030,
            train_ratio=0.6,
            filter_cid=self.filter_cid,
            fold_id=self.fold_id,
            preferred_test_site_ranks=preferred_test_site_ranks
        )

        self.train_sites = train_sites
        self.val_sites = val_sites
        self.test_sites = test_sites

        if self.verbose and self.set_type == 0:
            log_site_split(
                train_sites=train_sites,
                val_sites=val_sites,
                test_sites=test_sites,
                cluster_map=self.cluster_map,
                invalid_sites=invalid_sites
            )

        self.current_sites = select_current_sites(
            self.set_type,
            train_sites,
            val_sites,
            test_sites
        )

        # =========================
        # 5. Fit scalers using train sites only
        # =========================
        self.scaler = None
        if self.scale:
            self.scaler = fit_global_scaler_from_sites(
                df_data,
                train_sites=train_sites,
                year_col='year'
            )

        self.static_scaler = fit_static_scaler(
            train_sites=train_sites,
            site_meta_dict=self.site_meta_dict
        )

        # =========================
        # 6. Build per-site data lists
        # =========================
        self.static_feat_list = []
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []
        self.year_list = []
        self.valid_indices = []
        self.site_cluster_map = {}

        for site_idx, site in enumerate(self.current_sites):
            clean_s = clean_site_name(site)

            self.site_cluster_map[site_idx] = self.cluster_map.get(clean_s, -1)

            # static feature: [lat, sin_lon, cos_lon, elevation], scaled
            static_scaled = get_static_scaled(
                site,
                self.site_meta_dict,
                self.static_scaler
            )
            self.static_feat_list.append(static_scaled)

            # time-series values
            df_site = df_data[['year', site]].dropna().copy()
            years = df_site['year'].values.astype(np.float32)

            vals = df_site[[site]].values.astype(np.float32)
            if self.scale and self.scaler is not None:
                vals = self.scaler.transform(vals).astype(np.float32)

            data_stamp = make_year_stamp(years)

            self.data_x_list.append(vals)
            self.data_y_list.append(vals)
            self.data_stamp_list.append(data_stamp)
            self.year_list.append(years)

            seq_count = len(vals) - self.seq_len - self.pred_len + 1
            for i in range(max(0, seq_count)):
                self.valid_indices.append((site_idx, i))

        # =========================
        # 7. Few-shot truncation
        # =========================
        if self.set_type == 0 and self.percent < 100:
            keep_len = int(len(self.valid_indices) * (self.percent / 100))
            self.valid_indices = self.valid_indices[:keep_len]

        # =========================
        # 8. MAML support length
        # =========================
        self.support_len_list, self.support_len = compute_support_lengths(
            data_list=self.data_x_list,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            support_ratio=self.support_ratio,
            default_support_len=100
        )

        # =========================
        # 9. Cluster prototypes
        # =========================
        self.prototypes = build_cluster_prototypes(
            train_sites=train_sites,
            df=df_data,
            cluster_map=self.cluster_map,
            support_len=self.support_len,
            target_cids=(0, 1, 2),
            year_col='year'
        )

        # =========================
        # 10. Residual prediction cluster means
        # =========================
        self.cluster_time_means = {}

        if self.residual_prediction:
            self.cluster_time_means = build_cluster_time_means(
                train_sites=train_sites,
                df=df_data,
                cluster_map=self.cluster_map,
                target_cids=(0, 1, 2),
                year_col='year'
            )

            # 如果使用 scaler，则也把 cluster mean transform 到同一尺度
            if self.scale and self.scaler is not None:
                for cid, mean_arr in self.cluster_time_means.items():
                    mean_arr = np.asarray(mean_arr).reshape(-1, 1)
                    self.cluster_time_means[cid] = (
                        self.scaler.transform(mean_arr)
                        .flatten()
                        .astype(np.float32)
                    )

    def __getitem__(self, index):
        # 注意：这个 Dataset 是 site-level MAML，
        # 所以 index 对应一个 site，而不是一个普通 sliding window。
        site_idx = index

        site_name = self.current_sites[site_idx]
        cluster_id = self.site_cluster_map.get(site_idx, -1)

        data_x = self.data_x_list[site_idx]
        data_y = self.data_y_list[site_idx]
        data_stamp = self.data_stamp_list[site_idx]
        years = self.year_list[site_idx]

        total_len = len(data_x)
        site_support_len = self.support_len_list[site_idx]

        # =========================
        # 1. Optional residual prediction
        # =========================
        if self.residual_prediction:
            c_mean = self.cluster_time_means.get(
                cluster_id,
                np.zeros(total_len, dtype=np.float32)
            )

            data_x_model, data_y_model, mean_x, mean_y = apply_residual(
                data_x=data_x,
                data_y=data_y,
                cluster_id=cluster_id,
                cluster_mean_x=c_mean,
                cluster_mean_y=c_mean
            )
        else:
            data_x_model = data_x
            data_y_model = data_y
            mean_x = np.zeros_like(data_x, dtype=np.float32)
            mean_y = np.zeros_like(data_y, dtype=np.float32)

        # =========================
        # 2. Build support/query windows
        # =========================
        windows = build_forecast_windows(
            data_x=data_x_model,
            data_y=data_y_model,
            data_stamp=data_stamp,
            years=years,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            support_len=site_support_len,
            stride=1
        )

        query_x_mean, query_y_mean = build_mean_windows_forecast(
            mean_x=mean_x,
            mean_y=mean_y,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            support_len=site_support_len,
            total_len=total_len,
            stride=1
        )

        # =========================
        # 3. Build MAML static feature
        # =========================
        static_feat = self.static_feat_list[site_idx]

        # 用原始 scaled data_x 的 support 段算 local stats / prototype similarity
        local_series = data_x[:site_support_len].flatten()

        include_cluster_onehot = self.filter_cid is None
        include_prototype = self.filter_cid is None

        maml_static_feat = build_maml_static_feature(
            cluster_id=cluster_id,
            static_feat=static_feat,
            local_series=local_series,
            prototypes=self.prototypes,
            support_len=self.support_len,
            include_cluster_onehot=include_cluster_onehot,
            include_prototype=include_prototype,
            include_local_stats=True,
            target_cids=(0, 1, 2)
        )

        # =========================
        # 4. Return standard MAML tuple
        # =========================
        return pack_maml_return(
            windows=windows,
            maml_static_feat=maml_static_feat,
            query_x_mean=query_x_mean,
            query_y_mean=query_y_mean
        )

    def __len__(self):
        # site-level MAML dataset:
        # one item = one site/task
        return len(self.current_sites)

    def inverse_transform(self, data, site_name=None):
        """
        Backward compatible inverse_transform.

        原来这里写过:
            if self.scale and site_name in self.scalers:
                return self.scalers[site_name].inverse_transform(data)

        但当前类实际没有稳定维护 self.scalers。
        这里统一使用 global scaler。
        """
        if self.scale and self.scaler is not None:
            return self.scaler.inverse_transform(data)

        return data

class Dataset_IceCore_CrossVar(Dataset):
    """
    Cross-variable site-level MAML Dataset.

    用途:
    - x_data_path: 源变量，例如 accum.csv
    - y_data_path: 目标变量，例如 merged_chem_clean.csv
    - support_x/query_x 来自 x
    - support_y/query_y 来自 y
    - x/y 按 year 严格对齐
    - support/query 使用 same-time windows:
        x[t : t + seq_len] -> y[t : t + seq_len]
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S',
                 data_path=None,
                 x_data_path='accum.csv',
                 y_data_path='merged_chem_clean.csv',
                 target='na', scale=True, timeenc=0, freq='y',
                 percent=100, task_name='long_term_forecast', is_pretraining=0,
                 apply_pywt=False,
                 site_meta_path='dataset/icecore/data_sources.csv',
                 use_spectral_features=1,
                 cluster_result_path='cluster_results_cosine_zscore.csv',
                 residual_prediction=False,
                 target_anchor_residual=False,
                 target_anchor_len=16,
                 pred_seperate=None,
                 support_ratio=None,
                 filter_cid=None,
                 use_encoder_proto_static=False,
                 encoder_proto_path=None,
                 use_old_cluster_onehot=False,
                 use_pretrain_cluster_onehot=False,
                 fold_id=None,
                 verbose=True):

        # =========================
        # Basic config
        # =========================
        if size is None:
            self.seq_len = 64
            self.label_len = 32
            self.pred_len = 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.x_data_path = x_data_path
        self.y_data_path = y_data_path
        self.site_meta_path = site_meta_path
        self.cluster_result_path = cluster_result_path

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.task_name = task_name
        self.is_pretraining = is_pretraining

        self.apply_pywt = apply_pywt
        self.residual_prediction = residual_prediction
        self.target_anchor_residual = target_anchor_residual
        self.target_anchor_len = int(target_anchor_len)
        self.support_ratio = support_ratio
        self.filter_cid = filter_cid
        self.verbose = verbose
        self.use_encoder_proto_static = use_encoder_proto_static
        self.encoder_proto_path = encoder_proto_path
        self.use_old_cluster_onehot = use_old_cluster_onehot
        self.use_pretrain_cluster_onehot = use_pretrain_cluster_onehot
        self.fold_id = fold_id

        # 保留参数，避免 data_factory / argparse 调用时报 unexpected keyword
        self.use_spectral_features = use_spectral_features
        self.pred_seperate = pred_seperate
        self.pred_seperate_list = None
        self.encoder_proto_sim_map = {}
        self.encoder_proto_label_map = {}
        self.encoder_proto_dim = 0
        self.encoder_proto_metadata = {}
        if self.use_encoder_proto_static:
            self.encoder_proto_sim_map, self.encoder_proto_label_map, self.encoder_proto_dim, self.encoder_proto_metadata = load_encoder_proto_static(
                self.encoder_proto_path
            )

        self.__read_data__()

    def __read_data__(self):
        # =========================
        # 1. Load x/y files
        # =========================
        x_file = os.path.join(self.root_path, self.x_data_path)
        y_file = os.path.join(self.root_path, self.y_data_path)

        df_x_raw = pd.read_csv(x_file)
        df_y_raw = pd.read_csv(y_file)

        df_x_raw = clean_dataframe_columns(df_x_raw)
        df_y_raw = clean_dataframe_columns(df_y_raw)

        df_x_raw = df_x_raw.sort_values(by='year', ascending=True).reset_index(drop=True)
        df_y_raw = df_y_raw.sort_values(by='year', ascending=True).reset_index(drop=True)

        self.cluster_map = load_cluster_map(self.cluster_result_path)
        self.site_meta_dict = load_site_meta(self.site_meta_path)

        common_sites = get_common_sites(df_x_raw, df_y_raw, year_col='year')

        # =========================
        # 2. Optional PyWT denoising
        # =========================
        if self.apply_pywt:
            df_x_data = apply_pywt_to_dataframe(
                df_x_raw,
                sites=common_sites,
                wavelet='db2',
                level=1,
                verbose=(self.verbose and self.set_type == 0)
            )

            df_y_data = apply_pywt_to_dataframe(
                df_y_raw,
                sites=common_sites,
                wavelet='db2',
                level=1,
                verbose=False
            )
        else:
            df_x_data = df_x_raw.copy()
            df_y_data = df_y_raw.copy()

        self.df_x_data = df_x_data
        self.df_y_data = df_y_data

        # =========================
        # 3. Filter valid sites
        # =========================
        # CrossVar 是 same-time window，所以只要求长度 >= seq_len。
        valid_sites, invalid_sites = filter_valid_sites_pair(
            df_x=df_x_data,
            df_y=df_y_data,
            sites=common_sites,
            seq_len=self.seq_len,
            pred_len=0,
            year_col='year'
        )

        # =========================
        # 4. Split train / val / test
        # =========================
        preferred_test_site_ranks = build_preferred_test_site_ranks(
            root_path=self.root_path,
            year_col='year',
            verbose=(self.verbose and self.set_type == 0)
        )

        train_sites, val_sites, test_sites = split_sites_by_cluster(
            valid_sites=valid_sites,
            cluster_map=self.cluster_map,
            seed=2030,
            train_ratio=0.6,
            filter_cid=self.filter_cid,
            fold_id=self.fold_id,
            preferred_test_site_ranks=preferred_test_site_ranks
        )

        self.train_sites = train_sites
        self.val_sites = val_sites
        self.test_sites = test_sites

        if self.verbose and self.set_type == 0:
            log_site_split(
                train_sites=train_sites,
                val_sites=val_sites,
                test_sites=test_sites,
                cluster_map=self.cluster_map,
                invalid_sites=invalid_sites
            )

        self.current_sites = select_current_sites(
            self.set_type,
            train_sites,
            val_sites,
            test_sites
        )

        # =========================
        # 5. Fit x/y scalers using train sites only
        # =========================
        self.x_scaler = None
        self.y_scaler = None

        if self.scale:
            self.x_scaler = fit_global_scaler_from_sites(
                df_x_data,
                train_sites=train_sites,
                year_col='year'
            )

            self.y_scaler = fit_global_scaler_from_sites(
                df_y_data,
                train_sites=train_sites,
                year_col='year'
            )

        self.static_scaler = fit_static_scaler(
            train_sites=train_sites,
            site_meta_dict=self.site_meta_dict
        )

        # =========================
        # 6. Build scaled dataframes for prototype / residual
        # =========================
        df_x_scaled = df_x_data.copy()
        df_y_scaled = df_y_data.copy()

        if self.scale and self.x_scaler is not None and self.y_scaler is not None:
            for site in common_sites:
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

        self.df_x_scaled = df_x_scaled
        self.df_y_scaled = df_y_scaled

        # =========================
        # 7. Residual cluster mean series
        # =========================
        self.cluster_time_means_x = {}
        self.cluster_time_means_y = {}

        if self.residual_prediction:
            # 这里不用原 helper build_cluster_time_means，
            # 因为 CrossVar 每个 site 会按 x/y 的 common years 对齐。
            # 所以先构造按 year index 的 cluster mean，后面每个 site 再按自己的 years reindex。
            self.cluster_time_means_x = self._build_cluster_mean_series(
                df=df_x_scaled,
                train_sites=train_sites,
                cluster_map=self.cluster_map,
                year_col='year'
            )

            self.cluster_time_means_y = self._build_cluster_mean_series(
                df=df_y_scaled,
                train_sites=train_sites,
                cluster_map=self.cluster_map,
                year_col='year'
            )

        # =========================
        # 8. Build per-site data lists
        # =========================
        self.static_feat_list = []
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []
        self.year_list = []
        self.valid_indices = []

        self.site_cluster_map = {}
        self.site_pearson_map = {}

        for site_idx, site in enumerate(self.current_sites):
            clean_s = clean_site_name(site)
            cluster_id = self.cluster_map.get(clean_s, -1)
            self.site_cluster_map[site_idx] = cluster_id

            # -------------------------
            # 8.1 align x/y by year
            # -------------------------
            years, x_vals, y_vals = align_two_site_series(
                df_x=df_x_scaled,
                df_y=df_y_scaled,
                site=site,
                year_col='year'
            )

            # aligned raw x/y for historical correlation
            years_raw, x_raw, y_raw = align_two_site_series(
                df_x=df_x_data,
                df_y=df_y_data,
                site=site,
                year_col='year'
            )

            # -------------------------
            # 8.2 historical correlation
            # -------------------------
            hist_corr = compute_detrended_spearman(
                x=x_raw,
                y=y_raw,
                min_len=30
            )

            self.site_pearson_map[site] = hist_corr

            # -------------------------
            # 8.3 static feature
            # -------------------------
            static_scaled = get_static_scaled(
                site,
                self.site_meta_dict,
                self.static_scaler
            )

            # CrossVar static feature:
            # [lat, sin_lon, cos_lon, elevation, hist_corr]
            static_feat = np.concatenate(
                [static_scaled, np.array([hist_corr], dtype=np.float32)],
                axis=0
            )

            # -------------------------
            # 8.4 year stamp
            # -------------------------
            data_stamp = make_year_stamp(years)

            self.static_feat_list.append(static_feat.astype(np.float32))
            self.data_x_list.append(x_vals.astype(np.float32))
            self.data_y_list.append(y_vals.astype(np.float32))
            self.data_stamp_list.append(data_stamp.astype(np.float32))
            self.year_list.append(years.astype(np.float32))

            seq_count = len(x_vals) - self.seq_len + 1
            for i in range(max(0, seq_count)):
                self.valid_indices.append((site_idx, i))

        # =========================
        # 9. Few-shot truncation
        # =========================
        if self.set_type == 0 and self.percent < 100:
            keep_len = int(len(self.valid_indices) * (self.percent / 100))
            self.valid_indices = self.valid_indices[:keep_len]

        # =========================
        # 10. MAML support length
        # =========================
        # CrossVar 是 same-time windows，所以 pred_len=0。
        self.support_len_list, self.support_len = compute_support_lengths(
            data_list=self.data_x_list,
            seq_len=self.seq_len,
            pred_len=0,
            support_ratio=self.support_ratio,
            default_support_len=100
        )

        # =========================
        # 11. Cluster prototypes
        # =========================
        # prototype 用 x 变量构造，且用 scaled x，保证和 __getitem__ 的 local_series 同尺度。
        self.prototypes = build_cluster_prototypes(
            train_sites=train_sites,
            df=df_x_scaled,
            cluster_map=self.cluster_map,
            support_len=self.support_len,
            target_cids=(0, 1, 2),
            year_col='year'
        )

    def _build_cluster_mean_series(self, df, train_sites, cluster_map, year_col='year'):
        """
        Build cluster mean series indexed by year.

        Return:
            {
                cid: pd.Series(index=year, values=cluster_mean)
            }

        注意:
        - 这里使用 skipna mean。
        - 后面每个 site 会根据自己的 aligned years reindex。
        """
        df_indexed = df.set_index(year_col)

        cluster_mean = {}

        for cid in (0, 1, 2):
            c_sites = [
                s for s in train_sites
                if cluster_map.get(clean_site_name(s), -1) == cid
            ]

            c_sites = [s for s in c_sites if s in df_indexed.columns]

            if len(c_sites) == 0:
                continue

            mean_series = df_indexed[c_sites].mean(axis=1, skipna=True)
            mean_series = mean_series.replace([np.inf, -np.inf], np.nan)
            mean_series = mean_series.ffill().bfill().fillna(0.0)

            cluster_mean[cid] = mean_series

        return cluster_mean

    def _get_aligned_cluster_mean(self, cluster_mean_dict, cluster_id, years, total_len):
        """
        Get cluster mean aligned to one site's years.

        Return shape:
            [T, 1]
        """
        if cluster_id not in cluster_mean_dict:
            return np.zeros((total_len, 1), dtype=np.float32)

        mean_series = cluster_mean_dict[cluster_id]

        # years 可能是 float32，但 index 通常是 int year。
        # reindex 时 pandas 一般能匹配 1800.0 和 1800；
        # 如果有缺失，就用 ffill/bfill/0 补。
        aligned = mean_series.reindex(years)
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        aligned = aligned.ffill().bfill().fillna(0.0)

        arr = aligned.values.astype(np.float32).reshape(-1, 1)

        if len(arr) != total_len:
            arr = arr[:total_len]
            if len(arr) < total_len:
                pad = np.zeros((total_len - len(arr), 1), dtype=np.float32)
                arr = np.vstack([arr, pad])

        return arr

    def __getitem__(self, index):
        # site-level MAML:
        # one item = one site/task
        site_idx = index

        site_name = self.current_sites[site_idx]
        cluster_id = self.site_cluster_map.get(site_idx, -1)

        data_x = self.data_x_list[site_idx]
        data_y = self.data_y_list[site_idx]
        data_stamp = self.data_stamp_list[site_idx]
        years = self.year_list[site_idx]

        total_len = len(data_x)
        site_support_len = self.support_len_list[site_idx]

        # =========================
        # 1. Optional residual prediction
        # =========================
        if self.target_anchor_residual:
            anchor_len = max(1, min(self.target_anchor_len, site_support_len, total_len))
            anchor_value = np.mean(data_y[site_support_len - anchor_len:site_support_len], axis=0, keepdims=True)
            mean_x = np.zeros_like(data_x, dtype=np.float32)
            mean_y = np.repeat(anchor_value.astype(np.float32), total_len, axis=0)
            data_x_model = data_x
            data_y_model = data_y - mean_y
        elif self.residual_prediction:
            mean_x = self._get_aligned_cluster_mean(
                cluster_mean_dict=self.cluster_time_means_x,
                cluster_id=cluster_id,
                years=years,
                total_len=total_len
            )

            mean_y = self._get_aligned_cluster_mean(
                cluster_mean_dict=self.cluster_time_means_y,
                cluster_id=cluster_id,
                years=years,
                total_len=total_len
            )

            data_x_model, data_y_model, mean_x, mean_y = apply_residual(
                data_x=data_x,
                data_y=data_y,
                cluster_id=cluster_id,
                cluster_mean_x=mean_x,
                cluster_mean_y=mean_y
            )
        else:
            data_x_model = data_x
            data_y_model = data_y
            mean_x = np.zeros_like(data_x, dtype=np.float32)
            mean_y = np.zeros_like(data_y, dtype=np.float32)

        # =========================
        # 2. Same-time support/query windows
        # =========================
        support_x = data_x_model[:site_support_len]
        mu = support_x.mean(axis=0, keepdims=True)
        std = np.sqrt(support_x.var(axis=0, keepdims=True) + 1e-5)
        data_x_model = ((data_x_model - mu) / std).astype(np.float32)

        windows = build_same_time_windows(
            data_x=data_x_model,
            data_y=data_y_model,
            data_stamp=data_stamp,
            years=years,
            seq_len=self.seq_len,
            support_len=site_support_len,
            stride=1
        )

        query_x_mean, query_y_mean = build_mean_windows_same_time(
            mean_x=mean_x,
            mean_y=mean_y,
            seq_len=self.seq_len,
            support_len=site_support_len,
            total_len=total_len,
            stride=1
        )

        # =========================
        # 3. Static feature
        # =========================
        static_feat = self.static_feat_list[site_idx]

        # local stats / prototype similarity 用 x 的 support 段
        local_series = data_x[:site_support_len].flatten()

        include_cluster_onehot = self.use_old_cluster_onehot
        include_prototype = True

        maml_static_feat = build_maml_static_feature(
            cluster_id=cluster_id,
            static_feat=static_feat,
            local_series=local_series,
            prototypes=self.prototypes,
            support_len=self.support_len,
            include_cluster_onehot=include_cluster_onehot,
            include_prototype=include_prototype,
            include_local_stats=True,
            target_cids=(0, 1, 2)
        )

        if self.use_encoder_proto_static:
            maml_static_feat = append_encoder_proto_static(
                maml_static_feat=maml_static_feat,
                site_name=site_name,
                proto_sim_map=self.encoder_proto_sim_map,
                proto_dim=self.encoder_proto_dim,
                proto_label_map=self.encoder_proto_label_map,
                include_proto_onehot=self.use_pretrain_cluster_onehot
            )

        # =========================
        # 4. Return standard MAML tuple
        # =========================
        return pack_maml_return(
            windows=windows,
            maml_static_feat=maml_static_feat,
            query_x_mean=query_x_mean,
            query_y_mean=query_y_mean
        )

    def __len__(self):
        # site-level MAML dataset
        return len(self.current_sites)

    def inverse_transform(self, data, var='y'):
        """
        反归一化。

        var='x':
            使用 x_scaler

        var='y':
            使用 y_scaler，默认用于还原模型预测目标变量。
        """
        if not self.scale:
            return data

        if var == 'x':
            if self.x_scaler is not None:
                return self.x_scaler.inverse_transform(data)
            return data

        else:
            if self.y_scaler is not None:
                return self.y_scaler.inverse_transform(data)
            return data
        
class Dataset_FCR_CrossVar(Dataset_IceCore_CrossVar):
    """One simulation per task; day indices use the existing year field."""

    def __init__(self, root_path, x_data_path, y_data_path, flag='train',
                 size=(32, 0, 32), support_ratio=0.66, encoder_proto_path=None,
                 target_anchor_residual=True, target_anchor_len=16):
        self.seq_len = size[0]
        self.scale = True
        self.target_anchor_residual = target_anchor_residual
        self.target_anchor_len = target_anchor_len
        x = pd.read_csv(os.path.join(root_path, x_data_path)).set_index('year').sort_index()
        y = pd.read_csv(os.path.join(root_path, y_data_path)).set_index('year').sort_index()
        assert x.index.equals(y.index) and x.columns.equals(y.columns)
        assert np.isfinite(x.to_numpy()).all() and np.isfinite(y.to_numpy()).all()
        sites = [f'FCR_{i:04d}' for i in range(1, 51)]
        order = np.random.default_rng(2030).permutation(len(sites))
        shuffled = [sites[i] for i in order]
        n_train, n_val = int(len(sites) * 0.6), int(len(sites) * 0.2)
        self.train_sites = shuffled[:n_train]
        self.val_sites = shuffled[n_train:n_train + n_val]
        self.test_sites = shuffled[n_train + n_val:]
        self.current_sites = {'train': self.train_sites, 'val': self.val_sites,
                              'test': self.test_sites, 'pretrain': sites}[flag]
        self.support_len = int(len(x) * support_ratio)
        assert self.seq_len <= self.support_len <= len(x) - self.seq_len
        fit_sites = sites if flag == 'pretrain' else self.train_sites
        self.x_scaler = StandardScaler().fit(x[fit_sites].iloc[:self.support_len].to_numpy().reshape(-1, 1))
        self.y_scaler = StandardScaler().fit(y[fit_sites].iloc[:self.support_len].to_numpy().reshape(-1, 1))
        self.data_x_list = [self.x_scaler.transform(x[[site]]).astype(np.float32)
                            for site in self.current_sites]
        self.data_y_list = [self.y_scaler.transform(y[[site]]).astype(np.float32)
                            for site in self.current_sites]
        self.year_list = [x.index.to_numpy(dtype=np.float32)] * len(self.current_sites)
        self.data_stamp_list = [make_year_stamp(x.index)] * len(self.current_sites)
        self.support_len_list = [self.support_len] * len(self.current_sites)
        self.site_cluster_map = {}  # Random scenario split; no cluster-balanced sampling.
        if flag == 'pretrain':
            self.static_feat_list = [np.empty(0, dtype=np.float32)] * len(sites)
        else:
            sim_map, _, _, _ = load_encoder_proto_static(encoder_proto_path)
            self.static_feat_list = [sim_map[site] for site in self.current_sites]

    def __getitem__(self, index):
        x, y = self.data_x_list[index], self.data_y_list[index]
        support = x[:self.support_len]
        x = ((x - support.mean(0)) / np.sqrt(support.var(0) + 1e-5)).astype(np.float32)
        anchor = np.zeros((1, 1), dtype=np.float32)
        if self.target_anchor_residual:
            start = max(0, self.support_len - self.target_anchor_len)
            anchor = y[start:self.support_len].mean(0, keepdims=True)
        windows = build_same_time_windows(
            x, y - anchor, self.data_stamp_list[index], self.year_list[index],
            self.seq_len, self.support_len, stride=1)
        mean_x, mean_y = build_mean_windows_same_time(
            np.zeros_like(x), np.broadcast_to(anchor, y.shape),
            self.seq_len, self.support_len, len(x), stride=1)
        return pack_maml_return(windows, self.static_feat_list[index], mean_x, mean_y)


class Dataset_FCR_RelationPretrain(Dataset_IceCore_RelationPretrain):
    """Reuse relation window sampling on every scenario's support; no static prefix."""

    def __read_data__(self):
        data = Dataset_FCR_CrossVar(
            self.root_path, self.x_data_path, self.y_data_path, flag='pretrain',
            size=(self.seq_len, 0, self.seq_len), support_ratio=self.support_ratio)
        self.site_names = data.current_sites
        self.num_sites = len(self.site_names)
        self.data_x_list, self.data_y_list = [], []
        for values, target in ((data.data_x_list, self.data_x_list),
                               (data.data_y_list, self.data_y_list)):
            for series in values:
                history = series[:data.support_len]
                target.append(((history - history.mean(0)) /
                               np.sqrt(history.var(0) + 1e-5)).astype(np.float32))
        self.data_stamp_list = [make_year_stamp(years[:data.support_len])
                                for years in data.year_list]
        self.static_feat_list = data.static_feat_list

class Dataset_IceCore_MS(Dataset):
    """
    Multivariate site-level MAML Dataset.

    用途:
    - x_data_path 和 y_data_path 两个变量拼成多变量序列
    - x = [x_variable, y_variable]
    - y = [x_variable, y_variable]
    - 使用 forecasting-style window:
        x[t : t + seq_len]
        y[t + seq_len - label_len : t + seq_len + pred_len]

    注意:
    - cluster_result_path 保持自定义，不根据 target 自动切换
    - target 只用于决定:
        1. local stats 用哪个变量
        2. prototype 用哪个变量
        3. 默认 inverse_transform 用哪个 scaler
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S',
                 data_path=None,
                 x_data_path='accum/accum_100y.csv',
                 y_data_path='chem/chem.csv',
                 target='accum', scale=True, timeenc=0, freq='y',
                 percent=100, task_name='long_term_forecast', is_pretraining=0,
                 apply_pywt=False,
                 site_meta_path='dataset/icecore/data_sources.csv',
                 use_spectral_features=1,
                 cluster_result_path='cluster_results_cosine_zscore.csv',
                 residual_prediction=False,
                 pred_seperate=None,
                 support_ratio=None,
                 filter_cid=None,
                 use_encoder_proto_static=False,
                 encoder_proto_path=None,
                 use_old_cluster_onehot=False,
                 use_pretrain_cluster_onehot=False,
                 fold_id=None,
                 verbose=True):

        # =========================
        # Basic config
        # =========================
        if size is None:
            self.seq_len = 64
            self.label_len = 32
            self.pred_len = 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.x_data_path = x_data_path
        self.y_data_path = y_data_path
        self.site_meta_path = site_meta_path
        self.cluster_result_path = cluster_result_path
        self.fold_id = fold_id

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.task_name = task_name
        self.is_pretraining = is_pretraining

        self.apply_pywt = apply_pywt
        self.residual_prediction = residual_prediction
        self.support_ratio = support_ratio
        self.filter_cid = filter_cid
        self.verbose = verbose
        self.use_encoder_proto_static = use_encoder_proto_static
        self.encoder_proto_path = encoder_proto_path

        # 保留参数，避免 data_factory / argparse 调用时报 unexpected keyword
        self.use_spectral_features = use_spectral_features
        self.pred_seperate = pred_seperate
        self.pred_seperate_list = None
        self.encoder_proto_sim_map = {}
        self.encoder_proto_label_map = {}
        self.encoder_proto_dim = 0
        self.encoder_proto_metadata = {}
        if self.use_encoder_proto_static:
            self.encoder_proto_sim_map, self.encoder_proto_label_map, self.encoder_proto_dim, self.encoder_proto_metadata = load_encoder_proto_static(
                self.encoder_proto_path
            )

        self.var_names = [
            self._infer_variable_name(self.x_data_path),
            self._infer_variable_name(self.y_data_path),
        ]
        if self.var_names[0] == self.var_names[1]:
            raise ValueError(
                f"MS requires two different variables, got {self.var_names} "
                f"from {self.x_data_path} and {self.y_data_path}."
            )

        self.target_var = self._normalize_variable_name(self.target)
        if self.target_var not in self.var_names:
            raise ValueError(
                f"target={self.target!r} is not in configured MS variables "
                f"{self.var_names}. Set --target to one of them."
            )
        self.target_idx = self.var_names.index(self.target_var)

        self.__read_data__()

    @staticmethod
    def _normalize_variable_name(name):
        key = str(name).lower().replace('_', '').replace('-', '')
        if 'accum' in key:
            return 'accum'
        if 'chem' in key:
            return 'chem'
        if 'd18o' in key or 'δ18o' in key:
            return 'd18O'
        if 'dd' in key:
            return 'dD'
        return str(name)

    @classmethod
    def _infer_variable_name(cls, path):
        base = os.path.basename(str(path))
        parent = os.path.basename(os.path.dirname(str(path)))
        inferred = cls._normalize_variable_name(base)
        if inferred != base:
            return inferred
        return cls._normalize_variable_name(parent)

    def __read_data__(self):
        # =========================
        # 1. Load variable files
        # =========================
        var0_file = os.path.join(self.root_path, self.x_data_path)
        var1_file = os.path.join(self.root_path, self.y_data_path)

        df_var0_raw = pd.read_csv(var0_file)
        df_var1_raw = pd.read_csv(var1_file)

        df_var0_raw = clean_dataframe_columns(df_var0_raw)
        df_var1_raw = clean_dataframe_columns(df_var1_raw)

        df_var0_raw = df_var0_raw.sort_values(by='year', ascending=True).reset_index(drop=True)
        df_var1_raw = df_var1_raw.sort_values(by='year', ascending=True).reset_index(drop=True)

        self.cluster_map = load_cluster_map(self.cluster_result_path)
        self.site_meta_dict = load_site_meta(self.site_meta_path)

        common_sites = get_common_sites(df_var0_raw, df_var1_raw, year_col='year')

        # =========================
        # 2. Optional PyWT denoising
        # =========================
        if self.apply_pywt:
            df_var0_data = apply_pywt_to_dataframe(
                df_var0_raw,
                sites=common_sites,
                wavelet='db2',
                level=1,
                verbose=(self.verbose and self.set_type == 0)
            )

            df_var1_data = apply_pywt_to_dataframe(
                df_var1_raw,
                sites=common_sites,
                wavelet='db2',
                level=1,
                verbose=False
            )
        else:
            df_var0_data = df_var0_raw.copy()
            df_var1_data = df_var1_raw.copy()

        self.df_var0_data = df_var0_data
        self.df_var1_data = df_var1_data

        # =========================
        # 3. Filter valid sites
        # =========================
        # MS 是 forecasting window，所以要求长度 >= seq_len + pred_len。
        valid_sites, invalid_sites = filter_valid_sites_pair(
            df_x=df_var0_data,
            df_y=df_var1_data,
            sites=common_sites,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            year_col='year'
        )

        # =========================
        # 4. Split train / val / test
        # =========================
        # 注意:
        # split_sites_by_cluster 内部逻辑是:
        #   先对全部 valid_sites split
        #   再根据 filter_cid 过滤
        # 这样 filter_cid 和 non-filter_cid 的划分保持一致。
        preferred_test_site_ranks = build_preferred_test_site_ranks(
            root_path=self.root_path,
            year_col='year',
            verbose=(self.verbose and self.set_type == 0)
        )

        train_sites, val_sites, test_sites = split_sites_by_cluster(
            valid_sites=valid_sites,
            cluster_map=self.cluster_map,
            seed=2030,
            train_ratio=0.6,
            filter_cid=self.filter_cid,
            fold_id=self.fold_id,
            preferred_test_site_ranks=preferred_test_site_ranks
        )

        self.train_sites = train_sites
        self.val_sites = val_sites
        self.test_sites = test_sites

        if self.verbose and self.set_type == 0:
            log_site_split(
                train_sites=train_sites,
                val_sites=val_sites,
                test_sites=test_sites,
                cluster_map=self.cluster_map,
                invalid_sites=invalid_sites
            )

        self.current_sites = select_current_sites(
            self.set_type,
            train_sites,
            val_sites,
            test_sites
        )

        # =========================
        # 5. Fit global scalers using train sites only
        # =========================
        # 修改点:
        # 原版 MS 是 columnwise scaler:
        #     [T, num_train_sites]
        # 这里改成和 S / CrossVar 一致的 global scaler:
        #     所有 train site value vstack 成 [N, 1]
        self.var_scalers = {}

        if self.scale:
            self.var_scalers[self.var_names[0]] = fit_global_scaler_from_sites(
                df_var0_data,
                train_sites=train_sites,
                year_col='year'
            )

            self.var_scalers[self.var_names[1]] = fit_global_scaler_from_sites(
                df_var1_data,
                train_sites=train_sites,
                year_col='year'
            )

        self.static_scaler = fit_static_scaler(
            train_sites=train_sites,
            site_meta_dict=self.site_meta_dict
        )

        # =========================
        # 6. Build scaled dataframes
        # =========================
        df_var0_scaled = self._transform_df_by_global_scaler(
            df=df_var0_data,
            sites=common_sites,
            scaler=self.var_scalers.get(self.var_names[0]),
            scale=self.scale
        )

        df_var1_scaled = self._transform_df_by_global_scaler(
            df=df_var1_data,
            sites=common_sites,
            scaler=self.var_scalers.get(self.var_names[1]),
            scale=self.scale
        )

        self.df_var0_scaled = df_var0_scaled
        self.df_var1_scaled = df_var1_scaled

        # =========================
        # 7. Residual cluster mean series
        # =========================
        # 修改点:
        # 原版依赖重复列名 concat，这里显式构造 [var0, var1] 的 cluster mean。
        self.cluster_time_means = {}

        if self.residual_prediction:
            self.cluster_time_means = self._build_ms_cluster_mean_series(
                df_var0=df_var0_scaled,
                df_var1=df_var1_scaled,
                train_sites=train_sites,
                cluster_map=self.cluster_map,
                year_col='year'
            )

        # =========================
        # 8. Build per-site data lists
        # =========================
        self.static_feat_list = []
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []
        self.year_list = []
        self.valid_indices = []

        self.site_cluster_map = {}
        self.site_pearson_map = {}

        for site_idx, site in enumerate(self.current_sites):
            clean_s = clean_site_name(site)
            cluster_id = self.cluster_map.get(clean_s, -1)
            self.site_cluster_map[site_idx] = cluster_id

            # -------------------------
            # 8.1 Explicitly concat [var0, var1]
            # -------------------------
            years, data_vals = concat_multivariate_site(
                df_list=[df_var0_scaled, df_var1_scaled],
                site=site,
                year_col='year'
            )

            # raw aligned values for historical correlation
            years_raw, var0_raw, var1_raw = align_two_site_series(
                df_x=df_var0_data,
                df_y=df_var1_data,
                site=site,
                year_col='year'
            )

            # -------------------------
            # 8.2 historical correlation
            # -------------------------
            hist_corr = compute_detrended_spearman(
                x=var0_raw,
                y=var1_raw,
                min_len=30
            )

            self.site_pearson_map[site] = hist_corr

            # -------------------------
            # 8.3 static feature
            # -------------------------
            static_scaled = get_static_scaled(
                site,
                self.site_meta_dict,
                self.static_scaler
            )

            # MS static feature:
            # [lat, sin_lon, cos_lon, elevation, hist_corr]
            static_feat = np.concatenate(
                [static_scaled, np.array([hist_corr], dtype=np.float32)],
                axis=0
            )

            # -------------------------
            # 8.4 year stamp
            # -------------------------
            data_stamp = make_year_stamp(years)

            self.static_feat_list.append(static_feat.astype(np.float32))

            # MS:
            # x 和 y 都是 [var0, var1]
            self.data_x_list.append(data_vals.astype(np.float32))
            self.data_y_list.append(data_vals.astype(np.float32))

            self.data_stamp_list.append(data_stamp.astype(np.float32))
            self.year_list.append(years.astype(np.float32))

            seq_count = len(data_vals) - self.seq_len - self.pred_len + 1
            for i in range(max(0, seq_count)):
                self.valid_indices.append((site_idx, i))

        # =========================
        # 9. Few-shot truncation
        # =========================
        # 保留原接口。注意当前 __len__ 是 site-level，
        # 所以 valid_indices 主要用于兼容/debug，不直接控制 __len__。
        if self.set_type == 0 and self.percent < 100:
            keep_len = int(len(self.valid_indices) * (self.percent / 100))
            self.valid_indices = self.valid_indices[:keep_len]

        # =========================
        # 10. MAML support length
        # =========================
        # MS 是 forecasting-style window，所以 pred_len=self.pred_len。
        self.support_len_list, self.support_len = compute_support_lengths(
            data_list=self.data_x_list,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            support_ratio=self.support_ratio,
            default_support_len=100
        )

        # =========================
        # 11. Cluster prototypes
        # =========================
        # cluster_result_path 仍然完全自定义。
        df_for_proto = [df_var0_scaled, df_var1_scaled][self.target_idx]

        self.prototypes = build_cluster_prototypes(
            train_sites=train_sites,
            df=df_for_proto,
            cluster_map=self.cluster_map,
            support_len=self.support_len,
            target_cids=(0, 1, 2),
            year_col='year'
        )

    def _transform_df_by_global_scaler(self, df, sites, scaler, scale=True):
        """
        Transform each site column with one global scaler.

        和原版 MS 的区别:
        - 原版是 columnwise scaler，每个 train site 一个 mean/std
        - 这里是 global scaler，每个变量一个
        """
        df_scaled = df.copy()

        if (not scale) or scaler is None:
            return df_scaled

        for site in sites:
            if site not in df_scaled.columns:
                continue

            mask = df_scaled[site].notna()
            vals = df_scaled.loc[mask, [site]].values.astype(np.float32)

            if len(vals) > 0:
                df_scaled.loc[mask, site] = scaler.transform(vals).flatten()

        return df_scaled

    def _build_ms_cluster_mean_series(self, df_var0, df_var1, train_sites, cluster_map, year_col='year'):
        """
        Build multivariate cluster mean series.

        Return:
            {
                cid: pd.DataFrame(index=year, columns=self.var_names)
            }

        每个 cid 的 mean 是:
            var0_mean(year), var1_mean(year)

        注意:
        - cluster_result_path 保持自定义
        - 这里只是按这个 cluster map 聚合 train_sites
        """
        var0_indexed = df_var0.set_index(year_col)
        var1_indexed = df_var1.set_index(year_col)

        cluster_means = {}

        for cid in (0, 1, 2):
            c_sites = [
                s for s in train_sites
                if cluster_map.get(clean_site_name(s), -1) == cid
            ]

            c_sites = [
                s for s in c_sites
                if s in var0_indexed.columns and s in var1_indexed.columns
            ]

            if len(c_sites) == 0:
                continue

            var0_mean = var0_indexed[c_sites].mean(axis=1, skipna=True)
            var1_mean = var1_indexed[c_sites].mean(axis=1, skipna=True)

            mean_df = pd.DataFrame({
                self.var_names[0]: var0_mean,
                self.var_names[1]: var1_mean
            })

            mean_df = mean_df.replace([np.inf, -np.inf], np.nan)
            mean_df = mean_df.ffill().bfill().fillna(0.0)

            cluster_means[cid] = mean_df

        return cluster_means

    def _get_aligned_ms_cluster_mean(self, cluster_id, years, total_len):
        """
        Align cluster mean to one site's years.

        Return shape:
            [T, 2]
        """
        if cluster_id not in self.cluster_time_means:
            return np.zeros((total_len, 2), dtype=np.float32)

        mean_df = self.cluster_time_means[cluster_id]

        aligned = mean_df.reindex(years)
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        aligned = aligned.ffill().bfill().fillna(0.0)

        arr = aligned[self.var_names].values.astype(np.float32)

        if len(arr) != total_len:
            arr = arr[:total_len]

            if len(arr) < total_len:
                pad = np.zeros((total_len - len(arr), 2), dtype=np.float32)
                arr = np.vstack([arr, pad])

        return arr

    def __getitem__(self, index):
        # site-level MAML:
        # one item = one site/task
        site_idx = index

        site_name = self.current_sites[site_idx]
        cluster_id = self.site_cluster_map.get(site_idx, -1)

        data_x = self.data_x_list[site_idx]          # [T, 2]
        data_y = self.data_y_list[site_idx]          # [T, 2]
        data_stamp = self.data_stamp_list[site_idx]  # [T, 1]
        years = self.year_list[site_idx]             # [T]

        total_len = len(data_x)
        site_support_len = self.support_len_list[site_idx]

        # =========================
        # 1. Optional residual prediction
        # =========================
        if self.residual_prediction:
            c_mean = self._get_aligned_ms_cluster_mean(
                cluster_id=cluster_id,
                years=years,
                total_len=total_len
            )

            data_x_model, data_y_model, mean_x, mean_y = apply_residual(
                data_x=data_x,
                data_y=data_y,
                cluster_id=cluster_id,
                cluster_mean_x=c_mean,
                cluster_mean_y=c_mean
            )
        else:
            data_x_model = data_x
            data_y_model = data_y
            mean_x = np.zeros_like(data_x, dtype=np.float32)
            mean_y = np.zeros_like(data_y, dtype=np.float32)

        # =========================
        # 2. Forecast-style support/query windows
        # =========================
        windows = build_forecast_windows(
            data_x=data_x_model,
            data_y=data_y_model,
            data_stamp=data_stamp,
            years=years,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            support_len=site_support_len,
            stride=1
        )

        query_x_mean, query_y_mean = build_mean_windows_forecast(
            mean_x=mean_x,
            mean_y=mean_y,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            support_len=site_support_len,
            total_len=total_len,
            stride=1
        )

        # =========================
        # 3. Build MAML static feature
        # =========================
        static_feat = self.static_feat_list[site_idx]

        # MS 的 local stats / prototype similarity 只看 target 对应变量。
        local_series = data_x[:site_support_len, self.target_idx].flatten()

        include_cluster_onehot = self.filter_cid is None
        include_prototype = self.filter_cid is None

        maml_static_feat = build_maml_static_feature(
            cluster_id=cluster_id,
            static_feat=static_feat,
            local_series=local_series,
            prototypes=self.prototypes,
            support_len=self.support_len,
            include_cluster_onehot=include_cluster_onehot,
            include_prototype=include_prototype,
            include_local_stats=True,
            target_cids=(0, 1, 2)
        )

        # =========================
        # 4. Return standard MAML tuple
        # =========================
        return pack_maml_return(
            windows=windows,
            maml_static_feat=maml_static_feat,
            query_x_mean=query_x_mean,
            query_y_mean=query_y_mean
        )

    def __len__(self):
        # site-level MAML dataset
        return len(self.current_sites)

    def inverse_transform(self, data, var=None):
        """
        反归一化。

        参数:
        - var 为 self.var_names 中的变量名:
            使用对应变量的 scaler

        - var=None:
            如果 data 最后一维是 2，则同时还原 [var0, var1]
            如果 data 最后一维不是 2，则默认按 target 还原
        """
        if not self.scale:
            return data

        arr = np.asarray(data)

        if var is None:
            # 如果是 multivariate output: [..., 2]
            if arr.ndim >= 1 and arr.shape[-1] == 2:
                return self._inverse_transform_multivariate(arr)

            # 否则默认按 target 变量还原
            var = self.target_var

        var = self._normalize_variable_name(var)
        if var not in self.var_scalers:
            raise ValueError(f"var must be one of: None, {self.var_names}")

        return self._inverse_transform_single(arr, self.var_scalers.get(var))

    def _inverse_transform_single(self, arr, scaler):
        """
        Inverse transform single variable array with shape:
            [N]
            [N, 1]
            [B, T]
            [B, T, 1]
            etc.
        """
        if scaler is None:
            return arr

        original_shape = arr.shape
        flat = arr.reshape(-1, 1)
        inv = scaler.inverse_transform(flat)
        return inv.reshape(original_shape)

    def _inverse_transform_multivariate(self, arr):
        """
        Inverse transform multivariate array with last dim = 2.

        input shape:
            [..., 2]

        output shape:
            [..., 2]
        """
        original_shape = arr.shape

        flat = arr.reshape(-1, 2)

        columns = []
        for idx, var_name in enumerate(self.var_names):
            scaler = self.var_scalers.get(var_name)
            col = flat[:, [idx]]
            columns.append(
                scaler.inverse_transform(col) if scaler is not None else col
            )

        inv = np.concatenate(columns, axis=1)

        return inv.reshape(original_shape)


