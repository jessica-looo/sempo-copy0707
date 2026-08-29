import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.icecore_data_factory import _build_dataset, get_icecore_task_type, infer_static_dim
from models import SEMPO

'''
python scripts/extract_encoder_prototypes.py \
  --checkpoint checkpoints/long_term_forecast_SEMPO_d18O_accum_crossvar_protoNone_ftM_sl32_ll48_pl32_pl16_dm256_nh2_el3_dl3_df128_fc1_ebtimeF_dtTrue_0/checkpoint.pth \
  --num_prototypes 6 \
  --output_path prototypes/d18O_accum_crossvar_encoder_proto_k6.npz
'''


def _parse_args():
    parser = argparse.ArgumentParser(description='Extract offline SEMPO encoder prototypes')

    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--icecore_task_type', type=str, default=None, choices=['s', 'crossvar', 'ms'])
    parser.add_argument('--root_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='accum.csv')
    parser.add_argument('--x_data_path', type=str, default='accum.csv')
    parser.add_argument('--y_data_path', type=str, default='merged_chem.csv')
    parser.add_argument('--cluster_result_path', type=str, default=None)
    parser.add_argument('--site_meta_path', type=str, default='dataset/icecore/data_sources.csv')

    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--target', type=str, default='accum')
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--label_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=16)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--c_in', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--distil', type=bool, default=True)
    parser.add_argument('--head_type', type=str, default='prediction')
    parser.add_argument('--domain_len', type=int, default=128)
    parser.add_argument('--horizon_lengths', type=int, nargs='+', default=[16])

    parser.add_argument('--use_static_features', action='store_true')
    parser.add_argument('--use_static_bias', action='store_true')
    parser.add_argument('--use_static_concat', action='store_true')
    parser.add_argument('--use_static_kv', action='store_true')
    parser.add_argument('--meta_prefix_len', type=int, default=8)
    parser.add_argument('--static_emb_dim', type=int, default=8)
    parser.add_argument('--use_film', action='store_true')
    parser.add_argument('--film_hidden_dim', type=int, default=64)
    parser.add_argument('--film_scale_shift', action='store_true')
    parser.add_argument('--filter_approx', action='store_true')
    parser.add_argument('--use_swt_wavelet', action='store_true')
    parser.add_argument('--use_static_conditioned_wavelet', action='store_true')

    parser.add_argument('--apply_pywt', action='store_true')
    parser.add_argument('--use_spectral_features', type=int, default=1)
    parser.add_argument('--residual_prediction', action='store_true')
    parser.add_argument('--pred_seperate', type=str, default=None)
    parser.add_argument('--filter_cid', type=int, default=None)
    parser.add_argument('--support_ratio', type=float, default=None)
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_pretraining', type=int, default=0)
    parser.add_argument('--freq', type=str, default='y')
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--prototype_mode', type=str, default='auto', choices=['auto', 'x_only', 'relation'])
    parser.add_argument('--num_prototypes', type=int, default=8)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2021)

    return parser.parse_args()


def _provided_option_names(argv):
    names = set()
    for token in argv[1:]:
        if not token.startswith('--'):
            continue
        name = token[2:].split('=', 1)[0].replace('-', '_')
        names.add(name)
    return names


def _load_checkpoint_config(args, provided_names):
    config_path = args.config_path
    if config_path is None:
        config_path = str(Path(args.checkpoint).resolve().parent / 'checkpoint_config.json')

    if not os.path.exists(config_path):
        print(f'Checkpoint config not found, using CLI/default args: {config_path}')
        return args

    with open(config_path, 'r') as f:
        config = json.load(f)

    saved_args = config.get('args', config)
    skip_keys = {
        'checkpoint',
        'config_path',
        'num_prototypes',
        'output_path',
        'device',
        'seed',
        'use_encoder_proto_static',
        'encoder_proto_path',
    }

    for key, value in saved_args.items():
        if key in skip_keys or key in provided_names:
            continue
        setattr(args, key, value)

    print(f'Loaded checkpoint config from: {config_path}')
    return args


def _validate_required_args(args):
    required = ['data', 'root_path', 'cluster_result_path']
    missing = [name for name in required if getattr(args, name, None) in (None, '')]
    if missing:
        raise ValueError(
            f"Missing required args after reading checkpoint config: {missing}. "
            f"Pass them on CLI or provide --config_path."
        )


def _load_checkpoint(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    model_state = model.state_dict()
    filtered = {}
    for key, value in state.items():
        clean_key = key.replace('module.', '', 1)
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            filtered[clean_key] = value

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f'Loaded {len(filtered)} tensors from {checkpoint_path}')
    if missing:
        print(f'Missing tensors kept from current init: {len(missing)}')
    if unexpected:
        print(f'Unexpected tensors ignored: {len(unexpected)}')


def _l2_normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _safe_corr(x, y):
    x = np.asarray(x, dtype=np.float32).flatten()
    y = np.asarray(y, dtype=np.float32).flatten()
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x = x[:n]
    y = y[:n]
    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return 0.0
    value = np.corrcoef(x, y)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def _lag_corr_features(x, y, max_lag=2):
    x = np.asarray(x, dtype=np.float32).flatten()
    y = np.asarray(y, dtype=np.float32).flatten()
    n = min(len(x), len(y))
    if n < 3:
        return 0.0, 0.0
    x = x[:n]
    y = y[:n]

    best_corr = 0.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x_lag = x[-lag:]
            y_lag = y[:len(x_lag)]
        elif lag > 0:
            x_lag = x[:-lag]
            y_lag = y[lag:]
        else:
            x_lag = x
            y_lag = y

        corr = _safe_corr(x_lag, y_lag)
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return float(best_corr), float(best_lag)


def _linear_relation_features(x, y):
    x = np.asarray(x, dtype=np.float32).flatten()
    y = np.asarray(y, dtype=np.float32).flatten()
    n = min(len(x), len(y))
    if n < 3:
        return 0.0, 0.0
    x = x[:n]
    y = y[:n]

    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-6:
        slope = 0.0
    else:
        slope = float(np.cov(x, y, bias=True)[0, 1] / (x_std ** 2 + 1e-8))
    std_ratio = float(y_std / (x_std + 1e-6))

    if not np.isfinite(slope):
        slope = 0.0
    if not np.isfinite(std_ratio):
        std_ratio = 0.0

    return slope, std_ratio


def _relation_stats(support_x, support_y):
    x = support_x.detach().cpu().numpy().astype(np.float32).flatten()
    y = support_y.detach().cpu().numpy().astype(np.float32).flatten()
    pearson = _safe_corr(x, y)
    lag_corr, best_lag = _lag_corr_features(x, y, max_lag=2)
    slope, std_ratio = _linear_relation_features(x, y)
    return np.asarray([pearson, lag_corr, best_lag, slope, std_ratio], dtype=np.float32)


@torch.no_grad()
def _extract_encoder_z(model, x_enc, x_mark_enc, x_dec, x_mark_dec, x_static=None):
    if model.use_static_features and x_static is not None:
        e_site = model.static_encoder(x_static)
    else:
        e_site = None

    x = model.revin_layer_x(x_enc, 'norm')

    if model.use_static_concat and e_site is not None:
        e_site_repeat = e_site.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([x, e_site_repeat], dim=-1)
        x = model.feature_fusion(x)

    x = model.projection_x(x.permute(0, 2, 1)).permute(0, 2, 1)
    x = model.decomposed_wavelet_learning(x, e_site=e_site)

    x = x[:, :, model.s_begin:, :]
    x = x.unfold(dimension=2, size=model.patch_len, step=model.stride)
    freq_num, bs, patch_num, n_vars, patch_len = x.shape
    x = x.reshape(-1, patch_num, n_vars, patch_len)

    x = model.patch_embed(x)
    x = x.transpose(1, 2)
    u = torch.reshape(x, (-1, n_vars * patch_num, model.d_model))
    u = model.dropout1(u + model.W_pos)

    u_d = model.mixture_of_experts(u, bs=bs, encode=True)
    if model.use_static_kv and e_site is not None:
        u_d_static = model.static_to_encoder_prefix(e_site, bs)
        u_d = torch.cat([u_d, u_d_static], dim=3)

    x = model.encoder(u, u_d)
    x = x.reshape(-1, n_vars, patch_num, model.d_model)
    x = x.permute(0, 1, 3, 2)
    x = x.view(-1, bs, n_vars, model.d_model, patch_num).mean(dim=0)

    return x.mean(dim=(1, 3))


def _site_representation(model, sample, device):
    support_x, support_y, support_x_mark, support_y_mark, _, _, _, _, static_feat, _, _, _ = sample

    support_x = support_x.float().to(device)
    support_y = support_y.float().to(device)
    support_x_mark = support_x_mark.float().to(device)
    support_y_mark = support_y_mark.float().to(device)
    static_feat = static_feat.float().to(device)
    static_feat = static_feat.unsqueeze(0).repeat(support_x.size(0), 1)

    z = _extract_encoder_z(
        model=model,
        x_enc=support_x,
        x_mark_enc=support_x_mark,
        x_dec=support_y,
        x_mark_dec=support_y_mark,
        x_static=static_feat,
    )
    return z.mean(dim=0).detach().cpu().numpy().astype(np.float32)


def _relation_site_representation(model, sample, device):
    support_x, support_y, support_x_mark, support_y_mark, _, _, _, _, static_feat, _, _, _ = sample

    support_x = support_x.float().to(device)
    support_y = support_y.float().to(device)
    support_x_mark = support_x_mark.float().to(device)
    support_y_mark = support_y_mark.float().to(device)
    static_feat = static_feat.float().to(device)
    static_feat_rep = static_feat.unsqueeze(0).repeat(support_x.size(0), 1)

    z_x = _extract_encoder_z(
        model=model,
        x_enc=support_x,
        x_mark_enc=support_x_mark,
        x_dec=support_y,
        x_mark_dec=support_y_mark,
        x_static=static_feat_rep,
    ).mean(dim=0)

    z_y = _extract_encoder_z(
        model=model,
        x_enc=support_y,
        x_mark_enc=support_y_mark,
        x_dec=support_y,
        x_mark_dec=support_y_mark,
        x_static=static_feat_rep,
    ).mean(dim=0)

    z_x_np = z_x.detach().cpu().numpy().astype(np.float32)
    z_y_np = z_y.detach().cpu().numpy().astype(np.float32)
    stats = _relation_stats(support_x, support_y)

    return np.concatenate(
        [
            z_x_np,
            z_y_np,
            z_y_np - z_x_np,
            z_x_np * z_y_np,
            stats,
        ],
        axis=0,
    ).astype(np.float32)


def main():
    args = _parse_args()
    provided_names = _provided_option_names(sys.argv)
    args = _load_checkpoint_config(args, provided_names)
    _validate_required_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.use_gpu = args.device.startswith('cuda')
    args.use_multi_gpu = False
    args.gpu = 0
    args.checkpoints = './checkpoints/'
    args.setting = 'extract_encoder_prototypes'
    args.use_encoder_proto_static = False
    args.encoder_proto_path = None

    task_type = get_icecore_task_type(args)
    if args.prototype_mode == 'auto':
        prototype_mode = 'relation' if task_type == 'crossvar' else 'x_only'
    else:
        prototype_mode = args.prototype_mode
    if prototype_mode == 'relation' and task_type != 'crossvar':
        raise ValueError("prototype_mode=relation is currently supported only for --icecore_task_type crossvar")

    args.static_dim = infer_static_dim(args) if args.use_static_features else 0

    timeenc = 0 if args.embed != 'timeF' else 1
    datasets = {
        flag: _build_dataset(args, flag, timeenc)
        for flag in ('train', 'val', 'test')
    }

    if len(datasets['train']) < args.num_prototypes:
        raise ValueError(
            f"num_prototypes={args.num_prototypes} exceeds train site count={len(datasets['train'])}"
        )

    first_sample = datasets['train'][0]
    args.c_in = int(first_sample[0].shape[-1])

    device = torch.device(args.device)
    model = SEMPO.Model(args).to(device)
    _load_checkpoint(model, args.checkpoint, device)
    model.eval()

    site_names = []
    site_reps = []
    split_names = []

    for flag, dataset in datasets.items():
        for idx in range(len(dataset)):
            site = dataset.current_sites[idx]
            if prototype_mode == 'relation':
                rep = _relation_site_representation(model, dataset[idx], device)
            else:
                rep = _site_representation(model, dataset[idx], device)
            site_names.append(site)
            site_reps.append(rep)
            split_names.append(flag)

    site_reps = np.stack(site_reps, axis=0).astype(np.float32)
    train_mask = np.asarray([s == 'train' for s in split_names], dtype=bool)
    train_reps = site_reps[train_mask]

    kmeans = KMeans(n_clusters=args.num_prototypes, random_state=args.seed, n_init=20)
    kmeans.fit(train_reps)
    prototypes = kmeans.cluster_centers_.astype(np.float32)

    site_sim = _l2_normalize(site_reps) @ _l2_normalize(prototypes).T
    site_sim = site_sim.astype(np.float32)

    metadata = {
        'checkpoint': args.checkpoint,
        'data': args.data,
        'features': args.features,
        'target': args.target,
        'seq_len': args.seq_len,
        'label_len': args.label_len,
        'pred_len': args.pred_len,
        'patch_len': args.patch_len,
        'stride': args.stride,
        'c_in': args.c_in,
        'd_model': args.d_model,
        'num_prototypes': args.num_prototypes,
        'prototype_mode': prototype_mode,
        'icecore_task_type': task_type,
        'pooling': 'mean_windows_then_mean_vars_patches',
        'similarity': 'cosine',
        'cluster_method': 'kmeans',
        'train_site_count': int(train_mask.sum()),
    }
    if prototype_mode == 'relation':
        metadata['relation_components'] = [
            'z_x',
            'z_y',
            'z_y_minus_z_x',
            'z_x_times_z_y',
            'pearson_corr',
            'lag_corr_max_pm2',
            'best_lag',
            'slope_x_to_y',
            'std_ratio_y_over_x',
        ]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        site_names=np.asarray(site_names),
        split=np.asarray(split_names),
        site_reps=site_reps,
        prototypes=prototypes,
        site_sim=site_sim,
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    print(f'Saved encoder prototypes to {output_path}')
    print(f'site_sim shape: {site_sim.shape}, prototypes shape: {prototypes.shape}')


if __name__ == '__main__':
    main()
