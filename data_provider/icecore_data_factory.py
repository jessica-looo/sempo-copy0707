from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import WeightedRandomSampler
import torch
import collections

from data_provider.icecore_data_loader import (
    Dataset_IceCore_S,
    Dataset_IceCore_CrossVar,
    Dataset_IceCore_MS,
)
from data_provider.icecore_relation_pretrain_loader import Dataset_IceCore_RelationPretrain


data_dict = {
    'accum_s': Dataset_IceCore_S,
    'd18O_s': Dataset_IceCore_S,
    'chem_s': Dataset_IceCore_S,
    'accum_chem_crossvar': Dataset_IceCore_CrossVar,
    'chem_accum_crossvar': Dataset_IceCore_CrossVar,
    'd18O_accum_crossvar': Dataset_IceCore_CrossVar,
    'd18O_chem_crossvar': Dataset_IceCore_CrossVar,
    'icecore_ms': Dataset_IceCore_MS,
    'accum_chem_ms': Dataset_IceCore_MS,
    'chem_accum_ms': Dataset_IceCore_MS,
    'd18O_chem_ms': Dataset_IceCore_MS,
    'd18O_accum_ms': Dataset_IceCore_MS,
}


def get_icecore_task_type(args):
    task_type = getattr(args, 'icecore_task_type', None)
    if task_type is not None:
        task_type = str(task_type).lower()
        if task_type in ['s', 'crossvar', 'ms']:
            return task_type
        raise ValueError(f"Unknown icecore_task_type={task_type}. Expected s, crossvar, or ms.")

    data_name = str(args.data)
    if data_name.endswith('_crossvar'):
        return 'crossvar'
    if data_name.endswith('_ms') or data_name == 'icecore_ms':
        return 'ms'
    if data_name.endswith('_s'):
        return 's'

    raise ValueError(
        f"Cannot infer icecore_task_type from data={args.data}. "
        f"Pass --icecore_task_type s|crossvar|ms."
    )


def _get_common_kwargs(args, flag, timeenc):
    """
    三个冰芯 Dataset 共用参数。
    """
    return {
        'root_path': args.root_path,
        'flag': flag,
        'size': [args.seq_len, args.label_len, args.pred_len],
        'features': args.features,
        'target': args.target,
        'scale': True,
        'timeenc': timeenc,
        'freq': args.freq,
        'percent': args.percent,
        'task_name': args.task_name,
        'is_pretraining': args.is_pretraining,

        'apply_pywt': getattr(args, 'apply_pywt', False),
        'site_meta_path': getattr(args, 'site_meta_path', 'dataset/icecore/data_sources.csv'),
        'use_spectral_features': getattr(args, 'use_spectral_features', 1),
        'cluster_result_path': getattr(args, 'cluster_result_path', 'cluster_results_cosine_zscore.csv'),
        'residual_prediction': getattr(args, 'residual_prediction', False),
        'pred_seperate': getattr(args, 'pred_seperate', None),
        'support_ratio': getattr(args, 'support_ratio', None),
        'filter_cid': getattr(args, 'filter_cid', None),
        'use_encoder_proto_static': getattr(args, 'use_encoder_proto_static', False),
        'encoder_proto_path': getattr(args, 'encoder_proto_path', None),
        'use_old_cluster_onehot': getattr(args, 'use_old_cluster_onehot', False),
        'use_pretrain_cluster_onehot': getattr(args, 'use_pretrain_cluster_onehot', False),
        'verbose': getattr(args, 'verbose', True),
        'fold_id': getattr(args, 'fold_id', None),  
    }


def _build_dataset(args, flag, timeenc):
    """
    根据 args.data 构建 Dataset。
    只支持:
        icecore_s
        icecore_crossvar
        icecore_ms
    """
    task_type = get_icecore_task_type(args)

    if task_type == 's':
        Data = Dataset_IceCore_S
    elif task_type == 'crossvar':
        Data = Dataset_IceCore_CrossVar
    elif task_type == 'ms':
        Data = Dataset_IceCore_MS
    else:
        raise ValueError(f"Unknown icecore_task_type={task_type}")

    if args.data not in data_dict:
        raise ValueError(
            f"Unknown icecore dataset: {args.data}. "
            f"Available options: {list(data_dict.keys())}"
        )

    kwargs = _get_common_kwargs(args, flag, timeenc)

    if task_type == 's':
        kwargs['data_path'] = args.data_path

    elif task_type in ['crossvar', 'ms']:
        kwargs['data_path'] = None
        kwargs['x_data_path'] = args.x_data_path
        kwargs['y_data_path'] = args.y_data_path
        if task_type == 'crossvar':
            kwargs['target_anchor_residual'] = getattr(args, 'target_anchor_residual', False)
            kwargs['target_anchor_len'] = getattr(args, 'target_anchor_len', 16)

    data_set = Data(**kwargs)
    if hasattr(data_set, 'target_idx'):
        args.target_dim = int(data_set.target_idx)

    return data_set


def infer_static_dim(args):
    """
    Infer the actual static feature dimension from the train dataset.

    This keeps bash scripts from manually tracking static_dim when optional
    encoder-prototype features are appended.
    """
    timeenc = 0 if args.embed != 'timeF' else 1
    old_verbose = getattr(args, 'verbose', True)
    args.verbose = False
    try:
        data_set = _build_dataset(args, 'train', timeenc)
    finally:
        args.verbose = old_verbose

    if len(data_set) == 0:
        raise ValueError("Cannot infer static_dim from an empty train dataset.")

    sample = data_set[0]
    static_feat = sample[8]
    return int(static_feat.shape[-1])


def _build_weighted_sampler(data_set):
    """
    根据 dataset.site_cluster_map 做 cluster-balanced sampling。

    只在 train 且非 multi-gpu 时使用。
    """
    if not hasattr(data_set, 'site_cluster_map'):
        return None

    cluster_map = data_set.site_cluster_map

    if len(cluster_map) == 0:
        return None

    cluster_counts = collections.Counter(cluster_map.values())

    weights = [
        1.0 / cluster_counts[cluster_map[i]]
        for i in range(len(data_set))
    ]

    weights = torch.DoubleTensor(weights)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(data_set),
        replacement=True
    )


def data_provider(args, flag):
    """
    Return:
        data_set, data_loader
    """
    if args.task_name != 'long_term_forecast':
        raise NotImplementedError(
            f"Only long_term_forecast is supported, got {args.task_name}"
        )

    if getattr(args, 'is_relation_pretraining', False):
        if flag != 'train':
            raise ValueError("Relation pretraining only builds the train split.")
        data_set = Dataset_IceCore_RelationPretrain(
            root_path=args.root_path,
            x_data_path=args.x_data_path,
            y_data_path=args.y_data_path,
            seq_len=args.seq_len,
            samples_per_site=getattr(args, 'relation_pretrain_samples_per_site', 8),
            support_ratio=getattr(args, 'support_ratio', None),
            apply_pywt=getattr(args, 'apply_pywt', False),
            site_meta_path=getattr(args, 'site_meta_path', 'dataset/icecore/data_sources.csv'),
            scale=True,
            verbose=getattr(args, 'verbose', True),
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        print(flag, len(data_set))
        return data_set, data_loader

    timeenc = 0 if args.embed != 'timeF' else 1

    data_set = _build_dataset(args, flag, timeenc)

    if flag == 'test':
        batch_size = 1
        shuffle_flag = False
        drop_last = False
    else:
        batch_size = args.batch_size
        shuffle_flag = True
        drop_last = False

    sampler = None

    if getattr(args, 'use_multi_gpu', False):
        sampler = DistributedSampler(
            data_set,
            shuffle=shuffle_flag
        )
        shuffle_flag = False

    else:
        if flag == 'train':
            sampler = _build_weighted_sampler(data_set)
            if sampler is not None:
                shuffle_flag = False

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    print(flag, len(data_set))

    return data_set, data_loader
