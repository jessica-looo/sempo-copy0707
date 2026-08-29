from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_IceCore_M, Dataset_IceCore_Accum_Year, Dataset_IceCore_Accum_Site,Dataset_IceCore_CrossVar,Dataset_IceCore_MS
from data_provider.data_loader_benchmark import UTSDDatasetBenchmark, CIDatasetBenchmark
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'UTSD': UTSDDatasetBenchmark,
    'CI': CIDatasetBenchmark,
    'icecore': Dataset_IceCore_M,
    'icecore_accum_year': Dataset_IceCore_Accum_Year,
    'icecore_accum_site': Dataset_IceCore_Accum_Site,
    'icecore_d18o_site': Dataset_IceCore_Accum_Site,
    'icecore_crossvar': Dataset_IceCore_CrossVar,
    'icecore_ms': Dataset_IceCore_MS,
    'icecore_chem_site': Dataset_IceCore_Accum_Site,
}


def data_provider(args, flag):
    Data = data_dict[args.data]    
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'long_term_forecast' or args.task_name == 'long_term_forecast_chronos':          
        
        # 1. 构建所有 Dataset 共用的基础参数字典
        data_kwargs = {
            'root_path': args.root_path,
            'data_path': args.data_path,
            'flag': flag,
            'size': [args.seq_len, args.label_len, args.pred_len],
            'features': args.features,
            'scale': True,
            'timeenc': timeenc,
            'freq': args.freq,
            'percent': args.percent,
            'task_name': args.task_name,
            'is_pretraining': args.is_pretraining,
            'target': args.target,
        }

        # 2. 根据所选的数据集，动态追加特有参数
        if args.data in ('icecore_accum_site', 'icecore_chem_site','icecore_d18o_site', 'icecore_crossvar','icecore_ms'):
            # 使用 getattr 安全获取 args.mode，如果命令行没传，默认用 'random'
            data_kwargs['apply_pywt'] = getattr(args, 'apply_pywt', False)
            # data_kwargs['use_spectral_features'] = getattr(args, 'use_spectral_features', 1)
            data_kwargs['cluster_result_path'] = getattr(args, 'cluster_result_path', 'cluster_results_cosine_zscore.csv')
            data_kwargs['residual_prediction'] = getattr(args, 'residual_prediction', False)
            data_kwargs['pred_seperate'] = getattr(args, 'pred_seperate', False)
            data_kwargs['support_ratio'] = getattr(args, 'support_ratio', None)
            if args.data in ('icecore_crossvar','icecore_ms'):
                data_kwargs['x_data_path'] = getattr(args, 'x_data_path', None)
                data_kwargs['y_data_path'] = getattr(args, 'y_data_path', None)
                data_kwargs['filter_cid'] = getattr(args, 'filter_cid', None)
                data_kwargs['data_path'] = None
            # import pdb;pdb.set_trace()
        # 3. 使用 ** 解包字典进行实例化
        data_set = Data(**data_kwargs)

        sampler = None
        # 只有训练阶段，并且是冰芯数据才做加权采样平衡
        if flag == 'train' and args.data in ('icecore_accum_site', 'icecore_d18o_site','icecore_crossvar','icecore_ms'):
            from torch.utils.data import WeightedRandomSampler
            import collections
            
            # 获取数据集中每个 Task 对应的 Cluster ID
            # 注意：这里的 site_cluster_map 是我们在 data_loader 阶段第二步里存下来的
            cluster_map = data_set.site_cluster_map
            
            # 统计当前集合中各簇的数量 (比如 C0: 10, C1: 17, C2: 6)
            cluster_counts = collections.Counter(cluster_map.values())
            
            # 为数据集中的每一个样本分配权重 (该类总数的倒数)
            weights = [1.0 / cluster_counts[cluster_map[i]] for i in range(len(data_set))]
            weights = torch.DoubleTensor(weights)
            
            # 创建采样器 (num_samples=len(data_set) 保证一个 Epoch 的迭代次数不变)
            # replacement=True 允许 C2 这种少样本簇被重复抽样
            sampler = WeightedRandomSampler(weights, num_samples=len(data_set), replacement=True)
            shuffle_flag = False # 使用了自定义 Sampler，PyTorch 要求关闭自带的 shuffle
        
        print(flag, len(data_set))
        if args.use_multi_gpu:
            train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
            data_loader = DataLoader(data_set,
                                     batch_size=args.batch_size,
                                     sampler=train_datasampler,
                                     num_workers=args.num_workers,
                                     persistent_workers=(args.num_workers > 0),
                                     pin_memory=True,
                                     drop_last=drop_last,
                                     )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                sampler=sampler,
                num_workers=args.num_workers,
                drop_last=drop_last)
        return data_set, data_loader
    else:
        raise NotImplementedError
