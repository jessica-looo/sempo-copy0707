import argparse
import os
import torch
import torch.distributed as dist
from exp.maml_exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_relation_pretrain import Exp_Relation_Pretrain
# from exp.exp_long_term_forecasting_chronos import Exp_Long_Term_Forecast_Chronos
import random
import numpy as np
from utils.tools import HiddenPrints

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='SEMPO')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name, options:[long_term_forecast, long_term_forecast_chronos]')
    parser.add_argument('--is_pretraining', type=int, default=1, help='status')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--is_zeroshot', type=int, default=1, help='status')
    parser.add_argument('--train_test', type=int, default=1, help='train_test')
    parser.add_argument('--model_id', type=str, default='ETTm1', help='model id')
    parser.add_argument('--model', type=str, default='SEMPO', help='model name, options: [SEMPO_CL, SEMPO, Chronos, Moirai, Timer, TimesFM]')
    parser.add_argument('--apply_pywt', action='store_true', help='apply pywt denoise for icecore')
    parser.add_argument('--use_spectral_features', type=int, default=1, help='使用频谱静态特征 (1=开启, 0=使用地理特征)')
    parser.add_argument('--cluster_result_path', type=str, default='cluster_results_cosine_zscore.csv', help='聚类结果文件路径')
    parser.add_argument('--residual_prediction', action='store_true', help='残差预测模式: 模型预测簇均值偏差而非原始值')
    parser.add_argument('--target_anchor_residual', action='store_true', help='crossvar only: predict target residual around support target mean anchor')
    parser.add_argument('--target_anchor_len', type=int, default=16, help='crossvar only: number of support target points used for mean anchor')
    parser.add_argument('--static_dim', type=int, default=3, help='静态特征维度: 3=频谱特征, 4=地理特征')
    parser.add_argument('--use_encoder_proto_static', action='store_true', help='append offline encoder-prototype cosine similarity to static features')
    parser.add_argument('--encoder_proto_path', type=str, default=None, help='offline encoder prototype npz path')
    parser.add_argument('--use_old_cluster_onehot', action='store_true', help='crossvar only: include old hard cluster one-hot in static features')
    parser.add_argument('--use_pretrain_cluster_onehot', action='store_true', help='crossvar only: include relation-pretrain prototype cluster one-hot in static features')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--icecore_task_type', type=str, default=None, choices=['s', 'crossvar', 'ms'], help='icecore dataset task type used by data factory and prototype extraction')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--x_data_path', type=str, default='accum.csv', help='source variable csv file (e.g., accum.csv)')
    parser.add_argument('--y_data_path', type=str, default='merged_chem_clean.csv', help='target variable csv file (e.g., chem.csv)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--percent', type=int, default=10, help='few-shot or full-shot')
    parser.add_argument('--setting', type=str, default='experiment setting', help='randomly initialize an experimental setup')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--use_static_features', action='store_true', help='use static features')
    parser.add_argument('--pred_seperate', type=str, default=None, help='每个cluster单独预测') #弃用
    parser.add_argument('--filter_cid', type=int, default=None, help='每个cluster单独预测')
    parser.add_argument('--support_ratio', type=float, default=None, help='per-site support比例(如0.66), None=固定100年')
    parser.add_argument('--site_split_path', type=str, default=None, help='CSV file defining fixed site-level train/val/test split')


    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--horizon_lengths', type=int, nargs='+', default=[1,96,192,336,720], help='prediction sequence length list')
    # parser.add_argument('--horizon_lengths', type=int, nargs='+', default=[1, 8, 16, 32], help='prediction sequence length list')
    parser.add_argument('--patch_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--stride', type=int, default=64, help='stride between patch')
   
    # model define
    parser.add_argument('--c_in', type=int, default=1, help='input size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')    
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--head_type', default='prediction', type=str, help='head type of different task')
    parser.add_argument('--domain_len', type=int, help='the number of domain', default=128)

    parser.add_argument('--static_emb_dim', type=int, help='static embedding dim', default=8)
    parser.add_argument('--use_static_bias', action='store_true', help='use static bias')
    parser.add_argument('--use_static_concat', action='store_true', help='concat static feat')

    parser.add_argument('--use_static_kv', action='store_true', help='use static feat in K/V')
    parser.add_argument('--meta_prefix_len', type=int, default = 8, help='meta prefix len')

    parser.add_argument('--use_film', action='store_true', help='use FiLM after encoder and before decoder')
    parser.add_argument('--film_hidden_dim', type=int, default=64, help='hidden dim of FiLM MLP')
    parser.add_argument('--film_scale_shift', action='store_true', help='use both gamma and beta in FiLM; otherwise beta only')

    parser.add_argument('--filter_approx', action='store_true', help='filter approx for decomposed_wavelet_learnings')
    parser.add_argument('--use_swt_wavelet', action='store_true', help='使用可导 Haar SWT-like 小波分解替代 DWT')
    parser.add_argument('--use_static_conditioned_wavelet', action='store_true', help='返回每个样本自己的小波阈值')
    parser.add_argument('--use_context_memory', action='store_true', help='use support x/y cross-attention memory for cross-variable query prediction')
    parser.add_argument('--context_memory_gate_init', type=float, default=-2.0, help='initial logit for gated residual memory cross-attention')
    parser.add_argument('--context_memory_tokens', type=int, default=16, help='number of pooled support memory tokens')
    parser.add_argument('--context_memory_dropout', type=float, default=0.1, help='dropout for context memory attention and output')
    parser.add_argument('--use_pair_diff', type=int, default=1, choices=[0, 1], help='include y-x / zy-zx in support-memory and relation-pretrain pair features')
    parser.add_argument('--use_pair_product', type=int, default=1, choices=[0, 1], help='include x*y / zx*zy in support-memory and relation-pretrain pair features')
    parser.add_argument('--is_relation_pretraining', action='store_true', help='run crossvar relation self-supervised pretraining instead of forecasting MAML')
    parser.add_argument('--relation_loss_site_weight', type=float, default=1.0, help='weight for same-site relation contrastive loss')
    parser.add_argument('--relation_loss_aug_weight', type=float, default=0.5, help='weight for same-window augmentation consistency loss')
    parser.add_argument('--relation_loss_xy_weight', type=float, default=0.1, help='weight for same-window x/y alignment loss')
    parser.add_argument('--relation_loss_proto_weight', type=float, default=0.2, help='weight for online prototype assignment loss')
    parser.add_argument('--relation_aug_noise_std', type=float, default=0.05, help='std of Gaussian noise in relation pretrain augmentation')
    parser.add_argument('--relation_aug_mask_ratio', type=float, default=0.1, help='time mask ratio in relation pretrain augmentation')
    parser.add_argument('--relation_pretrain_samples_per_site', type=int, default=8, help='number of sampled window pairs per site per relation-pretrain epoch')
    parser.add_argument('--use_online_prototypes', action='store_true', help='use site memory bank and periodic KMeans in relation pretraining')
    parser.add_argument('--num_prototypes', type=int, default=6, help='number of KMeans prototypes for relation pretraining')
    parser.add_argument('--prototype_warmup_epochs', type=int, default=5, help='epochs before enabling prototype loss')
    parser.add_argument('--prototype_cluster_interval', type=int, default=2, help='KMeans recluster interval after warmup')
    parser.add_argument('--prototype_momentum', type=float, default=0.9, help='EMA momentum for site memory bank')
    parser.add_argument('--prototype_temperature', type=float, default=0.2, help='temperature for prototype assignment logits')
    parser.add_argument('--prototype_output_path', type=str, default=None, help='npz path for relation-pretrain prototype similarity')
    parser.add_argument('--relation_pretrain_checkpoint', type=str, default=None, help='checkpoint from relation pretraining; loads backbone.* weights into SEMPO before MAML')
    parser.add_argument('--load_utsd_checkpoint_for_relation_pretrain', action='store_true', help='initialize relation pretraining from the default UTSD checkpoint when tensor shapes match')
    parser.add_argument('--relation_early_stop_epoch', type=int, default=0, help='early stop relation pretraining after this many epochs without train-loss improvement; 0 disables it')


    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='pretrain epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', choices=['constant_with_warmup'], help='adjust learning rate')
    parser.add_argument('--grad_loss_weight', type=float, default=1, help='add gradient loss')
    parser.add_argument('--use_normalized_loss', type=int, default=1, help='是否使用归一化损失 (1开启, 0关闭)')
    parser.add_argument('--level_loss_weight', type=float, default=1, help='add level loss')
    parser.add_argument('--wavelet_loss_weight', type=float, default=0.0, help='wavelet-domain loss 总权重')
    parser.add_argument('--wavelet_band_weights', type=str, default='0.5,0.2,0.2,0.5', help='A3,D3,D2,D1 四个频段的 loss 权重')
    parser.add_argument('--pearson_loss_weight', type=float, default=0.0, help='Pearson correlation loss 总权重')
    parser.add_argument('--freddf_loss_weight', type=float, default=0.0, help='FreDF frequency-domain loss 总权重')
    parser.add_argument('--soft_dtw_loss_weight', type=float, default=0.0, help='Soft-DTW loss 总权重')
    parser.add_argument('--soft_dtw_gamma', type=float, default=0.1, help='Soft-DTW softmin 平滑系数')
    parser.add_argument('--soft_dtw_band', type=int, default=0, help='Soft-DTW Sakoe-Chiba band 宽度, 0 表示不限制')
    parser.add_argument('--soft_dtw_normalize', type=int, default=1, help='是否使用 Soft-DTW divergence 归一化形式')


    # weight decay
    parser.add_argument('--warmup_steps', type=int, default=0, help='warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.95, help='adam beta2')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

    # FOMAML
    parser.add_argument('--is_maml', type=int, default=1, help='是否开启 FOMAML 训练模式 (1:开启, 0:关闭)')
    parser.add_argument('--inner_steps', type=int, default=2, help='内层循环更新步数 (Support Set 上的 adaptation 步数)')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='内层循环的学习率 alpha')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='外层循环(全局优化)的学习率 beta')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # Set up multi-GPU training
    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64210")
        hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
        rank = int(os.environ.get("RANK", "0"))  # node id
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()  # gpus per node
        args.local_rank = local_rank
        print('ip: {}, port: {}, hosts: {}, rank: {}, local_rank: {}, gpus: {}'.format(ip, port, hosts, rank, local_rank, gpus))
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        print('init_process_group finished')
        torch.cuda.set_device(local_rank)

    if args.is_relation_pretraining:
        Exp = Exp_Relation_Pretrain
    elif args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    # elif args.task_name == 'long_term_forecast_chronos':
        # Exp = Exp_Long_Term_Forecast_Chronos
    else:
        raise ValueError('task name not found')
   
    with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
        print('Args in experiment:')
        # print_args(args)

        if args.is_relation_pretraining:
            ii = 0
            use_pair_diff = int(getattr(args, 'use_pair_diff', 1))
            use_pair_product = int(getattr(args, 'use_pair_product', 1))
            pair_suffix = (
                ''
                if use_pair_diff == 1 and use_pair_product == 1
                else '_pair_d{}_p{}'.format(use_pair_diff, use_pair_product)
            )
            setting = '{}_{}_{}_relation_pretrain_k{}_sl{}_dm{}_el{}{}_{}'.format(
                args.task_name,
                args.model,
                args.data,
                args.num_prototypes,
                args.seq_len,
                args.d_model,
                args.e_layers,
                pair_suffix,
                ii
            )
            args.setting = setting
            exp = Exp(args)
            print('>>>>>>>start relation pretraining : {}>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            raise SystemExit(0)
           
        if args.is_pretraining == 1:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_proto{}_ft{}_sl{}_ll{}_pl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                    args.task_name,
                    args.model,
                    args.data,
                    args.encoder_proto_path,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.patch_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    ii)
                args.setting = setting
                exp = Exp(args)  # set experiments

                print('>>>>>>>start pretraining : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.pretrain(setting)    

        if args.is_training == 1:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_proto{}_ft{}_sl{}_ll{}_pl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                    args.task_name,
                    args.model,
                    args.data,
                    args.encoder_proto_path,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.patch_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    ii)
                args.setting = setting
                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                if args.data == 'UTSD':
                    exp.train(setting)
                else:
                    exp.train(setting, train=1)
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.test(setting)
                torch.cuda.empty_cache()
        else:
            if args.data == 'CI':
                ii = 0
                setting = '{}_{}_{}_proto{}_ft{}_sl{}_ll{}_pl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                    args.task_name,
                    args.model,
                    args.data,
                    args.encoder_proto_path,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.patch_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    ii)
                args.setting = setting
                exp = Exp(args)  # set experiments
               
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, test=1)
                torch.cuda.empty_cache()
