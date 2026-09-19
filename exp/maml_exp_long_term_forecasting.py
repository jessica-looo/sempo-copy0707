from data_provider.icecore_data_factory import data_provider, infer_static_dim
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map
from utils.metrics import (
    a3_corr,
    direction_accuracy,
    lag_aware_corr,
    metric,
    peak_event_metrics,
    pearson_corr,
    smooth_corr,
    trend_corr,
)
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad
import os
import time
import re
import warnings
import json
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers.trainer_pt_utils import get_parameter_names

warnings.filterwarnings('ignore')


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
       
        # multi-resolution scheduling
        H_hat = 0
        self.J = []
        while H_hat < args.pred_len:
            for j in reversed(args.horizon_lengths):
                if H_hat + j <= args.pred_len:
                    H_hat += j
                    self.J.append(j)
                    break
        self.idx = {i for i, h in enumerate(args.horizon_lengths) if h in self.J}

    def _build_model(self):
        if getattr(self.args, 'use_static_features', False):
            self.args.static_dim = infer_static_dim(self.args)
            print(f'Auto inferred static_dim: {self.args.static_dim}')

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args).to(self.device)
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _ms_target_dim(self):
        return int(getattr(self.args, 'target_dim', -1))

    def _select_optimizer(self):
        # if self.args.use_weight_decay:
        #     model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
        #                              weight_decay=self.args.weight_decay)
        # else:
        #     model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
       
        optim_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        model_optim = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=1e-8,
        )
        return model_optim

    def _select_criterion(self):
        if hasattr(self.args, 'loss') and self.args.loss == 'MAE':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def _use_context_memory(self):
        return (
            getattr(self.args, 'use_context_memory', False)
            and getattr(self.args, 'icecore_task_type', None) == 'crossvar'
        )

    def _forward_query_with_optional_memory(
        self,
        params,
        t_qx,
        t_qx_mark,
        t_qy,
        t_qy_mark,
        t_static_q,
        t_sx,
        t_sy,
        t_sx_mark,
        t_sy_mark,
    ):
        if self._use_context_memory():
            return functional_call(
                self.model,
                params,
                (t_qx, t_qx_mark, t_qy, t_qy_mark, t_static_q, t_sx, t_sy, t_sx_mark, t_sy_mark),
            )
        return functional_call(self.model, params, (t_qx, t_qx_mark, t_qy, t_qy_mark, t_static_q))

    def _sample_loss(self, pred, true, loss_type='mae'):
        # pred, true: [B, T, C]
        if loss_type.lower() == 'mae':
            loss = torch.abs(pred - true)
        else:
            loss = (pred - true) ** 2
        return loss.mean(dim=(1, 2))   # [B]
    
    def _normalized_loss(self, pred, true, site_std, loss_type='mse', eps=1e-6):
        # pred, true: [B, T, C]
        # site_std: [B, 1] or [B]
        if site_std.ndim == 1:
            site_std = site_std.view(-1, 1, 1)
        elif site_std.ndim == 2:
            site_std = site_std.unsqueeze(-1)   # [B,1,1]

        err = (pred - true) / (site_std + eps)

        if loss_type.lower() == 'mae':
            return err.abs().mean()
        else:
            return (err ** 2).mean()
    def _volatility_weight(self, y, eps=1e-6):
        # y: [B, T, C]
        dy = y[:, 1:, :] - y[:, :-1, :]              # [B, T-1, C]
        vol = dy.abs().mean(dim=(1, 2))              # [B]

        # 归一化，避免权重过大过小
        vol_norm = vol / (vol.mean().detach() + eps)

        # 线性权重：1 + alpha * normalized_vol
        alpha = getattr(self.args, 'vol_weight_alpha', 0.5)
        w = 1.0 + alpha * vol_norm

        # 可选截断，避免极端样本权重太大
        w_min = getattr(self.args, 'vol_weight_min', 1.0)
        w_max = getattr(self.args, 'vol_weight_max', 3.0)
        w = torch.clamp(w, min=w_min, max=w_max)

        return w   # [B]

    def _level_loss(self, pred, true, loss_type='mse'):
        # pred, true: [B, T, C]
        pred_level = pred.mean(dim=1)   # [B, C]
        true_level = true.mean(dim=1)   # [B, C]

        if loss_type.lower() == 'mae':
            return torch.abs(pred_level - true_level).mean()
        else:
            return ((pred_level - true_level) ** 2).mean()
        
    def _swt_haar_coeffs_for_loss(self, y):
        """
        Wavelet-domain loss 用的小波分解。
        输入:
            y: [B, T, C]
        输出:
            [A3, D3, D2, D1]
            每个 shape 都是 [B, T, C]

        这里用 torch 写一个不下采样的 Haar SWT-like 分解，
        保证可以反向传播。
        """
        y = y.transpose(1, 2)  # [B, C, T]

        approx = y
        details = []

        for level in range(3):
            shift = 2 ** level
            shifted = torch.roll(approx, shifts=-shift, dims=-1)

            new_approx = 0.5 * (approx + shifted)
            detail = 0.5 * (approx - shifted)

            details.append(detail.transpose(1, 2))  # [B, T, C]
            approx = new_approx

        # 返回顺序: A3, D3, D2, D1
        return [approx.transpose(1, 2)] + details[::-1]


    def _wavelet_domain_loss(self, pred, true, loss_type='mse'):
        """
        对预测段 pred 和真实段 true 做小波域约束。
        band weight 顺序:
            A3, D3, D2, D1
        """
        pred_coeffs = self._swt_haar_coeffs_for_loss(pred)
        true_coeffs = self._swt_haar_coeffs_for_loss(true)

        weights = getattr(self.args, 'wavelet_band_weights', '0.5,0.2,0.2,0.5')

        if isinstance(weights, str):
            weights = [float(x) for x in weights.split(',')]

        if len(weights) != 4:
            raise ValueError('--wavelet_band_weights 必须是 4 个数，对应 A3,D3,D2,D1')

        loss = pred.new_tensor(0.0)

        for w, pc, tc in zip(weights, pred_coeffs, true_coeffs):
            if loss_type.lower() == 'mae':
                band_loss = torch.abs(pc - tc).mean()
            else:
                band_loss = ((pc - tc) ** 2).mean()

            loss = loss + float(w) * band_loss

        return loss

    def _pearson_correlation_loss(self, pred, true, eps=1e-6):
        """
        Pearson correlation loss.
        pred/true 支持 [B,T] 或 [B,T,C]，沿 T 维度计算每个样本/变量的相关性。
        """
        if pred.ndim == 2:
            pred = pred.unsqueeze(-1)
            true = true.unsqueeze(-1)

        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        true_centered = true - true.mean(dim=1, keepdim=True)

        numerator = (pred_centered * true_centered).sum(dim=1)
        pred_norm = torch.sqrt((pred_centered ** 2).sum(dim=1) + eps)
        true_norm = torch.sqrt((true_centered ** 2).sum(dim=1) + eps)
        corr = numerator / (pred_norm * true_norm + eps)
        corr = torch.clamp(corr, min=-1.0, max=1.0)

        return 1.0 - corr.mean()

    def _freddf_frequency_loss(self, pred, true):
        """
        FreDF-style frequency-domain loss:
        对预测段和真实段沿时间维做 rFFT，然后比较频谱 L1 距离。
        """
        if pred.ndim == 2:
            pred = pred.unsqueeze(-1)
            true = true.unsqueeze(-1)

        pred_freq = torch.fft.rfft(pred, dim=1, norm='ortho')
        true_freq = torch.fft.rfft(true, dim=1, norm='ortho')

        return torch.abs(pred_freq - true_freq).mean()

    def _soft_dtw_softmin(self, a, b, c, gamma):
        values = torch.stack((a, b, c), dim=0)
        return -gamma * torch.logsumexp(-values / gamma, dim=0)

    def _soft_dtw_value(self, pred, true, gamma=0.1, band=0):
        """
        Batch Soft-DTW.
        pred/true: [B,T] or [B,T,C]. 多变量时每个时间点用 C 维向量距离。
        band > 0 时只允许 |i-j| <= band 的局部错位。
        """
        if pred.ndim == 2:
            pred = pred.unsqueeze(-1)
            true = true.unsqueeze(-1)

        batch_size, seq_len, _ = pred.shape
        dist_mat = ((pred.unsqueeze(2) - true.unsqueeze(1)) ** 2).mean(dim=-1)

        inf = float('inf')
        acc = pred.new_full((batch_size, seq_len + 1, seq_len + 1), inf)
        acc[:, 0, 0] = 0.0

        for i in range(1, seq_len + 1):
            j_start = 1
            j_end = seq_len
            if band > 0:
                j_start = max(1, i - band)
                j_end = min(seq_len, i + band)

            for j in range(j_start, j_end + 1):
                soft_prev = self._soft_dtw_softmin(
                    acc[:, i - 1, j],
                    acc[:, i, j - 1],
                    acc[:, i - 1, j - 1],
                    gamma
                )
                acc[:, i, j] = dist_mat[:, i - 1, j - 1] + soft_prev

        return acc[:, seq_len, seq_len] / seq_len

    def _soft_dtw_loss(self, pred, true):
        gamma = getattr(self.args, 'soft_dtw_gamma', 0.1)
        band = getattr(self.args, 'soft_dtw_band', 0)
        normalize = getattr(self.args, 'soft_dtw_normalize', 1)

        gamma = max(float(gamma), 1e-6)
        band = int(band)

        xy = self._soft_dtw_value(pred, true, gamma=gamma, band=band)

        if normalize:
            xx = self._soft_dtw_value(pred, pred, gamma=gamma, band=band)
            yy = self._soft_dtw_value(true, true, gamma=gamma, band=band)
            xy = xy - 0.5 * (xx + yy)
            xy = torch.clamp(xy, min=0.0)

        return xy.mean()

    def _lag_aware_local_mse_loss(self, pred, true):
        """
        Fast Soft-DTW replacement for small local shifts.

        For each sample, compare pred/true under lags in [-band, band],
        then use a soft-min over lag losses. This keeps the intended
        small temporal tolerance without dynamic programming.
        """
        if pred.ndim == 2:
            pred = pred.unsqueeze(-1)
            true = true.unsqueeze(-1)

        band = int(getattr(self.args, 'soft_dtw_band', 0))
        gamma = max(float(getattr(self.args, 'soft_dtw_gamma', 0.1)), 1e-6)

        lag_losses = []
        seq_len = pred.size(1)

        for lag in range(-band, band + 1):
            if lag < 0:
                pred_aligned = pred[:, :seq_len + lag, :]
                true_aligned = true[:, -lag:, :]
            elif lag > 0:
                pred_aligned = pred[:, lag:, :]
                true_aligned = true[:, :seq_len - lag, :]
            else:
                pred_aligned = pred
                true_aligned = true

            if pred_aligned.size(1) == 0:
                continue

            lag_loss = ((pred_aligned - true_aligned) ** 2).mean(dim=(1, 2))
            lag_losses.append(lag_loss)

        if len(lag_losses) == 1:
            return lag_losses[0].mean()

        lag_losses = torch.stack(lag_losses, dim=1)  # [B, num_lags]
        return (-gamma * torch.logsumexp(-lag_losses / gamma, dim=1)).mean()


    def _forecast_loss(self, pred, true, criterion):
        """
        最终预测 loss:
            原本时域 loss + wavelet-domain loss + Pearson loss + FreDF loss + Soft-DTW loss
        """
        base_loss = criterion(pred, true)

        wavelet_w = getattr(self.args, 'wavelet_loss_weight', 0.0)
        pearson_w = getattr(self.args, 'pearson_loss_weight', 0.0)
        freddf_w = getattr(self.args, 'freddf_loss_weight', 0.0)
        soft_dtw_w = getattr(self.args, 'soft_dtw_loss_weight', 0.0)

        if wavelet_w > 0:
            base_loss = base_loss + wavelet_w * self._wavelet_domain_loss(
                pred,
                true,
                loss_type=getattr(self.args, 'loss', 'mse')
            )

        if pearson_w > 0:
            base_loss = base_loss + pearson_w * self._pearson_correlation_loss(pred, true)

        if freddf_w > 0:
            base_loss = base_loss + freddf_w * self._freddf_frequency_loss(pred, true)

        if soft_dtw_w > 0:
            base_loss = base_loss + soft_dtw_w * self._lag_aware_local_mse_loss(pred, true)

        return base_loss
            
    def _continuous_extreme_loss(self, pred, true, alpha=2.0, power=2):
        """
        连续加权极值 Loss (无硬阈值)
        真实的信号绝对值越大，赋予该样本的 MSE 惩罚权重就越高。
        """
        base_mse = (pred - true) ** 2
        # 权重公式: 1.0 + alpha * (|y| ^ power)
        weight = 1.0 + alpha * (torch.abs(true) ** power)
        return torch.mean(base_mse * weight)
    
    def _compute_recon_loss(self, recons, true_x, f_dim, criterion):
        """修复重构时 Patch 维度重叠导致的 shape 不匹配问题"""
        recons = recons[:, :, f_dim:]
        true_x = true_x[:, :, f_dim:]
        
        # 按照模型内部同样的逻辑计算切片参数
        patch_len, stride = self.args.patch_len, self.args.stride
        patch_num = (max(self.args.seq_len, patch_len) - patch_len) // stride + 1
        s_begin = self.args.seq_len - (patch_len + stride * (patch_num - 1))
        
        # 对真实序列进行切片并 Unfold 展平
        true_x_sliced = true_x[:, s_begin:, :]
        true_patches = true_x_sliced.unfold(dimension=1, size=patch_len, step=stride)
        true_patches = true_patches.transpose(2, 3).reshape(true_x.size(0), -1, true_x.size(-1))
        
        return criterion(recons, true_patches)


    @torch.no_grad()
    def _site_linear_baseline(self, sx, sy, qx):
        # Other forecasting tasks retain their original prediction behavior.
        if getattr(self.args, 'icecore_task_type', None) != 'crossvar':
            return 0.0, 0.0
        # Same-time, single-channel, ordered stride=1 windows only.
        assert self.args.features == "M"
        assert sx.ndim == 3 and sx.shape == sy.shape and sx.shape[-1] == 1
        assert qx.shape[1:] == sx.shape[1:]
        assert sx.size(0) > 0
        assert torch.allclose(sx[:-1, 1:], sx[1:, :-1])
        assert torch.allclose(sy[:-1, 1:], sy[1:, :-1])
        x = torch.cat([sx[0], sx[1:, -1]], dim=0)
        y = torch.cat([sy[0], sy[1:, -1]], dim=0)
        xc, yc = x - x.mean(), y - y.mean()
        a = (xc * yc).sum() / xc.square().sum().clamp_min(1e-12)
        b = y.mean() - a * x.mean()
        return a * sx + b, a * qx + b

    def vali(self, vali_data, vali_loader, criterion, maml_params_dict=None):
        total_loss = []
        # 注意：这里千万不要写 with torch.no_grad(): 
        # 将模型设为 eval，保证 BatchNorm/Dropout 不乱动，但允许 autograd 引擎工作
        self.model.eval() 
        
        inner_steps = getattr(self.args, 'inner_steps', 2)
        inner_lr = getattr(self.args, 'inner_lr', 0.01)

        for i, task_data in enumerate(vali_loader):
            # 解包 MAML 数据
            support_x, support_y, support_x_mark, support_y_mark, \
            query_x, query_y, query_x_mark, query_y_mark, static_feat, query_years, \
            query_x_mean, query_y_mean = [item.to(self.device) for item in task_data]
            
            meta_batch_size = support_x.size(0)

            for task_idx in range(meta_batch_size):
                t_sx = support_x[task_idx]     
                t_sy = support_y[task_idx]
                t_sx_mark = support_x_mark[task_idx]
                t_sy_mark = support_y_mark[task_idx]
                
                t_qx = query_x[task_idx]       
                t_qy = query_y[task_idx]
                t_qx_mark = query_x_mark[task_idx]
                t_qy_mark = query_y_mark[task_idx]
                linear_s, linear_q = self._site_linear_baseline(t_sx, t_sy, t_qx)
                
                t_static = static_feat[task_idx].unsqueeze(0).repeat(t_sx.size(0), 1)
                
                # 1. 复制全局参数 (拿最新的复习提纲)
                torch.set_grad_enabled(True)
                fast_weights = {k: v.clone() for k, v in maml_params_dict.items()}

                # 2. 内层适应 (在 Support 上临时算梯度！必须开启 Grad)
                for step in range(inner_steps):
                    outputs_supp, recons_supp = functional_call(self.model, fast_weights, (t_sx, t_sx_mark, t_sy, t_sy_mark, t_static))
                    if self.args.features =="MS":
                        f_dim = self._ms_target_dim()
                        outputs_supp = torch.cat([outputs_supp[j][:, -h:, f_dim] 
                                                for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                        t_sy_pred = t_sy[:, -self.args.pred_len:, f_dim]
                    else:
                        f_dim = 0
                        outputs_supp = torch.cat([outputs_supp[j][:, -h:, f_dim:] 
                                                for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                        t_sy_pred = t_sy[:, -self.args.pred_len:, f_dim:]
                    outputs_supp = outputs_supp + linear_s


                    # if getattr(self.args, 'residual_prediction', False):
                    #     t_sy_pred = t_sy_pred - support_y_mean[task_idx]

                    recon_loss_supp = self._compute_recon_loss(recons_supp, t_sx, f_dim, criterion)

                    # inner_loss = criterion(outputs_supp, t_sy_pred)+recon_loss_supp
                    # inner_loss = criterion(outputs_supp, t_sy_pred)
                    inner_loss = self._forecast_loss(outputs_supp, t_sy_pred, criterion)
                    grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=False, allow_unused=True)
                    
                    tau_lr_scale = 10000  # 将 1e-5 的梯度放大到 0.1 的更新量级
                    mask_lr_scale = 100.0   # 时序掩码网络通常也需要一点推力
                    
                    fast_weights_updated = {}
                    for (name, param), grad in zip(fast_weights.items(), grads):
                        if grad is not None:
                            # 针对不同模块应用不同的内层学习率
                            if 'wavelet_tau' in name:
                                actual_lr = inner_lr * tau_lr_scale
                            elif 'temporal_mask_net' in name:
                                actual_lr = inner_lr * mask_lr_scale
                            else:
                                actual_lr = inner_lr
                                
                            fast_weights_updated[name] = param - actual_lr * grad
                        else:
                            fast_weights_updated[name] = param.clone()
                            
                    fast_weights = fast_weights_updated

                # 3. 外层测试 (在 Query 上做预测！不需要算梯度了，节省显存)
                torch.set_grad_enabled(False)
                t_static_q = static_feat[task_idx].unsqueeze(0).repeat(t_qx.size(0), 1)
                outputs_query, recons_query = self._forward_query_with_optional_memory(
                    fast_weights,
                    t_qx,
                    t_qx_mark,
                    t_qy,
                    t_qy_mark,
                    t_static_q,
                    t_sx,
                    t_sy,
                    t_sx_mark,
                    t_sy_mark,
                )
                
                if self.args.features =="MS":
                    f_dim = self._ms_target_dim()
                    outputs_query = torch.cat([outputs_query[j][:, -h:, f_dim] 
                                            for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                    t_qy_pred = t_qy[:, -self.args.pred_len:, f_dim]
                else:
                    f_dim = 0
                    outputs_query = torch.cat([outputs_query[j][:, -h:, f_dim:] 
                                            for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                    t_qy_pred = t_qy[:, -self.args.pred_len:, f_dim:]
                outputs_query = outputs_query + linear_q

                # if getattr(self.args, 'residual_prediction', False):
                #     t_qy_pred_res = t_qy_pred - query_y_mean[task_idx]
                #     meta_loss = criterion(outputs_query, t_qy_pred_res)
                # else:
                #     meta_loss = criterion(outputs_query, t_qy_pred)
                # recon_loss_query = self._compute_recon_loss(recons_query, t_qx, f_dim, criterion)
                
                # meta_loss = criterion(outputs_query, t_qy_pred)+recon_loss_query
                # meta_loss = criterion(outputs_query, t_qy_pred)
                meta_loss = self._forecast_loss(outputs_query, t_qy_pred, criterion)
                total_loss.append(meta_loss.item())

        # 恢复全局开启梯度的状态，以免影响下一个 epoch 的 train
        torch.set_grad_enabled(True)
        self.model.train()
        return np.average(total_loss)
   

    def train(self, setting, train=0):
        train_data, train_loader = self._get_data(flag='train')
        if train == 1:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        # if train:
        #     print('Loading pretrained model with smart filtering...')
        #     ckpt_path = '/sharedata/home/cyyang/SEMPO/checkpoints/long_term_forecast_SEMPO_UTSD_ftM_sl512_ll48_pl96_pl64_dm256_nh2_el3_dl3_df128_fc1_ebtimeF_dtTrue_0/checkpoint.pth'
        #     # ckpt_path = '/sharedata/home/cyyang/SEMPO/checkpoints/full_long_term_forecast_SEMPO_icecore_crossvar_ftM_sl32_ll48_pl32_pl16_dm256_nh2_el3_dl3_df128_fc1_ebtimeF_dtTrue_0/checkpoint.pth'
            
        #     if os.path.exists(ckpt_path):
        #         pretrained_dict = torch.load(ckpt_path)
        #         model_dict = self.model.state_dict()
                
        #         filtered_dict = {}
        #         for k, v in pretrained_dict.items():
        #             k_model = k.replace("module.", "") if "module." in k else k

        #             if k_model not in model_dict:
        #                 continue

        #             if v.shape != model_dict[k_model].shape:
        #                 print(f"  [尺寸冲突] 丢弃并随机初始化: {k_model} | 预训练 {v.shape} != 当前 {model_dict[k_model].shape}")
        #                 continue
                        
        #             # if 'MoE' in k:
        #             #     print(f"  [语义冲突] 强制丢弃领域专家: {k}")
        #             #     continue
                        
        #             filtered_dict[k] = v
                        
        #         model_dict.update(filtered_dict)
        #         self.model.load_state_dict(model_dict, strict=False)
        #         print(f'Successfully loaded {len(filtered_dict)} layers from pretrained backbone.')
        #     else:
        #         print('Pretrained model not found. Training from scratch.')

        relation_ckpt_path = getattr(self.args, 'relation_pretrain_checkpoint', None)
        if relation_ckpt_path:
            if os.path.exists(relation_ckpt_path):
                print(f'Loading relation-pretrained backbone: {relation_ckpt_path}')
                relation_dict = torch.load(relation_ckpt_path, map_location=self.device)
                model_dict = self.model.state_dict()
                filtered_dict = {}
                for k, v in relation_dict.items():
                    if k.startswith('backbone.'):
                        k_model = k.replace('backbone.', '', 1)
                    elif k.startswith('module.backbone.'):
                        k_model = k.replace('module.backbone.', '', 1)
                    else:
                        continue

                    if k_model in model_dict and v.shape == model_dict[k_model].shape:
                        filtered_dict[k_model] = v

                model_dict.update(filtered_dict)
                self.model.load_state_dict(model_dict, strict=False)
                print(f'Successfully loaded {len(filtered_dict)} relation-pretrained backbone tensors.')
            else:
                print(f'Relation pretrain checkpoint not found: {relation_ckpt_path}')

        # Initialize the correction once, after loading relation-pretrained weights.
        if (getattr(self.args, 'icecore_task_type', None) == 'crossvar'
                and self.args.train_epochs > 0):
            assert self.model.disable_output_revin_denorm
            with torch.no_grad():
                for head in self.model.pretrain_heads:
                    head.linear.weight.zero_()
                    head.linear.bias.zero_()
                if self.model.use_static_bias:
                    self.model.bias_head.weight.zero_()
                    self.model.bias_head.bias.zero_()
            print('Prediction: support linear baseline + SEMPO correction (zero initialized)')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        config_path = os.path.join(path, 'checkpoint_config.json')
        with open(config_path, 'w') as f:
            json.dump(
                {
                    'format_version': 1,
                    'setting': setting,
                    'args': {k: _json_safe(v) for k, v in vars(self.args).items()},
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f'Checkpoint config saved to: {config_path}')

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 【MAML 核心 1】：获取需要 Meta-Learning 的参数，并设置外层优化器
        if getattr(self.args, 'is_maml', 1) == 1:
            maml_params_dict = self.model.get_maml_parameters()
            meta_optimizer = torch.optim.Adam(maml_params_dict.values(), lr=self.args.meta_lr)
            # 记录内层循环步数和学习率
            inner_steps = getattr(self.args, 'inner_steps', 2)
            inner_lr = getattr(self.args, 'inner_lr', 0.01)
            print(f"MAML 训练引擎启动! Inner Steps: {inner_steps}, Inner LR: {inner_lr}, Meta LR: {self.args.meta_lr}")
        else:
            meta_optimizer = self._select_optimizer()
            
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            # batch_x 此时解包出来的是一个 Task 的完整数据 (包含 Support 和 Query)
            for i, task_data in enumerate(train_loader): # 每个 task_data 有 batch_size 个站点
                iter_count += 1
                meta_optimizer.zero_grad()
                
                # 【MAML 核心 2】：解析 Task 数据 (Tuple unpacking)
                # 形状通常为 [batch_size, n_windows, seq_len, vars]
                # 注意这里的 batch_size 实际上指的是 meta_batch_size (即这次抓了几个站点)
                support_x, support_y, support_x_mark, support_y_mark, \
                query_x, query_y, query_x_mark, query_y_mark, static_feat, query_years, \
                query_x_mean, query_y_mean = [item.to(self.device) for item in task_data]
                # support_x: [batch_size, 每个冰心的样本数, seq_len, c_in=1]

                meta_batch_size = support_x.size(0)
                batch_meta_loss = 0.0

                # 【MAML 核心 3】：遍历 Meta-Batch 中的每一个站点 (Task)
                for task_idx in range(meta_batch_size):
                    # 取出当前站点的所有 Support 和 Query 窗口
                    t_sx = support_x[task_idx]     # [n_supp_win(样本数), seq_len（32）, c_in（1）]
                    t_sy = support_y[task_idx]
                    t_sx_mark = support_x_mark[task_idx]
                    t_sy_mark = support_y_mark[task_idx]
                    
                    t_qx = query_x[task_idx]       # [n_query_win, seq_len, c_in]
                    t_qy = query_y[task_idx]
                    t_qx_mark = query_x_mark[task_idx]
                    t_qy_mark = query_y_mark[task_idx]
                    linear_s, linear_q = self._site_linear_baseline(t_sx, t_sy, t_qx)
                    
                    t_static = static_feat[task_idx].unsqueeze(0).repeat(t_sx.size(0), 1) # 扩展到窗口维度(样本数量)

                    # 1. 初始化临时参数字典 (Functional 方式)
                    # 我们需要把 names 和 params 分开，方便传给 functional_call
                    fast_weights = {k: v.clone() for k, v in maml_params_dict.items()}

                    if epoch == 4 and i == 0 and task_idx == 0:
                        self.model.save_vis_flag = True

                    # if task_idx == 0:
                    #     tau_key = [k for k in fast_weights.keys() if 'wavelet_tau' in k][0]
                    #     init_tau_A3 = fast_weights[tau_key][0].item()
                    #     init_tau_D1 = fast_weights[tau_key][-1].item()
                    #     print(f"\n--- Epoch {epoch+1} Batch {i} Task {task_idx} ---")
                    #     print(f" [Meta-Weights] 初始阈值: A3(低频)={init_tau_A3:.4f}, D1(极高频)={init_tau_D1:.4f}")

                    # 2. 内层循环 (Inner Loop) - 在 Support Set 上快速适应
                    for step in range(inner_steps):
                        # 【避坑 1】：RevIN 隔离。在 forward 里，RevIN 会以当前传入的 t_sx 算均值方差
                        # 确保不要和 t_qx 混在一起过模型
                        
                        # 使用 functional_call 进行前向传播，不会污染 model 里的全局参数
                        outputs_supp, recons_supp = functional_call(self.model, fast_weights, (t_sx, t_sx_mark, t_sy, t_sy_mark, t_static))
                        # 我们做预测任务，取最后几个 step
                        if self.args.features =="MS":
                            f_dim = self._ms_target_dim()
                            outputs_supp = torch.cat([outputs_supp[j][:, -h:, f_dim] 
                                                    for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                            t_sy_pred = t_sy[:, -self.args.pred_len:, f_dim]
                        else:
                            f_dim = 0
                            outputs_supp = torch.cat([outputs_supp[j][:, -h:, f_dim:] 
                                                    for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                            t_sy_pred = t_sy[:, -self.args.pred_len:, f_dim:]
                        outputs_supp = outputs_supp + linear_s
                        # if getattr(self.args, 'residual_prediction', False):
                        #     t_sy_pred = t_sy_pred - support_y_mean[task_idx]
                        recon_loss_supp = self._compute_recon_loss(recons_supp, t_sx, f_dim, criterion)

                        # inner_loss = criterion(outputs_supp, t_sy_pred)+ recon_loss_supp
                        # inner_loss = criterion(outputs_supp, t_sy_pred)
                        inner_loss = self._forecast_loss(outputs_supp, t_sy_pred, criterion)
                        grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=False, allow_unused=True) #   - tau_main / mu_main / tau_res / mu_res
                        dead_tensors = []
                        for (name, param), grad in zip(fast_weights.items(), grads):
                            if grad is None and not name.startswith('memory_'):
                                dead_tensors.append(name)
                        
                        if len(dead_tensors) > 0:
                            print("\n以下参数未参与梯度计算：")
                            for name in dead_tensors:
                                print(f"  - {name}")
                        # =======================================================
                        # ---> [全局梯度排查] 检查所有内层参数的梯度健康状况
                        # =======================================================

                        tau_lr_scale = 10000  # 将 1e-5 的梯度放大到 0.1 的更新量级
                        mask_lr_scale = 100.0   # 时序掩码网络通常也需要一点推力
                        
                        fast_weights_updated = {}
                        for (name, param), grad in zip(fast_weights.items(), grads):
                            if grad is not None:
                                # 针对不同模块应用不同的内层学习率
                                if 'wavelet_tau' in name:
                                    actual_lr = inner_lr * tau_lr_scale
                                elif 'temporal_mask_net' in name:
                                    actual_lr = inner_lr * mask_lr_scale
                                else:
                                    actual_lr = inner_lr
                                    
                                fast_weights_updated[name] = param - actual_lr * grad
                            else:
                                fast_weights_updated[name] = param.clone()
                                
                        fast_weights = fast_weights_updated

                        # if task_idx == 0:
                        #     tau_key = [k for k in fast_weights.keys() if 'wavelet_tau' in k][0]
                        #     # 找到 wavelet_tau 在 values() 列表中的索引
                        #     tau_idx = list(fast_weights.keys()).index(tau_key)
                        #     tau_grad = grads[tau_idx]
                            
                        #     print(f"   [Debug] Inner Step {step} | wavelet_tau 原生梯度: {tau_grad.tolist()}")
                        #     if torch.all(tau_grad == 0):
                        #         print("   [警告] 梯度全为 0! 检查是否陷入 Dead ReLU。")
                        # # =======================================================

                    
                        # if task_idx == 0:
                        #     updated_tau_A3 = fast_weights[tau_key][0].item()
                        #     updated_tau_D1 = fast_weights[tau_key][-1].item()
                        #     print(f" [Fast-Weights] 适应后阈值: A3(低频)={updated_tau_A3:.4f}, D1(极高频)={updated_tau_D1:.4f}")
                        #     print(f" [Delta] 变化量: A3={updated_tau_A3 - init_tau_A3:.4f}, D1={updated_tau_D1 - init_tau_D1:.4f}")

                    # 3. 外层评估 (Outer Loop) - 在 Query Set 上算期末成绩
                    t_static_q = static_feat[task_idx].unsqueeze(0).repeat(t_qx.size(0), 1)
                    
                    # 用适应好的 fast_weights 在未知未来数据上测试
                    outputs_query, recons_query = self._forward_query_with_optional_memory(
                        fast_weights,
                        t_qx,
                        t_qx_mark,
                        t_qy,
                        t_qy_mark,
                        t_static_q,
                        t_sx,
                        t_sy,
                        t_sx_mark,
                        t_sy_mark,
                    )
                    
                    if self.args.features =="MS":
                        f_dim = self._ms_target_dim()
                        outputs_query = torch.cat([outputs_query[j][:, -h:, f_dim] 
                                                for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                        t_qy_pred = t_qy[:, -self.args.pred_len:, f_dim]
                    else:
                        f_dim = 0
                        outputs_query = torch.cat([outputs_query[j][:, -h:, f_dim:] 
                                                for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                        t_qy_pred = t_qy[:, -self.args.pred_len:, f_dim:]
                    outputs_query = outputs_query + linear_q

                    # if getattr(self.args, 'residual_prediction', False):
                    #     t_qy_pred_res = t_qy_pred - query_y_mean[task_idx]
                    #     meta_loss = criterion(outputs_query, t_qy_pred_res)
                    # else:
                    #     meta_loss = criterion(outputs_query, t_qy_pred)
                    recon_loss_query = self._compute_recon_loss(recons_query, t_qx, f_dim, criterion)

                    # meta_loss = criterion(outputs_query, t_qy_pred)+recon_loss_query
                    # meta_loss = criterion(outputs_query, t_qy_pred)
                    meta_loss = self._forecast_loss(outputs_query, t_qy_pred, criterion)
                    batch_meta_loss += meta_loss

                # 【MAML 核心 4】：对全局参数 $\theta$ 求梯度并更新
                # 取平均
                batch_meta_loss = batch_meta_loss / meta_batch_size
                train_loss.append(batch_meta_loss.item())

                # 这里算出来的梯度会直接累加到 maml_params_dict (即 self.model) 的 .grad 中
                batch_meta_loss.backward()
                meta_optimizer.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | Meta-Loss: {2:.7f}".format(i + 1, epoch + 1, batch_meta_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            # --- Validation & Test 逻辑 (需要同步修改) ---
            if train == 1:
                vali_loss = self.vali(vali_data, vali_loader, criterion, maml_params_dict)
                test_loss = self.vali(test_data, test_loader, criterion, maml_params_dict)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(meta_optimizer, epoch + 1, self.args) # 如果有 LrScheduler 注意适配

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, test=0):
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputs = []
        sample_sites = []
        sample_years = []     
        sample_clusters = []
        
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        maml_params_dict = self.model.get_maml_parameters()
        inner_steps = getattr(self.args, 'inner_steps', 2)
        inner_lr = getattr(self.args, 'inner_lr', 0.01)
        criterion = self._select_criterion()

        # 注意这里的 global_task_idx 对应 test_sites 里的站点
        global_task_idx = 0 

        for i, task_data in enumerate(test_loader):
            support_x, support_y, support_x_mark, support_y_mark, \
            query_x, query_y, query_x_mark, query_y_mark, static_feat, query_years, \
            query_x_mean, query_y_mean = [item.to(self.device) for item in task_data]

            meta_batch_size = support_x.size(0)

            for task_idx in range(meta_batch_size):
                site_idx = global_task_idx
                site_name = test_data.current_sites[site_idx]
                global_task_idx += 1

                
                cluster_id = getattr(self.args, 'filter_cid', None)
                if cluster_id is None:
                    cluster_id = test_data.site_cluster_map.get(site_idx, -1)
                    
                t_qy_years = query_years[task_idx].detach().cpu().numpy()
                
                t_sx = support_x[task_idx]     
                t_sy = support_y[task_idx]
                t_sx_mark = support_x_mark[task_idx]
                t_sy_mark = support_y_mark[task_idx]
                
                t_qx = query_x[task_idx]       
                t_qy = query_y[task_idx]
                t_qx_mark = query_x_mark[task_idx]
                t_qy_mark = query_y_mark[task_idx]
                linear_s, linear_q = self._site_linear_baseline(t_sx, t_sy, t_qx)
                
                t_static = static_feat[task_idx].unsqueeze(0).repeat(t_sx.size(0), 1)
                torch.set_grad_enabled(True)
                fast_weights = {k: v.clone() for k, v in maml_params_dict.items()}

                # if task_idx == 0:
                #     tau_key = [k for k in fast_weights.keys() if 'wavelet_tau' in k][0]
                #     test_init_tau_A3 = fast_weights[tau_key][0].item()
                #     test_init_tau_D1 = fast_weights[tau_key][-1].item()

                # 内层微调
                for step in range(inner_steps):
                    outputs_supp, recons_supp = functional_call(self.model, fast_weights, (t_sx, t_sx_mark, t_sy, t_sy_mark, t_static))
                    if self.args.features =="MS":
                        f_dim = self._ms_target_dim()
                        outputs_supp = torch.cat([outputs_supp[j][:, -h:, f_dim] 
                                                for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                        t_sy_pred = t_sy[:, -self.args.pred_len:, f_dim]
                    else:
                        f_dim = 0
                        outputs_supp = torch.cat([outputs_supp[j][:, -h:, f_dim:] 
                                                for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                        t_sy_pred = t_sy[:, -self.args.pred_len:, f_dim:]

                    outputs_supp = outputs_supp + linear_s

                    # if getattr(self.args, 'residual_prediction', False):
                    #         t_sy_pred = t_sy_pred - support_y_mean[task_idx]
                    recon_loss_supp = self._compute_recon_loss(recons_supp, t_sx, f_dim, criterion)
                    # inner_loss = criterion(outputs_supp, t_sy_pred)+recon_loss_supp
                    # inner_loss = criterion(outputs_supp, t_sy_pred)
                    inner_loss = self._forecast_loss(outputs_supp, t_sy_pred, criterion)
                    grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=False, allow_unused=True)

                    tau_lr_scale=10000  # 将 1e-5 的梯度放大到 0.1 的更新量级
                    mask_lr_scale = 100.0   # 时序掩码网络通常也需要一点推力
                    
                    fast_weights_updated = {}
                    for (name, param), grad in zip(fast_weights.items(), grads):
                        if grad is not None:
                            # 针对不同模块应用不同的内层学习率
                            if 'wavelet_tau' in name:
                                actual_lr = inner_lr * tau_lr_scale
                            elif 'temporal_mask_net' in name:
                                actual_lr = inner_lr * mask_lr_scale
                            else:
                                actual_lr = inner_lr
                                
                            fast_weights_updated[name] = param - actual_lr * grad
                        else:
                            fast_weights_updated[name] = param.clone()
                            
                    fast_weights = fast_weights_updated

                    # fast_weights = {
                    #     name: param - inner_lr * grad
                    #     for (name, param), grad in zip(fast_weights.items(), grads)
                    # }
                
                # if i == 0 and task_idx == 0:
                #     test_updated_tau_D1 = fast_weights[tau_key][-1].item()
                #     test_updated_tau_A3 = fast_weights[tau_key][0].item()
                #     print(f"\n>>> Meta-Testing 快速适应情况 (Site: {site_name}) <<<")
                #     print(f"测试集初始 A3 阈值: {test_init_tau_A3:.4f}")
                #     print(f"测试集适应后 A3 阈值: {test_updated_tau_A3:.4f}")
                #     print(f"阈值变化幅度 (Delta): {test_updated_tau_A3 - test_init_tau_A3:.4f}\n")
                #     print(f"测试集初始 D1 阈值: {test_init_tau_D1:.4f}")
                #     print(f"测试集适应后 D1 阈值: {test_updated_tau_D1:.4f}")
                #     print(f"阈值变化幅度 (Delta): {test_updated_tau_D1 - test_init_tau_D1:.4f}\n")

                # 外层预测
                torch.set_grad_enabled(False)
                t_static_q = static_feat[task_idx].unsqueeze(0).repeat(t_qx.size(0), 1)
                outputs_query, _ = self._forward_query_with_optional_memory(
                    fast_weights,
                    t_qx,
                    t_qx_mark,
                    t_qy,
                    t_qy_mark,
                    t_static_q,
                    t_sx,
                    t_sy,
                    t_sx_mark,
                    t_sy_mark,
                )
                if self.args.features =="MS":
                    f_dim = self._ms_target_dim()
                    outputs_query = torch.cat([outputs_query[j][:, -h:, f_dim] 
                                            for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                    t_qy_pred = t_qy[:, -self.args.pred_len:, f_dim]
                else:
                    f_dim = 0
                    outputs_query = torch.cat([outputs_query[j][:, -h:, f_dim:] 
                                            for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                    t_qy_pred = t_qy[:, -self.args.pred_len:, f_dim:]
                outputs_query = outputs_query + linear_q
                if getattr(self.args, 'residual_prediction', False) or getattr(self.args, 'target_anchor_residual', False):
                    if self.args.features == "MS":
                        f_dim = self._ms_target_dim()
                        outputs_query = outputs_query + query_y_mean[task_idx][:, :, f_dim]
                        t_qy_pred = t_qy_pred + query_y_mean[task_idx][:, :, f_dim]
                        t_qx = t_qx + query_x_mean[task_idx]
                    else:
                        outputs_query = outputs_query + query_y_mean[task_idx]
                        t_qy_pred = t_qy_pred + query_y_mean[task_idx]
                        t_qx = t_qx + query_x_mean[task_idx]
                
                # 反归一化 (如果在 getitem 中用了 self.scaler，可能需要单独还原，或者直接存缩放后的值)
                pred = outputs_query.detach().cpu().numpy()
                true = t_qy_pred.detach().cpu().numpy()
                inp = t_qx.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
                inputs.append(inp)
                
                # 记录站点名，方便后期画图 (因为一个站点有几十个 query window，所以 repeat)
                n_windows = pred.shape[0]
                sample_sites.extend([site_name] * n_windows)
                sample_clusters.extend([cluster_id] * n_windows)  # 存入聚类 ID
                sample_years.extend(t_qy_years)

        torch.set_grad_enabled(True)
        
        preds = np.concatenate(preds, axis=0) # 把所有站点的 query window 拼起来
        trues = np.concatenate(trues, axis=0)
        inputs = np.concatenate(inputs, axis=0)
        sample_sites = np.array(sample_sites)
        sample_clusters = np.array(sample_clusters)
        sample_years = np.array(sample_years)
        extra_npz_fields = {}

        if getattr(self.args, 'icecore_task_type', None) == 'crossvar':
            full_x_series = []
            full_y_series = []
            full_years = []
            full_sites = []
            full_support_lens = []
            for site_idx, site_name in enumerate(test_data.current_sites):
                full_sites.append(site_name)
                full_x_series.append(np.asarray(test_data.data_x_list[site_idx], dtype=np.float32))
                full_y_series.append(np.asarray(test_data.data_y_list[site_idx], dtype=np.float32))
                full_years.append(np.asarray(test_data.year_list[site_idx], dtype=np.float32))
                full_support_lens.append(int(test_data.support_len_list[site_idx]))

            extra_npz_fields = {
                'crossvar_full_sites': np.asarray(full_sites),
                'crossvar_full_x': np.asarray(full_x_series, dtype=object),
                'crossvar_full_y': np.asarray(full_y_series, dtype=object),
                'crossvar_full_years': np.asarray(full_years, dtype=object),
                'crossvar_full_support_lens': np.asarray(full_support_lens, dtype=np.int64),
                'crossvar_x_path': np.asarray(getattr(self.args, 'x_data_path', '')),
                'crossvar_y_path': np.asarray(getattr(self.args, 'y_data_path', '')),
            }

        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        corr = pearson_corr(preds, trues)
        trend_corr_value = trend_corr(preds, trues)
        direction_acc = direction_accuracy(preds, trues)
        lag_corr, best_lag = lag_aware_corr(preds, trues, max_lag=2)
        corr_smooth_3 = smooth_corr(preds, trues, window=3)
        a3_corr_value = a3_corr(preds, trues)
        peak_recall, peak_precision, peak_f1 = peak_event_metrics(
            preds,
            trues,
            top_ratio=0.1,
            lag_window=2
        )
        metrics_text = (
            'mse: {:.4f}, mae: {:.4f}\n'
            'corr: {:.4f}\n'
            'corr_smooth_3: {:.4f}\n'
            'a3_corr: {:.4f}\n'
            'trend_corr: {:.4f}\n'
            'direction_acc: {:.1f}%\n'
            'lag_corr@±2: {:.4f}, best_lag: {}\n'
            'PeakRecall@10%_±2: {:.4f}\n'
            'PeakPrecision@10%_±2: {:.4f}\n'
            'PeakF1@10%_±2: {:.4f}'
        ).format(
            mse,
            mae,
            corr,
            corr_smooth_3,
            a3_corr_value,
            trend_corr_value,
            direction_acc * 100,
            lag_corr,
            best_lag,
            peak_recall,
            peak_precision,
            peak_f1
        )
        print(metrics_text)

        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write(metrics_text)
            f.write('\n\n')

        if self.args.cv_dir:
            if self.args.fold_id is None:
                raise ValueError("cv_dir 需要同时指定 fold_id")
            folder_path = os.path.join(
                self.args.cv_dir, f"fold_{self.args.fold_id}"
            ) + "/"
        else:
            folder_path = './results/' + setting + '/'
            
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.savez(folder_path + 'results.npz', 
                 inputs=inputs,
                 preds=preds, 
                 trues=trues, 
                 sites=sample_sites,
                 clusters=sample_clusters, 
                 years=sample_years,
                 mae=mae,
                 mse=mse,
                 rmse=rmse,
                 mape=mape,
                 mspe=mspe,
                 corr=corr,
                 trend_corr=trend_corr_value,
                 direction_acc=direction_acc,
                 lag_corr=lag_corr,
                 best_lag=best_lag,
                 corr_smooth_3=corr_smooth_3,
                 a3_corr=a3_corr_value,
                 peak_recall_10_lag2=peak_recall,
                 peak_precision_10_lag2=peak_precision,
                 peak_f1_10_lag2=peak_f1,
                 **extra_npz_fields)
        print(f"results saved to: {folder_path}results.npz")

        return
