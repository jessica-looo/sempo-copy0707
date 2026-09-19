import re, os
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from layers.SEMPO_EncDec import SharedPatchPredictionHead
from layers.SEMPO_EncDec import TSTEncoder, TowerEncoder, MixtrueExpertsLayer, PredictionHead, PretrainHead
from layers.pos_encoding import positional_encoding
import pywt
import ptwt
import numpy as np



class Model(nn.Module):
    """
    Output dimension:
         [bs x target_dim x nvars] for prediction
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, configs):

        super().__init__()

        assert configs.head_type in ['pretrain', 'prediction'], 'head type should be either pretrain or prediction'
        n_heads:int = 16
        head_dropout:float = 0.2
        individual:bool = False
        d_ff:int = 256
        norm:str = 'RMSNorm'
        attn_dropout:float = 0.
        dropout1:float = 0.
        dropout2:float = 0.2
        act:str = "silu"       # or silu
        res_attention:bool = True
        pre_norm:bool = True
        store_attn:bool = False
        pe:str = 'zeros'
        learn_pe:bool = True
        self.freq_num:int = 4
        self.n_vars = configs.c_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.head_type = configs.head_type
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.domain_len = configs.domain_len
        self.horizon_lengths = configs.horizon_lengths
        self.use_static_features = configs.use_static_features
        self.use_static_bias = configs.use_static_bias
        self.use_static_concat = configs.use_static_concat     
        self.use_static_kv = configs.use_static_kv       
        self.meta_prefix_len = configs.meta_prefix_len  
        self.use_film = getattr(configs, 'use_film', False)
        self.film_hidden_dim = getattr(configs, 'film_hidden_dim', 64)
        self.film_scale_shift = getattr(configs, 'film_scale_shift', False)
        self.filter_approx = getattr(configs, 'filter_approx', False)  
        self.use_swt_wavelet = getattr(configs, 'use_swt_wavelet', False) 
        self.freq_weight = nn.Parameter(torch.ones(self.freq_num))  
        self.use_swt_wavelet = getattr(configs, 'use_swt_wavelet', False)
        self.use_static_conditioned_wavelet = getattr(configs, 'use_static_conditioned_wavelet', False)
        self.static_emb_dim = getattr(configs, 'static_emb_dim', False)
        self.features=getattr(configs, 'features', False)
        self.use_context_memory = getattr(configs, 'use_context_memory', False)
        self.context_memory_tokens = getattr(configs, 'context_memory_tokens', 16)
        self.context_memory_dropout_p = getattr(configs, 'context_memory_dropout', 0.1)
        self.context_memory_gate_init = getattr(configs, 'context_memory_gate_init', -2.0)
        self.use_pair_diff = bool(getattr(configs, 'use_pair_diff', 1))
        self.use_pair_product = bool(getattr(configs, 'use_pair_product', 1))
        self.disable_output_revin_denorm = (
            getattr(configs, 'target_anchor_residual', False)
            and getattr(configs, 'icecore_task_type', None) == 'crossvar'
        )

        # Norm
        self.revin_layer_x = RevIN(self.n_vars, affine=True)
       
        # Projection
        self.projection_x = nn.Linear(self.seq_len, self.seq_len)

        # Patching
        self.patch_num = (max(self.seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        tgt_len = self.patch_len  + self.stride * (self.patch_num - 1)
        self.s_begin = self.seq_len - tgt_len
       
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, self.patch_num * self.n_vars, self.d_model)

        # Residual dropout
        self.dropout1 = nn.Dropout(dropout1)
       
        # Input encoding
        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
       
        # Encoder
        self.encoder = TSTEncoder(self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout1, pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                  n_layers=self.e_layers, store_attn=store_attn)
        self.decoder = TowerEncoder(self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout1, pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                n_layers=self.d_layers, store_attn=store_attn)
        self.head = PretrainHead(self.d_model, self.patch_len, head_dropout)

       
        # self.freq_len = self.seq_len // 2 + 1
        # self.theta = nn.Parameter(torch.rand(1))
        # self.tau_main = nn.Parameter(torch.rand(self.freq_num) * (self.seq_len // 2 + 1)) # 截断频率
        # self.mu_main = nn.Parameter(torch.bernoulli(torch.full((self.freq_num, self.freq_len), 0.5))) # 掩码方向指示器
        # self.tau_res = nn.Parameter(torch.rand(self.freq_num) * (self.seq_len // 2 + 1))
        # self.mu_res = nn.Parameter(torch.bernoulli(torch.full((self.freq_num, self.freq_len), 0.5)))

        self.wavelet_name = 'haar'  # 推荐使用 db4 或 haar
        self.decomposition_levels = 3  # 分解层级
        # # 每一个层级（A3, D3, D2, D1）分配一个可学习阈值
        # self.wavelet_tau = nn.Parameter(torch.full((self.decomposition_levels + 1,), -1.0))
        # # self.wavelet_tau = nn.Parameter(torch.zeros(self.decomposition_levels + 1))
        # if self.use_static_conditioned_wavelet:
        #     self.wavelet_tau_mlp = nn.Sequential(
        #         nn.Linear(self.static_emb_dim, 32),
        #         nn.GELU(),
        #         nn.Linear(32, self.decomposition_levels + 1)
        #     )
            
        #     # 稍微控制一下初始化，让它一开始输出的值接近你之前的 -1.0
        #     nn.init.constant_(self.wavelet_tau_mlp[-1].bias, -1.0)
        #     nn.init.zeros_(self.wavelet_tau_mlp[-1].weight)

        # # 用于时域聚焦的轻量级卷积掩码网络
        # self.temporal_mask_net = nn.Sequential(
        #     nn.Conv1d(1, 1, kernel_size=3, padding=1),
        #     nn.ReLU()
        # )

        # nn.init.ones_(self.temporal_mask_net[0].weight)
        # nn.init.constant_(self.temporal_mask_net[0].bias, 1)

        # [修改点 1-C] 增加极简趋势预测头 (绕过 Transformer)
        self.trend_projection = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),  # 升维或保持维度
            nn.GELU(),                              # 非线性激活
            nn.Dropout(0.1),                        # 必须加 Dropout 防止 MAML 过拟合
            nn.Linear(self.d_model, self.pred_len)  # 输出目标长度
        )
        self.save_vis_flag = False
        self.vis_save_dir = './vis_wavelet_results/'
        if not os.path.exists(self.vis_save_dir):
            os.makedirs(self.vis_save_dir)
        
        # load weight
        if configs.data == 'UTSD':
            setting = re.sub(r'_SEMPO_', '_SEMPO_CL_', configs.setting)
            print('loading pretrained encoder-decoder')
            state_dict = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
           
            modules_to_load = {
                "module.W_pos": self.W_pos,
                # "module.theta": self.theta,
                # "module.tau_main": self.tau_main,
                # "module.mu_main": self.mu_main,
                # "module.tau_res": self.tau_res,
                # "module.mu_res": self.mu_res,
                "module.patch_embed": self.patch_embed,
                "module.encoder": self.encoder,
                "module.decoder": self.decoder,
                "module.head": self.head,
            }
            for prefix, target in modules_to_load.items():
                if isinstance(target, torch.nn.Module):
                    module_state_dict = {
                        k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k.startswith(prefix)
                    }
                    target.load_state_dict(module_state_dict, strict=True)
                else:
                    target.data.copy_(state_dict[prefix])
           
        # Frozen
        # for param in [self.W_pos, self.theta, self.tau_main, self.mu_main, self.tau_res, self.mu_res]:
        for param in [self.W_pos]:
            param.requires_grad = False
        for module in [self.patch_embed, self.encoder, self.decoder, self.head]:    
            for param in module.parameters():
                param.requires_grad = False

        # Prefix
        self.domain_EnMoE = MixtrueExpertsLayer(prefix_projection=True, num_virtual_tokens=self.domain_len,
                            token_dim=self.d_model, encoder_hidden_size=self.d_model, n_layers=self.e_layers)
        self.domain_DeMoE = MixtrueExpertsLayer(prefix_projection=True, num_virtual_tokens=self.domain_len,
                            token_dim=self.d_model, encoder_hidden_size=self.d_model, n_layers=self.d_layers)
        self.domain_tokens = torch.arange(self.domain_len).long()
        # prefix dropout
        self.dropout2 = nn.Dropout(dropout2)
       
        # Head    

        # Same-time 32-year experiment: shared patch decoder with overlap averaging.
        assert self.seq_len == self.pred_len == 32
        assert self.patch_len == 16 and self.stride == 8
        assert self.s_begin == 0 and list(self.horizon_lengths) == [32]
        self.pretrain_heads = nn.ModuleList([
            SharedPatchPredictionHead(
                self.d_model, self.patch_len, self.stride, self.pred_len, head_dropout
            )
        ])

        # ===== FiLM: encoder output -> decoder input =====
        self.static_emb_dim = getattr(configs, 'static_emb_dim', 8)
        if self.use_film:
            if not self.use_static_features:
                raise ValueError("use_film=True 时必须同时开启 use_static_features")

            film_out_dim = self.d_model * 2 if self.film_scale_shift else self.d_model
            self.film_mlp = nn.Sequential(
                nn.Linear(self.static_emb_dim, self.film_hidden_dim),
                nn.GELU(),
                nn.Linear(self.film_hidden_dim, film_out_dim)
            )

        # static
        self.static_dim = getattr(configs, 'static_dim', 4) # 自动适配: 3=频谱特征, 4=地理特征

        
        if self.use_static_features:
            self.static_encoder = nn.Linear(self.static_dim, self.static_emb_dim)

            if self.use_static_bias:
                self.bias_head = nn.Linear(self.static_emb_dim, 1)
            if self.use_static_concat:
                self.feature_fusion = nn.Linear(configs.c_in + self.static_emb_dim, configs.c_in)

        if self.use_static_features and self.use_static_kv:
            self.static_meta_encoder = nn.Sequential(
                nn.Linear(self.static_emb_dim, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model)
            )
            self.static_kv_proj_en = nn.Linear(
                self.d_model,
                self.e_layers * 2 * self.meta_prefix_len * self.d_model
            )

            self.static_kv_proj_de = nn.Linear(
                self.d_model,
                self.d_layers * 2 * self.meta_prefix_len * self.d_model
            )

        if self.use_context_memory:
            memory_pair_components = 2 + int(self.use_pair_diff) + int(self.use_pair_product)
            self.memory_pair_proj = nn.Linear(self.n_vars * memory_pair_components, self.d_model)
            self.memory_encoder = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Dropout(self.context_memory_dropout_p),
                nn.Linear(self.d_model, self.d_model),
            )
            self.memory_pos = nn.Parameter(torch.zeros(self.context_memory_tokens, self.d_model))
            self.memory_cross_attn = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=2,
                dropout=self.context_memory_dropout_p,
                batch_first=True,
            )
            self.memory_dropout = nn.Dropout(self.context_memory_dropout_p)
            self.memory_norm = nn.LayerNorm(self.d_model)
            self.memory_gate = nn.Parameter(torch.tensor(float(self.context_memory_gate_init)))


    
 
    def mixture_of_experts(self, x, bs, encode=True):
        if encode:
            past_key_values = self.domain_EnMoE(x, self.domain_tokens.to(x.device))      # [bs' x patch_num * n_vars x 2 * e_layers * d_model]
            past_key_values = past_key_values.reshape(bs * self.freq_num, -1, self.e_layers, 2, self.d_model).permute(2, 3, 0, 1, 4)
        else:
            x = x.permute(0, 1, 3, 2).reshape(bs, -1, self.d_model)
            past_key_values = self.domain_DeMoE(x, self.domain_tokens.to(x.device))   # [bs x patch_num * n_vars x 2 * d_layers * d_model]
            past_key_values = past_key_values.reshape(bs, -1, self.d_layers, 2, self.d_model).permute(2, 3, 0, 1, 4)
        past_key_values = self.dropout2(past_key_values)
        return past_key_values
   
    def adaptive_energy_mask(self, z):
        bs, _, _ = z.shape
        # Calculate energy in the frequency domain
        energy = torch.abs(z).pow(2).sum(dim=-1)

        # Flatten energy across freq_len and n_vars dimensions and then compute median
        # Flattening freq_len and n_vars into a single dimension
        flat_energy = energy.view(bs, -1)
        # Compute median
        median_energy = flat_energy.median(dim=1, keepdim=True)[0] 
        # Reshape to match the original dimensions
        median_energy = median_energy.view(bs, 1)  
        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        energy_mask = ((normalized_energy > self.theta).float() - self.theta).detach() + self.theta
        energy_mask = energy_mask.unsqueeze(-1)
        return energy_mask
   
    def adaptive_frequency_mask(self, z, tau):
        freq_num, bs, freq_len, patch_num = z.shape
        
        # 【关键修正 1】：必须把 freq_indices 转成 float32，否则无法参与 Sigmoid 运算
        freq_indices = torch.arange(freq_len, device=z.device, dtype=torch.float32)\
                            .unsqueeze(0).unsqueeze(2).expand(bs, -1, patch_num)
        
        tau = tau.view(freq_num, 1, 1, 1).expand(-1, bs, freq_len, patch_num)
        half = freq_num // 2
        
        # 温度系数：控制滤波器的“陡峭程度”。
        # temp 越大，越接近硬截断；temp 越小，过渡越平滑。10.0 是一个很好的经验起点。
        temp = 10.0 
        
        # 【关键修正 2】：使用 Sigmoid 构建可导的“软掩码” (Soft Mask)
        # 1. 低通滤波 (前 half 个分支)：当 freq < tau 时，(tau - freq) > 0，Sigmoid 趋近于 1
        mask_low = torch.sigmoid((tau[:half] - freq_indices) * temp)
        
        # 2. 高通滤波 (后 half 个分支)：当 freq > tau 时，(freq - tau) > 0，Sigmoid 趋近于 1
        mask_high = torch.sigmoid((freq_indices - tau[half:]) * temp)
        
        freq_mask = torch.cat([mask_low, mask_high], dim=0)
        
        return freq_mask
   
    def decomposed_frequency_learning(self, x):
        bs, seq_len, n_vars = x.shape                     
        # apply FFT along the time dimension
        z = torch.fft.rfft(x, dim=1, norm='ortho')  # [bs x freq_len x n_vars]
        
        # dominant energy mask
        energy_mask = self.adaptive_energy_mask(z)
        # main energy part
        z_main = z * energy_mask
        # residual energy part
        z_res = z - z_main
        z_res = z_res.unsqueeze(0).expand(self.freq_num, -1, -1, -1)  # [freq_num x bs x freq_len x n_vars]
        z_main = z_main.unsqueeze(0).expand(self.freq_num, -1, -1, -1)  # [freq_num x bs x freq_len x n_vars]
        
        # frequency mask in main energy part
        main_freq_mask = self.adaptive_frequency_mask(z_main, self.tau_main)
        # frequency mask in residual energy part
        res_freq_mask = self.adaptive_frequency_mask(z_res, self.tau_res)
        z = z_main * main_freq_mask + z_res * res_freq_mask  # [freq_num x bs x freq_len x n_vars]
        
        # apply inverse FFT
        x = torch.fft.irfft(z, n=seq_len, dim=2, norm='ortho')  # [freq_num x bs x seq_len x n_vars]
        return x
    
    def _swt_haar_decompose(self, x_in):
        """
        可导 Haar SWT-like 分解。

        输入:
            x_in: [bs*n_vars, 1, seq_len]

        输出:
            [A3, D3, D2, D1]
            每个都是 [bs*n_vars, 1, seq_len]
        """
        approx = x_in
        details = []

        for level in range(self.decomposition_levels):
            shift = 2 ** level
            shifted = torch.roll(approx, shifts=-shift, dims=-1)

            new_approx = 0.5 * (approx + shifted)
            detail = 0.5 * (approx - shifted)

            details.append(detail)
            approx = new_approx

        return [approx] + details[::-1]
    

    def _soft_threshold_sigmoid(self,z, tau, scale, temp=50.0):
        x_norm = (torch.abs(z) - tau) / scale
        mask = torch.sigmoid(x_norm * temp)
        # 在 _soft_threshold_sigmoid 里
        x = x_norm * temp
        # if torch.rand(1).item() < 0.05:
        #     print(f"sigmoid input stats: mean={x.mean():.2f}, std={x.std():.2f}, "
        #         f"饱和比例={(x.abs() > 4).float().mean():.2%}")
        return z * mask
    
    def decomposed_wavelet_learning(self, x, e_site):
        bs, seq_len, n_vars = x.shape
        # 调整维度: [bs * n_vars, 1, seq_len]
        x_in = x.permute(0, 2, 1).reshape(-1, 1, seq_len)
        
        # 1. 离散小波分解 (DWT)
        wavelet = pywt.Wavelet(self.wavelet_name)

        if self.use_swt_wavelet:
            coeffs = self._swt_haar_decompose(x_in)
        else:
            coeffs = ptwt.wavedec(x_in, wavelet, level=self.decomposition_levels)
        
    #    # 2. 振幅甄别 (软阈值截断)
    #     temp = 10.0
    #     coeffs_filtered = []
        
    #     # [根据开关决定 tau 的来源]
    #     if self.use_static_conditioned_wavelet:
    #         # 动态域自适应 tau: [bs, 4]
    #         tau_raw = self.wavelet_tau_mlp(e_site)
    #         # 扩展到 [bs*n_vars, 4]
    #         tau_raw = tau_raw.repeat_interleave(n_vars, dim=0)
    #         tau_pos = torch.nn.functional.softplus(tau_raw)
    #     else:
    #         # 全局统一静态 tau: [4]
    #         tau_static = torch.nn.functional.softplus(self.wavelet_tau)
    #         # 同样扩展到 [bs*n_vars, 4] 以保持下游代码形状一致
    #         tau_pos = tau_static.unsqueeze(0).expand(bs * n_vars, -1)
        
    #     # 处理近似层 (VIP通道开关)
    #     if self.filter_approx:
    #         scale_0 = coeffs[0].detach().abs().mean().clamp(min=1e-6)
    #         tau_approx = tau_pos[:, 0].view(-1, 1, 1) * scale_0
    #         filtered_approx = self._soft_threshold_sigmoid(coeffs[0], tau_approx, scale_0, temp=5.0)
    #         coeffs_filtered.append(filtered_approx)
    #     else:
    #         coeffs_filtered.append(coeffs[0])  # VIP 通道，不截断

    #     # 处理细节层
    #     for i in range(1, len(coeffs)):
    #         scale_i = coeffs[i].detach().abs().mean().clamp(min=1e-6)
    #         tau_detail = tau_pos[:, i].view(-1, 1, 1) * scale_i
    #         filtered_detail = self._soft_threshold_sigmoid(coeffs[i], tau_detail, scale_i, temp=5.0)
    #         coeffs_filtered.append(filtered_detail)
        coeffs_filtered = coeffs
            
        # 3. 多分辨率独立重构 (MRA)
        mra_branches = []
        for i in range(len(coeffs_filtered)):
            if self.use_swt_wavelet:
                branch_recon = coeffs_filtered[i]
            else:
                single_level_coeffs = []
                for j in range(len(coeffs_filtered)):
                    if i == j:
                        single_level_coeffs.append(coeffs_filtered[j])
                    else:
                        single_level_coeffs.append(torch.zeros_like(coeffs_filtered[j]))

                branch_recon = ptwt.waverec(single_level_coeffs, wavelet)

                if branch_recon.shape[-1] > seq_len:
                    branch_recon = branch_recon[..., :seq_len]
                elif branch_recon.shape[-1] < seq_len:
                    branch_recon = torch.nn.functional.pad(
                        branch_recon,
                        (0, seq_len - branch_recon.shape[-1])
                    )
            mra_branches.append(branch_recon)  # append: [bs * n_vars, 1, seq_len]

        # 堆叠成张量: [freq_num, bs * n_vars, 1, seq_len]
        x_mra = torch.stack(mra_branches, dim=0)
        
        # 4. 时序聚焦 (全分支统一处理)
        x_mra_flat = x_mra.view(-1, 1, seq_len)  # 展平以便通过 Conv1d
        # time_mask = self.temporal_mask_net(x_mra_flat)
        # x_mra_focused = x_mra_flat * time_mask
        
        # 5. 恢复 4D 结构以兼容原 SEMPO
        # [freq_num * bs * n_vars, 1, seq_len] -> [freq_num, bs, n_vars, seq_len]
        x_out = x_mra_flat.view(self.freq_num, bs, n_vars, seq_len)
            
        # [freq_num, bs, seq_len, n_vars]
        x_out = x_out.permute(0, 1, 3, 2)

        
        return x_out

    
    def static_to_encoder_prefix(self, e_site, bs):
        """
        e_site: [bs, static_emb_dim]
        return: [e_layers, 2, bs*freq_num, meta_prefix_len, d_model]
        """
        # [bs, d_model]
        z_meta = self.static_meta_encoder(e_site)

        # [bs, e_layers * 2 * Lm * d_model]
        kv = self.static_kv_proj_en(z_meta)

        # [bs, e_layers, 2, Lm, d_model]
        kv = kv.view(bs, self.e_layers, 2, self.meta_prefix_len, self.d_model)

        # [e_layers, 2, bs, Lm, d_model]
        kv = kv.permute(1, 2, 0, 3, 4).contiguous()

        # 复制到 freq_num 个分支
        # [e_layers, 2, bs, 1, Lm, d_model]
        kv = kv.unsqueeze(3)

        # [e_layers, 2, bs, freq_num, Lm, d_model]
        kv = kv.expand(-1, -1, -1, self.freq_num, -1, -1)

        # [e_layers, 2, bs*freq_num, Lm, d_model]
        kv = kv.reshape(self.e_layers, 2, bs * self.freq_num, self.meta_prefix_len, self.d_model)

        kv = self.dropout2(kv)
        return kv
    
    def static_to_decoder_prefix(self, e_site, bs):
        """
        e_site: [bs, static_emb_dim]
        return: [d_layers, 2, bs, meta_prefix_len, d_model]
        """
        z_meta = self.static_meta_encoder(e_site)   # [bs, d_model]

        kv = self.static_kv_proj_de(z_meta)         # [bs, d_layers * 2 * Lm * d_model]
        kv = kv.view(bs, self.d_layers, 2, self.meta_prefix_len, self.d_model)
        kv = kv.permute(1, 2, 0, 3, 4).contiguous() # [d_layers, 2, bs, Lm, d_model]

        kv = self.dropout2(kv)
        return kv
    
    def apply_film(self, x, e_site):
        """
        x: [bs, n_vars, d_model, patch_num]
        e_site: [bs, static_emb_dim]
        """
        if (not self.use_film) or (e_site is None):
            return x

        film_params = self.film_mlp(e_site)  # [bs, d_model] or [bs, 2*d_model]

        if self.film_scale_shift:
            gamma, beta = torch.chunk(film_params, 2, dim=-1)   # [bs, d_model], [bs, d_model]
            gamma = gamma.unsqueeze(1).unsqueeze(-1)            # [bs, 1, d_model, 1]
            beta  = beta.unsqueeze(1).unsqueeze(-1)             # [bs, 1, d_model, 1]
            x = x * (1.0 + gamma) + beta
        else:
            beta = film_params.unsqueeze(1).unsqueeze(-1)       # [bs, 1, d_model, 1]
            x = x + beta

        return x

    def build_context_memory(self, memory_x, memory_y):
        """
        Build station-level memory tokens from support x/y.

        memory_x/memory_y: [n_support, seq_len, n_vars]
        return: [context_memory_tokens, d_model]
        """
        if (not self.use_context_memory) or memory_x is None or memory_y is None:
            return None

        seq_len = min(memory_x.shape[1], memory_y.shape[1])
        x = memory_x[:, :seq_len, :]
        y = memory_y[:, :seq_len, :]

        if x.shape[-1] != self.n_vars:
            x = x[..., :self.n_vars]
        if y.shape[-1] != self.n_vars:
            y = y[..., :self.n_vars]

        pair_components = [x, y]
        if self.use_pair_diff:
            pair_components.append(y - x)
        if self.use_pair_product:
            pair_components.append(x * y)
        pair = torch.cat(pair_components, dim=-1)
        tokens = self.memory_pair_proj(pair.reshape(-1, pair.shape[-1]))  # [n_support*seq_len, d_model]
        tokens = tokens + self.memory_encoder(tokens)
        tokens = tokens.transpose(0, 1).unsqueeze(0)  # [1, d_model, L]
        tokens = F.adaptive_avg_pool1d(tokens, self.context_memory_tokens)
        tokens = tokens.squeeze(0).transpose(0, 1).contiguous()  # [M, d_model]
        tokens = tokens + self.memory_pos
        return tokens

    def apply_context_memory(self, x, memory_x=None, memory_y=None):
        """
        Cross-attend query encoder tokens to support x/y memory.

        x: [bs, n_vars, d_model, patch_num]
        """
        if (not self.use_context_memory) or memory_x is None or memory_y is None:
            return x

        bs, n_vars, d_model, patch_num = x.shape
        memory_tokens = self.build_context_memory(memory_x, memory_y)
        if memory_tokens is None:
            return x

        query_tokens = x.permute(0, 1, 3, 2).reshape(bs, n_vars * patch_num, d_model)
        memory_tokens = memory_tokens.unsqueeze(0).expand(bs, -1, -1)
        memory_out, _ = self.memory_cross_attn(query_tokens, memory_tokens, memory_tokens)
        gate = torch.sigmoid(self.memory_gate)
        query_tokens = self.memory_norm(query_tokens + gate * self.memory_dropout(memory_out))
        x = query_tokens.reshape(bs, n_vars, patch_num, d_model).permute(0, 1, 3, 2).contiguous()
        return x

    def encode_series(self, x_enc, x_static=None):
        """
        Encoder-only representation for relation pretraining.

        Return:
            [bs, d_model] pooled representation.
        """
        if self.use_static_features and x_static is not None:
            e_site = self.static_encoder(x_static)
        else:
            e_site = None

        # x = self.revin_layer_x(x_enc, 'norm')
        x = x_enc

        if self.use_static_concat and e_site is not None:
            e_site_repeat = e_site.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat([x, e_site_repeat], dim=-1)
            x = self.feature_fusion(x)

        x = self.projection_x(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.decomposed_wavelet_learning(x, e_site=e_site)
        x = x[:, :, self.s_begin:, :]
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        freq_num, bs, patch_num, n_vars, patch_len = x.shape
        x = x.reshape(-1, patch_num, n_vars, patch_len)
        x = self.patch_embed(x)

        x = x.transpose(1, 2)
        u = torch.reshape(x, (-1, n_vars * patch_num, self.d_model))
        u = self.dropout1(u + self.W_pos)

        u_d = self.mixture_of_experts(u, bs=bs, encode=True)
        if self.use_static_kv and e_site is not None:
            u_d_static = self.static_to_encoder_prefix(e_site, bs)
            u_d = torch.cat([u_d, u_d_static], dim=3)

        x = self.encoder(u, u_d)
        x = x.reshape(-1, n_vars, patch_num, self.d_model)
        x = x.permute(0, 1, 3, 2)
        x = x.view(-1, bs, n_vars, self.d_model, patch_num).mean(dim=0)
        return x.mean(dim=(1, 3))
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_static=None,
                memory_x=None, memory_y=None, memory_x_mark=None, memory_y_mark=None):
        if self.use_static_features and x_static is not None:
            e_site = self.static_encoder(x_static)
        else:
            e_site = None

        if self.use_static_bias and e_site is not None:
            b_site = self.bias_head(e_site).unsqueeze(1)
        else:
            b_site = 0
        # norm  
        # x = self.revin_layer_x(x_enc, 'norm')        # [bs x seq_len x n_vars]
        x = x_enc

        if self.use_static_concat and e_site is not None:
            e_site_repeat = e_site.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat([x, e_site_repeat], dim=-1)
            x = self.feature_fusion(x)

        # projection 
        x = self.projection_x(x.permute(0, 2, 1)).permute(0, 2, 1)
        # import pdb;pdb.set_trace()
        # decomposed frequency learning
        x_wavelet= self.decomposed_wavelet_learning(x,e_site=e_site)    
        x = x_wavelet

        # do patching    
        x = x[:, :, self.s_begin:, :] 
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # [freq_num x bs x patch_num x n_vars x patch_len]
        freq_num, bs, patch_num, n_vars, patch_len = x.shape
        x = x.reshape(-1, patch_num, n_vars, patch_len)

        # patch embedding
        x = self.patch_embed(x)    # [bs * freq_num x patch_num x n_vars x d_model]   
        

        # # [修改点 3-A] 双流预测入口
        # x_wavelet, mutation_mask = self.decomposed_wavelet_learning(x)    
        
        # # === 趋势流 (Trend Stream) ===
        # # 极简线性层预测未来 A3 趋势：[bs, seq_len, n_vars] -> [bs, pred_len, n_vars]
        # # y_trend = self.trend_projection(x_trend.permute(0, 2, 1)).permute(0, 2, 1)
        # y_trend = x_trend[:, -1:, :].repeat(1, self.pred_len, 1)

        # # === 细节流 (Detail Stream) ===
        # x = x_detail # 只有高频进入 Transformer

        # # do patching    
        # x = x[:, :, self.s_begin:, :] 
        # x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  
        # freq_num, bs, patch_num, n_vars, patch_len = x.shape
        # x = x.reshape(-1, patch_num, n_vars, patch_len)

        # # 对突变掩码做同样的 Unfold 对齐切片
        # mut = mutation_mask[:, :, self.s_begin:, :]
        # mut = mut.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # mut = mut.reshape(-1, patch_num, n_vars, patch_len)

        # # [神来之笔] Point-wise 突变先验注入
        # x_enhanced = x * (1.0 + mut)

        # # patch embedding
        # x = self.patch_embed(x_enhanced)



        # pos embedding
        x = x.transpose(1, 2)  # [bs * freq_num x n_vars x patch_num x d_model]        
        u = torch.reshape(x, (-1, n_vars * patch_num, self.d_model))  # [bs' x n_vars * patch_num x d_model]
        u = self.dropout1(u + self.W_pos)  # [bs' x n_vars * patch_num x d_model] 

        # domain prefix, concat in K and V
        u_d = self.mixture_of_experts(u, bs=bs, encode=True)  # [e_layers x 2 x bs' x patch_num * n_vars x d_model]

        if self.use_static_kv and e_site is not None:
            u_d_static = self.static_to_encoder_prefix(e_site, bs)
            # 在 prefix 长度维拼接
            u_d = torch.cat([u_d, u_d_static], dim=3)

        # encoder
        x = self.encoder(u, u_d)  # [bs' x d_model x n_vars * patch_num]
        x = x.reshape(-1, n_vars, patch_num, self.d_model)  # [bs' x n_vars x patch_num x d_model]
        x = x.permute(0, 1, 3, 2)  # [bs' x n_vars x d_model x patch_num]                                                      
        
        # multi-scale frequency aggregation
        x = x.view(-1, bs, n_vars, self.d_model, patch_num).mean(dim=0)  # [bs x n_vars x d_model x patch_num]
        # x = x.view(self.freq_num, bs, n_vars, self.d_model, patch_num)
        # # 2. 对权重进行 Softmax 归一化，保证总和为 1
        # # fw 维度变为 [freq_num]
        # fw = torch.softmax(self.freq_weight, dim=0)
        # # 3. 维度对齐，利用广播机制相乘 [freq_num, 1, 1, 1, 1]
        # fw = fw.view(-1, 1, 1, 1, 1)
        # # 4. 加权并沿 freq_num 维度求和 -> [bs, n_vars, d_model, patch_num]
        # x = torch.sum(x * fw, dim=0)

        x = self.apply_context_memory(x, memory_x=memory_x, memory_y=memory_y)
          
        # domain prefix, concat in K and V
        x_d = self.mixture_of_experts(x, bs=bs, encode=False)  # [d_layers x 2 x bs x patch_num * n_vars x d_model]

        if self.use_static_kv and e_site is not None:
            x_d_static = self.static_to_decoder_prefix(e_site, bs)
            x_d = torch.cat([x_d, x_d_static], dim=3)

        x = self.apply_film(x, e_site)

        # decoder
        x = self.decoder(x, x_d)  # [bs x n_vars x d_model x patch_num]                                                 
        
        # head
        if self.disable_output_revin_denorm:
            y = [head(x) for head in self.pretrain_heads]
        else:
            y = [self.revin_layer_x(head(x), 'denorm') for head in self.pretrain_heads]
        # y = [head(x) for head in self.pretrain_heads]
        if isinstance(y, list):
            y = [yi + b_site for yi in y]
        else:
            y = y + b_site
        x = self.head(x)
        x = x.reshape(bs, patch_num * patch_len, n_vars)
        # x = self.revin_layer_x(x, 'denorm')
        return y,  x

        # # [修改点 3-B] 预测头合并 (细节预测 + 趋势预测)
        # y_detail = [head(x) for head in self.pretrain_heads]
        
        # if isinstance(y_detail, list):
        #     # 将主干预测的细节，加上刚算出的 y_trend，再做反归一化
        #     y = [self.revin_layer_x(yd + y_trend + b_site, 'denorm') for yd in y_detail]
        # else:
        #     y = self.revin_layer_x(y_detail + y_trend + b_site, 'denorm')
            
        # x = self.head(x)
        # x = x.reshape(bs, patch_num * patch_len, n_vars)
        # x = self.revin_layer_x(x, 'denorm')
        # return y, x

       
    def get_maml_parameters(self):
        """
        专门为 FOMAML 准备：
        1. 冻结未选中的参数；当前 encoder、decoder 和前缀模块参与更新。
        2. 返回需要进行 Meta-Learning 更新的参数字典。
        """
        # 1. 首先，暴力的全局冻结
        for name, param in self.named_parameters():
            param.requires_grad = False
            
        # 2. 定义哪些模块需要被“复活”（参与 FOMAML 更新）
        maml_module_names = [
            'pretrain_heads',          # 预测头：重对齐最终尺度
            'domain_EnMoE.gate',       # 编码器门控：学会选专家
            'domain_EnMoE.embedding', 
            'domain_EnMoE.transform', 
            'domain_DeMoE.gate',       # 解码器门控
            'domain_DeMoE.embedding', 
            'domain_DeMoE.transform', 
            'static_encoder',          # 12维特征处理器：学会理解新环境提示词
            'static_meta_encoder',
            'static_kv_proj_en',
            'static_kv_proj_de',
            'patch_embed',             # 局部块嵌入
            'projection_x',
            'W_pos',
            'temporal_mask_net',
        ]
        maml_module_names.extend(['encoder.', 'decoder.'])
        # # [动态追加 MAML 学习模块]
        # if getattr(self, 'use_static_conditioned_wavelet', False):
        #     maml_module_names.append('wavelet_tau_mlp')
        # else:
        #     maml_module_names.append('wavelet_tau')

        if getattr(self, 'use_context_memory', False):
            maml_module_names.extend([
                'memory_pair_proj',
                'memory_encoder',
                'memory_pos',
                'memory_cross_attn',
                'memory_norm',
                'memory_gate',
            ])
        
        # 3. 针对 RevIN 的仿射参数（如果开启了 affine=True）
        if hasattr(self, 'revin_layer_x') and self.revin_layer_x.affine:
            self.revin_layer_x.affine_weight.requires_grad = True
            self.revin_layer_x.affine_bias.requires_grad = True
            
        # 4. 解冻我们指定的模块
        for name, param in self.named_parameters():
            for maml_module in maml_module_names:
                if maml_module in name:
                    param.requires_grad = True
                    break
        
        # # 5. 再次严查防漏：强行冻结预训练的 MoE 专家网络
        # if hasattr(self, 'domain_EnMoE'):
        #     for param in self.domain_EnMoE.transform.parameters():
        #         param.requires_grad = False
        # if hasattr(self, 'domain_DeMoE'):
        #     for param in self.domain_DeMoE.transform.parameters():
        #         param.requires_grad = False
                
        # 6. 打包返回 requires_grad=True 的参数（用于传入 MAML 优化器）
        maml_params = {name: param for name, param in self.named_parameters() if param.requires_grad}
                    
        return maml_params
