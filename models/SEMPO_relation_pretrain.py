'''
复用 SEMPO encoder
pretrain 阶段解冻 encoder/patch/projection/wavelet/static encoder 等
不走 decoder/head/MAML
relation 表示：
    r = MLP([zx, zy] plus optional [zy-zx, zx*zy])
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SEMPO import Model as SEMPOModel


class Model(nn.Module):
    """
    Relation self-supervised wrapper around SEMPO encoder.
    """

    def __init__(self, configs):
        super().__init__()
        self.backbone = SEMPOModel(configs)
        d_model = configs.d_model
        self.use_pair_diff = bool(getattr(configs, 'use_pair_diff', 1))
        self.use_pair_product = bool(getattr(configs, 'use_pair_product', 1))
        relation_components = 2 + int(self.use_pair_diff) + int(self.use_pair_product)
        self.relation_projector = nn.Sequential(
            nn.LayerNorm(d_model * relation_components),
            nn.Linear(d_model * relation_components, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._set_pretrain_trainable()

    def _set_pretrain_trainable(self):
        for _, param in self.backbone.named_parameters():
            param.requires_grad = False

        trainable_prefixes = (
            'W_pos',
            'projection_x',
            'patch_embed',
            'encoder',
            'wavelet_tau',
            'wavelet_tau_mlp',
            'temporal_mask_net',
            'domain_EnMoE',
            'static_encoder',
            'static_meta_encoder',
            'static_kv_proj_en',
            'revin_layer_x',
        )
        for name, param in self.backbone.named_parameters():
            if name.startswith(trainable_prefixes):
                param.requires_grad = True

    def encode_pair(self, x, y, static=None):
        zx = self.backbone.encode_series(x, x_static=static)
        zy = self.backbone.encode_series(y, x_static=static)
        rel_components = [zx, zy]
        if self.use_pair_diff:
            rel_components.append(zy - zx)
        if self.use_pair_product:
            rel_components.append(zx * zy)
        rel = torch.cat(rel_components, dim=-1)
        rel = self.relation_projector(rel)
        return F.normalize(zx, dim=-1), F.normalize(zy, dim=-1), F.normalize(rel, dim=-1)

    def forward(self, x, y, static=None):
        return self.encode_pair(x, y, static=static)
