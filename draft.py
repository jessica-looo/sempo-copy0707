DistributedDataParallel_icecore(
  (module): Model(
    (revin_layer_x): RevIN()
    (projection_x): Linear(in_features=64, out_features=64, bias=True) # drop
    (dropout1): Dropout(p=0.0, inplace=False)
    (patch_embed): Linear(in_features=16, out_features=256, bias=True) # drop
    (encoder): TSTEncoder(
      (layers): ModuleList(
        (0-2): 3 x TSTEncoderLayer(
          (self_attn): MultiheadAttention(
            (W_Q): Linear(in_features=256, out_features=256, bias=True)
            (W_K): Linear(in_features=256, out_features=256, bias=True)
            (W_V): Linear(in_features=256, out_features=256, bias=True)
            (sdp_attn): ScaledDotProductAttention(
              (attn_dropout): Dropout(p=0.0, inplace=False)
            )
            (to_out): Sequential(
              (0): Linear(in_features=256, out_features=256, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (dropout_attn): Dropout(p=0.0, inplace=False)
          (norm_attn): RMSNorm()
          (ff): Sequential(
            (0): Linear(in_features=256, out_features=256, bias=True)
            (1): SiLU()
            (2): Dropout(p=0.0, inplace=False)
            (3): Linear(in_features=256, out_features=256, bias=True)
          )
          (dropout_ffn): Dropout(p=0.0, inplace=False)
          (norm_ffn): RMSNorm()
        )
      )
    )
    (decoder): TowerEncoder(
      (encoder): TSTEncoder(
        (layers): ModuleList(
          (0-2): 3 x TSTEncoderLayer(
            (self_attn): MultiheadAttention(
              (W_Q): Linear(in_features=256, out_features=256, bias=True)
              (W_K): Linear(in_features=256, out_features=256, bias=True)
              (W_V): Linear(in_features=256, out_features=256, bias=True)
              (sdp_attn): ScaledDotProductAttention(
                (attn_dropout): Dropout(p=0.0, inplace=False)
              )
              (to_out): Sequential(
                (0): Linear(in_features=256, out_features=256, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (dropout_attn): Dropout(p=0.0, inplace=False)
            (norm_attn): RMSNorm()
            (ff): Sequential(
              (0): Linear(in_features=256, out_features=256, bias=True)
              (1): SiLU()
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=256, out_features=256, bias=True)
            )
            (dropout_ffn): Dropout(p=0.0, inplace=False)
            (norm_ffn): RMSNorm()
          )
        )
      )
    )
    (head): PretrainHead(
      (dropout): Dropout(p=0.2, inplace=False)
      (linear): Linear(in_features=256, out_features=16, bias=True)
    )
    (domain_EnMoE): MixtrueExpertsLayer(
      (gate): Linear(in_features=256, out_features=128, bias=True)
      (embedding): Embedding(128, 256)
      (transform): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): Tanh()
        (2): Linear(in_features=256, out_features=1536, bias=True)
      )
    )
    (domain_DeMoE): MixtrueExpertsLayer(
      (gate): Linear(in_features=256, out_features=128, bias=True)
      (embedding): Embedding(128, 256)
      (transform): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): Tanh()
        (2): Linear(in_features=256, out_features=1536, bias=True)
      )
    )
    (dropout2): Dropout(p=0.2, inplace=False)
    (pretrain_heads): ModuleList(
      (0): PredictionHead(
        (flatten): Flatten(start_dim=-2, end_dim=-1)
        (linear): Linear(in_features=1024, out_features=1, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (1): PredictionHead(
        (flatten): Flatten(start_dim=-2, end_dim=-1)
        (linear): Linear(in_features=1024, out_features=8, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (2): PredictionHead(
        (flatten): Flatten(start_dim=-2, end_dim=-1)
        (linear): Linear(in_features=1024, out_features=16, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (3): PredictionHead(
        (flatten): Flatten(start_dim=-2, end_dim=-1)
        (linear): Linear(in_features=1024, out_features=32, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (4): PredictionHead(
        (flatten): Flatten(start_dim=-2, end_dim=-1)
        (linear): Linear(in_features=1024, out_features=64, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
      )
    )
  )
)

DistributedDataParallel_utsd(
  (module): Model(
    (revin_layer_x): RevIN()
    (projection_x): Linear(in_features=512, out_features=512, bias=True)
    (dropout1): Dropout(p=0.0, inplace=False)
    (patch_embed): Linear(in_features=64, out_features=256, bias=True)
    (encoder): TSTEncoder(
      (layers): ModuleList(
        (0-2): 3 x TSTEncoderLayer(
          (self_attn): MultiheadAttention(
            (W_Q): Linear(in_features=256, out_features=256, bias=True)
            (W_K): Linear(in_features=256, out_features=256, bias=True)
            (W_V): Linear(in_features=256, out_features=256, bias=True)
            (sdp_attn): ScaledDotProductAttention(
              (attn_dropout): Dropout(p=0.0, inplace=False)
            )
            (to_out): Sequential(
              (0): Linear(in_features=256, out_features=256, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (dropout_attn): Dropout(p=0.0, inplace=False)
          (norm_attn): RMSNorm()
          (ff): Sequential(
            (0): Linear(in_features=256, out_features=256, bias=True)
            (1): SiLU()
            (2): Dropout(p=0.0, inplace=False)
            (3): Linear(in_features=256, out_features=256, bias=True)
          )
          (dropout_ffn): Dropout(p=0.0, inplace=False)
          (norm_ffn): RMSNorm()
        )
      )
    )
    (decoder): TowerEncoder(
      (encoder): TSTEncoder(
        (layers): ModuleList(
          (0-2): 3 x TSTEncoderLayer(
            (self_attn): MultiheadAttention(
              (W_Q): Linear(in_features=256, out_features=256, bias=True)
              (W_K): Linear(in_features=256, out_features=256, bias=True)
              (W_V): Linear(in_features=256, out_features=256, bias=True)
              (sdp_attn): ScaledDotProductAttention(
                (attn_dropout): Dropout(p=0.0, inplace=False)
              )
              (to_out): Sequential(
                (0): Linear(in_features=256, out_features=256, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (dropout_attn): Dropout(p=0.0, inplace=False)
            (norm_attn): RMSNorm()
            (ff): Sequential(
              (0): Linear(in_features=256, out_features=256, bias=True)
              (1): SiLU()
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=256, out_features=256, bias=True)
            )
            (dropout_ffn): Dropout(p=0.0, inplace=False)
            (norm_ffn): RMSNorm()
          )
        )
      )
    )
    (head): PretrainHead(
      (dropout): Dropout(p=0.2, inplace=False)
      (linear): Linear(in_features=256, out_features=64, bias=True)
    )
    (domain_EnMoE): MixtrueExpertsLayer(
      (gate): Linear(in_features=256, out_features=128, bias=True)
      (embedding): Embedding(128, 256)
      (transform): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): Tanh()
        (2): Linear(in_features=256, out_features=1536, bias=True)
      )
    )
    (domain_DeMoE): MixtrueExpertsLayer(
      (gate): Linear(in_features=256, out_features=128, bias=True)
      (embedding): Embedding(128, 256)
      (transform): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): Tanh()
        (2): Linear(in_features=256, out_features=1536, bias=True)
      )
    )
    (dropout2): Dropout(p=0.2, inplace=False)
    (pretrain_heads): ModuleList(
      (0): PredictionHead(
        (flatten): Flatten(start_dim=-2, end_dim=-1)
        (linear): Linear(in_features=2048, out_features=1, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (1): PredictionHead(
        (flatten): Flatten(start_dim=-2, end_dim=-1)
        (linear): Linear(in_features=2048, out_features=96, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (2): PredictionHead(
        (flatten): Flatten(start_dim=-2, end_dim=-1)
        (linear): Linear(in_features=2048, out_features=192, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (3): PredictionHead(
        (flatten): Flatten(start_dim=-2, end_dim=-1)
        (linear): Linear(in_features=2048, out_features=336, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
      )
      (4): PredictionHead(
        (flatten): Flatten(start_dim=-2, end_dim=-1)
        (linear): Linear(in_features=2048, out_features=720, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
      )
    )
  )
)
