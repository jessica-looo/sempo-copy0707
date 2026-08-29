#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/CrossVar" ]; then mkdir ./logs/CrossVar; fi

# 基础配置
data_name=icecore
model_name=SEMPO
seq_len=32
patch_len=16
stride=8
num_prototypes=3

# Pair-interaction ablation for support memory and relation pretraining.
# Defaults keep the previous full feature set: [x, y, y-x, x*y].
# Run examples:
#   use_pair_diff=0 use_pair_product=0 bash scripts/time_series_forecatsing/few_shot/sempo_icecore_crossvar_d18O_accum.sh
#   use_pair_diff=1 use_pair_product=0 bash scripts/time_series_forecatsing/few_shot/sempo_icecore_crossvar_d18O_accum.sh
#   use_pair_diff=0 use_pair_product=1 bash scripts/time_series_forecatsing/few_shot/sempo_icecore_crossvar_d18O_accum.sh
use_pair_diff=${use_pair_diff:-0}
use_pair_product=${use_pair_product:-1}
if [ "$use_pair_diff" = "1" ] && [ "$use_pair_product" = "1" ]; then
  pair_suffix=""
else
  pair_suffix="_pair_d${use_pair_diff}_p${use_pair_product}"
fi
relation_pretrain_setting=long_term_forecast_SEMPO_d18O_accum_crossvar_relation_pretrain_k${num_prototypes}_sl${seq_len}_dm256_el3${pair_suffix}_0
split_path="./dataset/icecore/splits.csv"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29503 run.py \
  --task_name long_term_forecast \
  --is_pretraining 0 \
  --is_training 1 \
  --is_zeroshot 0 \
  --root_path ./dataset/$data_name/ \
  --model_id ${data_name}_${seq_len} \
  --model $model_name \
  --data d18O_accum_crossvar \
  --icecore_task_type crossvar \
  --apply_pywt \
  --features M \
  --seq_len $seq_len \
  --pred_len $seq_len \
  --patch_len $patch_len \
  --stride $stride \
  --horizon_lengths $seq_len \
  --c_in 1 \
  --e_layers 3 \
  --d_layers 3 \
  --percent 100 \
  --train_epochs 100 \
  --pretrain_epochs 10 \
  --batch_size 6 \
  --des 'Exp' \
  --domain_len 128 \
  --d_model 256 \
  --learning_rate 1e-3 \
  --loss MSE \
  --warmup_steps 1000 \
  --lradj constant_with_warmup \
  --head_type prediction \
  --num_workers 10 \
  --patience 6 \
  --use_static_features \
  --use_static_kv \
  --static_emb_dim 64 \
  --static_dim 14 \
  --cluster_result_path "cluster/accum/accum_100y_cluster_mapped.csv" \
  --is_maml 1 \
  --inner_steps 2 \
  --inner_lr 5e-3 \
  --meta_lr 5e-5 \
  --filter_approx \
  --wavelet_loss_weight 0.1 \
  --wavelet_band_weights "0.5,0.2,0.2,0.5" \
  --pearson_loss_weight 1 \
  --freddf_loss_weight 0.1 \
  --soft_dtw_loss_weight 0.05 \
  --soft_dtw_gamma 0.1 \
  --soft_dtw_band 2 \
  --soft_dtw_normalize 1 \
  --support_ratio 0.66 \
  --x_data_path "d18O/d18O_clean.csv" \
  --y_data_path "accum/accum_100y.csv" \
  --site_split_path "$split_path" \
  --target_anchor_residual \
  --target_anchor_len 16 \
  --use_context_memory \
  --context_memory_gate_init -2.0 \
  --context_memory_tokens 16 \
  --context_memory_dropout 0.1 \
  --use_pair_diff ${use_pair_diff} \
  --use_pair_product ${use_pair_product} \
  --relation_pretrain_checkpoint checkpoints/${relation_pretrain_setting}/checkpoint.pth \
  --use_encoder_proto_static \
  --use_old_cluster_onehot \
  --use_pair_diff 0 \
  --use_pair_product 1 \
  --encoder_proto_path prototypes/d18O_accum_crossvar_relation_pretrain_proto_k${num_prototypes}${pair_suffix}.npz
