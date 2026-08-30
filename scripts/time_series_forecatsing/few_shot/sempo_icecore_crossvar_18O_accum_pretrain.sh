#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/CrossVar" ]; then mkdir ./logs/CrossVar; fi
if [ ! -d "./prototypes" ]; then mkdir ./prototypes; fi

data_name=icecore
model_name=SEMPO
seq_len=32
patch_len=16
stride=8
num_prototypes=3

# Pair-interaction ablation for relation representation.
# Defaults keep the previous full feature set: [zx, zy, zy-zx, zx*zy].
use_pair_diff=${use_pair_diff:-0}
use_pair_product=${use_pair_product:-1}
if [ "$use_pair_diff" = "1" ] && [ "$use_pair_product" = "1" ]; then
  pair_suffix=""
else
  pair_suffix="_pair_d${use_pair_diff}_p${use_pair_product}"
fi

python run.py \
  --task_name long_term_forecast \
  --is_relation_pretraining \
  --is_pretraining 0 \
  --is_training 0 \
  --is_zeroshot 0 \
  --root_path ./dataset/$data_name/ \
  --model_id ${data_name}_${seq_len}_relation_pretrain \
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
  --batch_size 16 \
  --domain_len 128 \
  --d_model 256 \
  --learning_rate 1e-4 \
  --weight_decay 0.05 \
  --head_type prediction \
  --num_workers 4 \
  --use_static_features \
  --use_static_kv \
  --static_emb_dim 64 \
  --static_dim 3 \
  --filter_approx \
  --support_ratio 0.66 \
  --x_data_path "d18O/d18O_clean.csv" \
  --y_data_path "accum/accum_100y.csv" \
  --relation_early_stop_epoch 15 \
  --relation_pretrain_samples_per_site 8 \
  --relation_loss_site_weight 1.0 \
  --relation_loss_aug_weight 0.5 \
  --relation_loss_xy_weight 0.1 \
  --relation_loss_proto_weight 0.2 \
  --relation_aug_noise_std 0.05 \
  --relation_aug_mask_ratio 0.1 \
  --use_pair_diff ${use_pair_diff} \
  --use_pair_product ${use_pair_product} \
  --use_online_prototypes \
  --num_prototypes $num_prototypes \
  --prototype_warmup_epochs 5 \
  --prototype_cluster_interval 2 \
  --prototype_momentum 0.9 \
  --prototype_temperature 0.2 \
  --prototype_output_path prototypes/d18O_accum_crossvar_relation_pretrain_proto_k${num_prototypes}${pair_suffix}.npz
