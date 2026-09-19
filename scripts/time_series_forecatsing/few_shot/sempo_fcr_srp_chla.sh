#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

pretrain_setting=long_term_forecast_SEMPO_fcr_crossvar_relation_pretrain_k5_sl256_dm256_el3_pair_d0_p1_0

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29504 run.py \
  --task_name long_term_forecast \
  --model SEMPO \
  --data fcr_crossvar \
  --icecore_task_type crossvar \
  --root_path ./dataset/fcr/ \
  --x_data_path SRP_ugL.csv \
  --y_data_path Chla_ugL.csv \
  --is_pretraining 0 \
  --is_training 1 \
  --is_zeroshot 0 \
  --features M \
  --c_in 1 \
  --seq_len 256 \
  --pred_len 256 \
  --horizon_lengths 256 \
  --patch_len 64 \
  --stride 32 \
  --d_model 256 \
  --e_layers 3 \
  --d_layers 3 \
  --head_type prediction \
  --domain_len 128 \
  --filter_approx \
  --support_ratio 0.66 \
  --train_epochs 100 \
  --batch_size 6 \
  --num_workers 4 \
  --patience 6 \
  --loss MAE \
  --learning_rate 1e-3 \
  --warmup_steps 1000 \
  --lradj constant_with_warmup \
  --is_maml 1 \
  --inner_steps 2 \
  --inner_lr 5e-3 \
  --meta_lr 5e-5 \
  --target_anchor_residual \
  --target_anchor_len 50 \
  --use_static_features \
  --use_static_kv \
  --static_emb_dim 64 \
  --use_pair_diff 0 \
  --use_pair_product 1 \
  --use_encoder_proto_static \
  --encoder_proto_path prototypes/fcr_srp_chla_k3.npz \
  --relation_pretrain_checkpoint checkpoints/fcr_srp_chla/pretrain/${pretrain_setting}/checkpoint.pth \
  --checkpoints checkpoints/fcr_srp_chla/maml \
  "$@"
