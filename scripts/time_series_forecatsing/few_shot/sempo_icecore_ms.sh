#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/MS" ]; then mkdir ./logs/MS; fi

# 基础配置
data_name=icecore
model_name=SEMPO
seq_len=32
label_len=16
pred_len=16
patch_len=16
stride=8

# MS 两变量输入配置。
# 支持 accum / chem / d18O 中任意两个变量；target_var 必须是 input_var_1 或 input_var_2。
# 例如:
#   accum -> chem: input_var_1=accum, input_path_1="accum/accum_100y.csv", input_var_2=chem, input_path_2="chem/chem.csv", target_var=chem
#   d18O  -> chem: input_var_1=d18O,  input_path_1="d18O/d18O_clean.csv", input_var_2=chem, input_path_2="chem/chem.csv", target_var=chem
#   chem  -> d18O: input_var_1=chem,  input_path_1="chem/chem.csv", input_var_2=d18O, input_path_2="d18O/d18O_clean.csv", target_var=d18O
input_var_1=d18O
input_path_1="d18O/d18O_clean.csv"
input_var_2=accum
input_path_2="accum/accum_100y.csv"
target_var=accum

case "$target_var" in
  accum)
    cluster_result_path="cluster/accum/accum_100y_cluster_mapped.csv"
    ;;
  chem)
    cluster_result_path="cluster/chem/chem_cluster_mapped.csv"
    ;;
  d18O)
    cluster_result_path="cluster/d18O/d18O_cluster_mapped.csv"
    ;;
  *)
    echo "Unsupported target_var: $target_var. Expected accum, chem, or d18O."
    exit 1
    ;;
esac

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29503 run.py \
  --task_name long_term_forecast \
  --is_pretraining 0 \
  --is_training 1 \
  --is_zeroshot 0 \
  --root_path ./dataset/$data_name/ \
  --model_id ${data_name}_${input_var_1}_${input_var_2}_to_${target_var}_${seq_len} \
  --model $model_name \
  --data d18O_accum_ms \
  --icecore_task_type ms \
  --features MS \
  --target $target_var \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len $label_len \
  --patch_len $patch_len \
  --stride $stride \
  --horizon_lengths $pred_len \
  --c_in 2 \
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
  --cluster_result_path "$cluster_result_path" \
  --is_maml 1 \
  --inner_steps 2 \
  --inner_lr 5e-3 \
  --meta_lr 5e-5 \
  --filter_approx \
  --wavelet_loss_weight 0 \
  --wavelet_band_weights "0.5,0.2,0.2,0.5" \
  --pearson_loss_weight 0.1 \
  --freddf_loss_weight 0.1 \
  --soft_dtw_loss_weight 0.05 \
  --soft_dtw_gamma 0.1 \
  --soft_dtw_band 2 \
  --soft_dtw_normalize 1 \
  --support_ratio 0.66 \
  --x_data_path "$input_path_1" \
  --y_data_path "$input_path_2" \
  --residual_prediction \
  --apply_pywt \
  --use_encoder_proto_static \
  --encoder_proto_path prototypes/accum_s_encoder_proto_k6.npz
