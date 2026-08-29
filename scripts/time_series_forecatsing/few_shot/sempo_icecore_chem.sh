export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Few-shot" ]; then
    mkdir ./logs/Few-shot
fi

data_name=icecore
var_name=chem
model_name=SEMPO

seq_len=32
label_len=16
patch_len=16
stride=8

for percent in 100
do
for pred_len in 16
do
  torchrun --nnodes=1 --nproc_per_node=1 --master_port=29502 run.py \
    --task_name long_term_forecast \
    --is_pretraining 0 \
    --is_training 1 \
    --is_zeroshot 0 \
    --root_path ./dataset/$data_name/$var_name \
    --data_path chem.csv \
    --model_id $data_name'_chem_'$seq_len'_'$pred_len \
    --model $model_name \
    --data chem_s \
    --apply_pywt \
    --features S \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --stride $stride \
    --horizon_lengths 16 \
    --c_in 1 \
    --e_layers 3 \
    --d_layers 3 \
    --percent $percent \
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
    --static_dim 13 \
    --cluster_result_path "cluster/chem/chem_cluster_mapped.csv" \
    --is_maml 1 \
    --inner_steps 2 \
    --inner_lr 5e-3 \
    --meta_lr 5e-5 \
    --filter_approx \
    --wavelet_loss_weight 0.1 \
    --wavelet_band_weights "0.5,0.2,0.2,0.5" \
    --pearson_loss_weight 0.1 \
    --freddf_loss_weight 0.1 \
    --soft_dtw_loss_weight 0.05 \
    --soft_dtw_gamma 0.1 \
    --soft_dtw_band 2 \
    --soft_dtw_normalize 1 \
    --residual_prediction \
    --support_ratio 0.66 \

done
done
