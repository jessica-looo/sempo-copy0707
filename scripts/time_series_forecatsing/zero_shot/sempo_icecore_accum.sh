export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Zero-shot" ]; then
    mkdir ./logs/Zero-shot
fi

# 1. 修改数据集名称，这通常对应你在 dataset 目录下的文件夹名
data_name=icecore
model_name=SEMPO

# 2. 序列长度设置 (假设你想用过去 64 年预测未来 16 年)
seq_len=32
label_len=16  # 通常为 seq_len 的一半
patch_len=8
stride=8

# 3. 目标预测长度设定
for percent in 50
do
for pred_len in 16
do
  torchrun --nnodes=1 --nproc_per_node=1 --master_port=29502 run.py \
    --task_name long_term_forecast \
    --is_pretraining 0 \
    --is_training 0 \
    --is_zeroshot 1 \
    --root_path ./dataset/$data_name/ \
    --data_path merged_accumulation_pywt.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data icecore_accum_year \
    --features S \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --stride $stride \
    --c_in 3 \
    --e_layers 3 \
    --d_layers 3 \
    --percent $percent \
    --train_epochs 10 \
    --pretrain_epochs 10 \
    --batch_size 32 \
    --des 'Exp' \
    --domain_len 128 \
    --d_model 256 \
    --learning_rate 1e-3 \
    --warmup_steps 1000 \
    --lradj constant_with_warmup \
    --head_type prediction \
    --num_workers 10 \
    --patience 6 \
    --pearson_loss_weight 0.1 \
    --freddf_loss_weight 0.1 \
    --soft_dtw_loss_weight 0.05 \
    --soft_dtw_gamma 0.1 \
    --soft_dtw_band 2 \
    --soft_dtw_normalize 1 \
    --use_multi_gpu 
done
done
