import torch

def inspect_checkpoint(file_path):
    # 将模型加载到 CPU，避免在没有 GPU 的环境下报错
    checkpoint = torch.load(file_path, map_location='cpu')
    # 有些框架会把权重包在一层字典里，如 'state_dict' 或 'model'
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # 如果直接保存的就是 state_dict
        state_dict = checkpoint

    print(f"总计包含 {len(state_dict)} 个参数层。")
    print("=" * 70)
    print(f"{'Layer Name (层级名称)':<50} | {'Shape (张量形状)'}")
    print("-" * 70)
    
    # 遍历字典并打印名称和形状
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:<50} | {list(value.shape)}")
        else:
            print(f"{key:<50} | Non-Tensor ({type(value).__name__})")

# 替换为你的 pth 文件路径进行查看
if __name__ == "__main__":
    inspect_checkpoint('checkpoints/long_term_forecast_SEMPO_UTSD_ftM_sl512_ll48_pl96_pl64_dm256_nh2_el3_dl3_df128_fc1_ebtimeF_dtTrue_0/checkpoint.pth')