import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# from statsmodels.robust import mad
from scipy import signal
import os
import pywt 

def modwt(x, wname='db2', level=2):
    """
    Maximal Overlap Discrete Wavelet Transform (MODWT)
    PyWavelets 默认支持的 Discrete Wavelet Transform (DWT) 是降采样的。
    由于没有原生的 MODWT，这里使用近似或无降采样的小波变换 (Stationary Wavelet Transform, SWT) 代替。
    """
    if len(x) % (2 ** level) != 0:
        # SWT 要求输入长度为 2^level 的倍数，先做一个简单的 padding
        pad_width = (2 ** level) - (len(x) % (2 ** level))
        x = np.pad(x, (0, pad_width), 'symmetric')
        padded = True
    else:
        padded = False

    # SWT 相当于 MODWT 的一维无降采样变换
    coeffs = pywt.swt(x, wname, level=level, start_level=0)
    
    return coeffs, padded, pad_width if padded else 0

def imodwt(coeffs, wname='db2'):
    """
    Inverse Stationary Wavelet Transform
    """
    return pywt.iswt(coeffs, wname)

def process_single_series(series, wname='db2', level=2):
    """
    对单个无 NaN 的连续序列进行小波去噪
    """
    # 获取有效部分
    valid_data = series.dropna().values
    if len(valid_data) == 0:
         return series

    # 1. 变换
    coeffs, padded, pad_width = modwt(valid_data, wname=wname, level=level)
    
    # 2. 阈值去噪 (假设去除高频噪声，即细节系数)
    # coeffs 的结构是 [(cA_level, cD_level), ..., (cA_1, cD_1)]
    new_coeffs = []
    for cA, cD in coeffs:
        # 可以置零，或者使用软阈值
        # 这里演示简单地将细节特征 (cD) 置 0 来移除高频
        cD_clean = np.zeros_like(cD)
        new_coeffs.append((cA, cD_clean))
        
    # 3. 重构
    clean_data = imodwt(new_coeffs, wname=wname)
    
    # 移除 padding
    if padded:
         clean_data = clean_data[:-pad_width]
         
    # 放回原始结构
    new_series = pd.Series(index=series.dropna().index, data=clean_data)
    
    # 返回一个与原输入长度相同的 Series，保留原始的 NaN 位置
    result = series.copy()
    result.loc[new_series.index] = new_series
    return result

def preprocess_icecore_data(accum_file, d18o_file, meta_file, output_path):
    # 1. 读取数据
    print("Loading datasets...")
    df_accum = pd.read_csv(accum_file)
    df_d18o = pd.read_csv(d18o_file)
    df_meta = pd.read_csv(meta_file) # 假设你有一个包含了 site, lat, lon, elev 的 metadata 
    
    # 将 "Year" 或 "Year CE" 设为索引
    year_col = "year" if "year" in df_accum.columns else df_accum.columns[0]
    
    df_accum.set_index(year_col, inplace=True)
    df_d18o.set_index(year_col, inplace=True)
    
    # 对齐站名：每个文件站点名称头尾空格删除
    df_accum.columns = df_accum.columns.str.strip()
    df_d18o.columns = df_d18o.columns.str.strip()
    df_meta['site'] = df_meta['site'].astype(str).str.strip()
    
    # 只修改偏差 1：以 Accumulation 数据为主，不再取交集
    target_sites = df_accum.columns.tolist()
    if 'year' in target_sites: target_sites.remove('year') # 排除 year 列
    
    print(f"Total Accumulation sites to process: {len(target_sites)}")

    # 初始化存储字典
    processed_data = {}
    
    # 2. 全局标准化准备 (在去噪后进行)
    accum_all_clean = []
    d18o_all_clean = []

    print("Phase 2: Wavelet Denoising...")
    for site in target_sites:
        accum_series = df_accum[site]
        accum_series = pd.to_numeric(accum_series, errors='coerce')
        # 如果降雪量数据中间断了，用线性插值补全 
        # (limit_area='inside' 保证只插值中间的缺口，不延长头尾)
        accum_series = accum_series.interpolate(method='linear', limit_area='inside')
        
        # 如果该站点在 d18O 文件中不存在，直接构造全为 NaN 的序列
        if site in df_d18o.columns:
            d18o_series = df_d18o[site]
        else:
            d18o_series = pd.Series(np.nan, index=accum_series.index)
        
        # 这里直接对齐两者共有的时间轴
        combined_df = pd.concat([accum_series, d18o_series], axis=1, keys=['Accum', 'd18O'])
        
        # 去噪
        accum_clean = process_single_series(combined_df['Accum'])
        d18o_clean = process_single_series(combined_df['d18O'])
        
        combined_df['Accum_Clean'] = accum_clean
        combined_df['d18O_Clean'] = d18o_clean
        
        # 记录下来用于全局标准化
        accum_all_clean.append(accum_clean.dropna().values)
        # 只有当 d18o 不是全 NaN 时才记录
        if combined_df['d18O_Clean'].notna().any():
            d18o_all_clean.append(d18o_clean.dropna().values)
        
        processed_data[site] = combined_df

    # 3. 计算全局 Z-score 参数
    print("Phase 3: Global Scaling and Mask Generation...")
    accum_global_data = np.concatenate(accum_all_clean)
    # 防止 d18o_all_clean 为空导致 concatenate 报错
    d18o_global_data = np.concatenate(d18o_all_clean) if len(d18o_all_clean) > 0 else np.array([0.0])
    
    accum_scaler = StandardScaler().fit(accum_global_data.reshape(-1, 1))
    d18o_scaler = StandardScaler().fit(d18o_global_data.reshape(-1, 1))

    final_tensors = {}

    for site in target_sites:  # 这里也要把 common_sites 换成 target_sites
        df = processed_data[site]
        
        # 标准化 Accumulation
        mask_accum = df['Accum_Clean'].notna()
        df.loc[mask_accum, 'Accum_Scaled'] = accum_scaler.transform(df.loc[mask_accum, 'Accum_Clean'].values.reshape(-1, 1)).flatten()
        
        # 标准化 d18O 并生成 Mask
        mask_d18o = df['d18O_Clean'].notna()
        df['d18O_Mask'] = mask_d18o.astype(int)
        
        # 有值的地方做标准化，没值的地方填 0
        df['d18O_Scaled'] = 0.0 
        if mask_d18o.any():
            df.loc[mask_d18o, 'd18O_Scaled'] = d18o_scaler.transform(df.loc[mask_d18o, 'd18O_Clean'].values.reshape(-1, 1)).flatten()
        
        # --- 获取静态特征 (假设 df_meta 已经准备好) ---
        site_meta = df_meta[df_meta['site'] == site].iloc[0]
        lat_norm = site_meta['Latitude'] 
        lon_sin = np.sin(site_meta['Longitude'] * np.pi / 180)
        lon_cos = np.cos(site_meta['Longitude'] * np.pi / 180)
        elev_norm = site_meta['Elevation']
        static_feat = np.array([lat_norm, lon_sin, lon_cos, elev_norm])
        
        # 保存结构
        # 过滤掉 Accumulation 为 NaN 的行
        valid_df = df.dropna(subset=['Accum_Scaled'])
        
        if len(valid_df) > 0:
            final_tensors[site] = {
                'time': valid_df.index.values,
                'dynamic': valid_df[['Accum_Scaled', 'd18O_Scaled', 'd18O_Mask']].values,
                'static': static_feat
            }

    # 保存为 npz 供 Dataloader 读取
    print(f"Saving to {output_path}...")
    np.savez(output_path, **final_tensors)
    print("Done!")

import pandas as pd
import numpy as np
import pywt
from scipy.stats import median_abs_deviation
import os

def process_single_series_level_dependent(series, wname='db2', level=4):
    """
    对单个包含 NaN 的连续序列进行：
    1. 提取非 NaN 数据
    2. 分层阈值小波去噪 (高频去噪，中低频保留)
    3. 放回原始序列的位置
    """
    valid_data = series.dropna().values
    if len(valid_data) == 0:
         return series

    # 1. 解决 SWT 的 Padding 问题 (必须是 2^level 的倍数)
    if len(valid_data) % (2 ** level) != 0:
        pad_width = (2 ** level) - (len(valid_data) % (2 ** level))
        x = np.pad(valid_data, (0, pad_width), 'symmetric')
        padded = True
    else:
        x = valid_data
        padded = False
        pad_width = 0

    # 保护机制：如果有效数据长度不足以做 level 层的分解，动态降低层级
    max_level = pywt.swt_max_level(len(x))
    actual_level = min(level, max_level)
    
    # 如果数据极短，连 1 层都做不了，直接返回原序列
    if actual_level == 0:
        return series

    # 2. 小波变换
    coeffs = pywt.swt(x, wname, level=actual_level, start_level=0)
    
    # 3. 分层阈值处理
    new_coeffs = []
    N = len(x)
    
    for i, (cA, cD) in enumerate(coeffs):
        # pywt.swt 的结果从最高层往下排，所以第一组是实际最高层
        current_level = actual_level - i 
        
        # 计算当前层的稳健噪音标准差
        sigma = median_abs_deviation(cD, scale='normal') 
        
        # 如果序列非常平，sigma 可能是 0，需要避免不必要的计算
        if sigma > 0:
            if current_level == 1:
                # cD1 (约 2-4 年周期): 重度去噪 (VisuShrink)
                threshold = sigma * np.sqrt(2 * np.log(N))
            # elif current_level == 2:
            #     # cD2 (约 4-8 年周期): 中度去噪
            #     threshold = sigma * 2.0 
            else:
                # > cD3 (大于 8 年周期): 完全保留
                threshold = 0.0 
        else:
            threshold = 0.0
            
        # 4. 应用软阈值 (Soft Thresholding)
        if threshold > 0:
            cD_clean = pywt.threshold(cD, value=threshold, mode='soft')
        else:
            cD_clean = cD
            
        new_coeffs.append((cA, cD_clean))
        
    # 5. 逆变换重构
    clean_data = pywt.iswt(new_coeffs, wname)
    
    # 6. 移除 padding
    if padded:
         clean_data = clean_data[:-pad_width]
         
    # 7. 塞回带有 NaN 的原始 Series 中
    result = series.copy()
    result.loc[series.dropna().index] = clean_data
    
    return result


def process_and_denoise_csv(input_csv, output_csv, wname='db2', level=4):
    """
    读取降雪量数据，线性插值后进行小波去噪，并输出
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 清理列名两端的空格
    df.columns = df.columns.str.strip()
    
    # 设置年份为索引
    year_col = "year" if "year" in df.columns else df.columns[0]
    df.set_index(year_col, inplace=True)
    
    # 记录站点（排除年份列，如果它被当成普通列保留了）
    sites = [col for col in df.columns if col.lower() != 'year']
    print(f"Total sites to process: {len(sites)}")
    
    # 初始化一个空的 DataFrame 用于存放清洗后的数据
    df_clean = pd.DataFrame(index=df.index)
    
    for site in sites:
        # 强制转换为数值格式，遇到无法解析的转为 NaN
        series = pd.to_numeric(df[site], errors='coerce')
        
        # Step 1: 线性插值（limit_area='inside' 确保只填补头尾之间的空缺，不向外延长）
        series_interp = series.interpolate(method='linear', limit_area='inside')
        
        # Step 2: 执行分层阈值小波去噪
        series_denoised = process_single_series_level_dependent(series_interp, wname=wname, level=level)
        
        df_clean[site] = series_denoised

    # 保存文件
    df_clean.to_csv(output_csv)
    print(f"Processing complete. Denoised data saved to '{output_csv}'.")

if __name__ == "__main__":
    input_file = "dataset/icecore/accum.csv"          # 替换为你实际的路径
    output_file = "dataset/icecore/aligned_clean.csv" # 替换为输出保存的路径
    
    if os.path.exists(input_file):
        process_and_denoise_csv(input_file, output_file)
    else:
        print(f"Error: {input_file} not found.")


    # preprocess_icecore_data('dataset/icecore/merged_accumulation.csv', 
    #                         'dataset/icecore/merged_d18O.csv', 
    #                         'dataset/icecore/data_sources.csv', 
    #                         'dataset/icecore/icecore_processed.npz')
    # pass