import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# from pathlib import Path
# import sys
# PROJECT_ROOT = Path.cwd().resolve().parent
# sys.path.insert(0, str(PROJECT_ROOT))

def clean_site_name(name: str) -> str:
    """
    清理站点名：
    1. 去掉前后空白
    2. 去掉前后引号（英文/中文引号都处理）
    3. 压缩多余空格
    """
    name = str(name).strip()
    name = name.strip('"\'“”‘’')
    name = ' '.join(name.split())
    return name



def load_and_align_data(accum_csv: str, data_sources_csv: str):
    """
    读取 accum 和 data_sources，并完成：
    - 站点名清理
    - 列名与 site 对齐
    - 输出聚类矩阵和位置表

    返回：
    X_df: 行=站点, 列=年份
    loc_df: index=站点, columns=[lat, lon, elevation]
    """
    accum = pd.read_csv(accum_csv)
    data_sources = pd.read_csv(data_sources_csv)

    accum = accum.rename(columns={c: clean_site_name(c) for c in accum.columns})

    data_sources = data_sources.copy()
    data_sources['site'] = data_sources['site'].map(clean_site_name)

    loc_df = data_sources[['site', 'Latitude', 'Longitude', 'Elevation']].copy()
    loc_df = loc_df.rename(columns={
        'Latitude': 'lat',
        'Longitude': 'lon',
        'Elevation': 'elevation'
    }).set_index('site')

    site_cols = [c for c in accum.columns if c != 'year']

    missing_sites = sorted(set(site_cols) - set(loc_df.index))
    if missing_sites:
        raise ValueError(
            '以下 站点在 data_sources 中没有匹配到：\n' + '\n'.join(missing_sites)
        )

    X_df = accum.set_index('year')[site_cols].T
    X_df = X_df.dropna(how='all')
    X_df = X_df.apply(lambda row: row.fillna(row.mean()), axis=1)

    loc_df = loc_df.loc[X_df.index]
    return X_df, loc_df



def standardize_sites(X: np.ndarray, mode: str | None = None) -> np.ndarray:
    """
    标准化方式：
    - None: 不标准化
    - 'zscore_per_site': 每个站点（每一行）按时间做 z-score
      => 更关注“变化形状”，弱化绝对大小
    - 'zscore_per_year': 每个年份（每一列）按站点做 z-score
      => 一般不建议用于这个任务
    """
    if mode is None:
        return X

    X = X.astype(float).copy()

    if mode == 'zscore_per_site':
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (X - mean) / std

    if mode == 'zscore_per_year':
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    raise ValueError(f'未知 standardize mode: {mode}')


# =====================
# FINCH风格层次聚类
# =====================
def clust_rank(mat, metric='cosine'):
    dist_mat = metrics.pairwise.pairwise_distances(mat, mat, metric=metric)
    np.fill_diagonal(dist_mat, float('inf'))
    first_neighbors = np.argmin(dist_mat, axis=1)
    adj_matrix = sp.csr_matrix(
        (np.ones_like(first_neighbors, dtype=np.float32),
         (np.arange(mat.shape[0]), first_neighbors)),
        shape=(mat.shape[0], mat.shape[0])
    )
    return adj_matrix



def get_clust(adj_matrix):
    num_components, labels = sp.csgraph.connected_components(
        csgraph=adj_matrix,
        directed=True,
        connection='weak',
        return_labels=True
    )
    return num_components, labels



def cool_mean(data, partition):
    unique_labels, counts = np.unique(partition, return_counts=True)
    num_features = data.shape[1]
    centroids = np.zeros((len(unique_labels), num_features), dtype=np.float32)

    for i in range(num_features):
        centroids[:, i] = np.bincount(partition, weights=data[:, i])

    centroids /= counts[:, np.newaxis]
    return centroids



def get_merge(partition, group, data):
    if partition.size > 0:
        _, inv_idx = np.unique(partition, return_inverse=True)
        new_partition = group[inv_idx]
    else:
        new_partition = group

    new_centroids = cool_mean(data, new_partition)
    return new_partition, new_centroids



def hierarchical_clustering(x, distance='cosine', max_layers=10, verbose=True):
    """
    x: ndarray, shape=(n_sites, n_years)
    返回：
    all_partitions: 每一列是一层聚类标签
    num_clusters_per_level: 每层簇数
    """
    if verbose:
        print(f'\n开始执行FINCH风格层次聚类 (distance={distance})...')

    adj = clust_rank(x, metric=distance)
    num_clust_initial, initial_partition = get_clust(adj)

    if num_clust_initial == x.shape[0]:
        if verbose:
            print('初始层没有发生合并，每个站点单独成簇。')
        return np.array([]), []

    all_partitions, current_centroids = get_merge(np.array([]), initial_partition, x)
    latest_partition = initial_partition
    num_clusters_per_level = [num_clust_initial]

    if verbose:
        print(f'第 0 层: {num_clust_initial} 个聚类')

    for k in range(1, max_layers):
        adj = clust_rank(current_centroids, metric=distance)
        num_clust_curr, group = get_clust(adj)

        if num_clust_curr >= num_clusters_per_level[-1] or num_clust_curr == 1:
            if verbose:
                print('聚类数量不再减少，或已合并为 1，提前停止。')
            break

        latest_partition, current_centroids = get_merge(latest_partition, group, x)
        all_partitions = np.column_stack((all_partitions, latest_partition))
        num_clusters_per_level.append(num_clust_curr)

        if verbose:
            print(f'第 {k} 层: {num_clust_curr} 个聚类')

    return all_partitions, num_clusters_per_level



def visualize_clusters_on_map(
    cluster_df: pd.DataFrame,
    level_to_plot: int = 0,
    output_png: str | None = None,
    title: str | None = None,
    annotate: bool = True,
    jitter: float = 0.25,
):
    """
    按经纬度把聚类结果画成 2D 地图散点图并保存。
    不依赖 cartopy，直接使用 lon-lat 平面坐标，保证大多数环境能运行。

    参数：
    - cluster_df: index=site，包含 Level_k、lat、lon
    - level_to_plot: 画哪一层聚类
    - output_png: 输出图片文件名；为 None 时自动生成
    - annotate: 是否标注站点名
    - jitter: 给点加一点随机扰动，避免完全重叠
    """
    col_name = f'Level_{level_to_plot}'
    if col_name not in cluster_df.columns:
        raise ValueError(f'{col_name} 不存在，可用列: {cluster_df.columns.tolist()}')

    plot_df = cluster_df.dropna(subset=['lat', 'lon', col_name]).copy()
    if plot_df.empty:
        raise ValueError('没有可用于绘图的经纬度或聚类标签。')

    unique_labels = sorted(plot_df[col_name].unique())
    cmap = plt.cm.get_cmap('tab20', max(len(unique_labels), 1))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

    rng = np.random.default_rng(42)
    lon = plot_df['lon'].to_numpy(dtype=float) + rng.normal(0, jitter, size=len(plot_df))
    lat = plot_df['lat'].to_numpy(dtype=float) + rng.normal(0, jitter, size=len(plot_df))

    fig, ax = plt.subplots(figsize=(12, 8))

    for lab in unique_labels:
        mask = plot_df[col_name].to_numpy() == lab
        ax.scatter(
            lon[mask], lat[mask],
            s=70,
            color=cmap(label_to_idx[lab]),
            edgecolors='black',
            linewidths=0.5,
            alpha=0.85,
            label=f'Cluster {lab}'
        )

    if annotate:
        for site, x, y in zip(plot_df.index, lon, lat):
            ax.text(x + 0.2, y + 0.1, site, fontsize=7, alpha=0.8)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, -55)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(title='Clusters', bbox_to_anchor=(1.02, 1), loc='upper left')

    if title is None:
        title = f'Antarctic sites clustering map ({col_name})'
    ax.set_title(title)

    plt.tight_layout()

    if output_png is None:
        output_png = f'cluster_map_{col_name}.png'

    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'聚类地图已保存到: {output_png}')
    return output_png



def save_all_cluster_maps(cluster_df: pd.DataFrame, output_prefix: str = 'cluster_map', annotate: bool = True):
    """
    把所有 Level_* 都画出来并保存。
    返回所有输出图片路径列表。
    """
    level_cols = [c for c in cluster_df.columns if c.startswith('Level_')]
    outputs = []
    for col in level_cols:
        level = int(col.split('_')[1])
        out = f'{output_prefix}_{col}.png'
        visualize_clusters_on_map(
            cluster_df,
            level_to_plot=level,
            output_png=out,
            annotate=annotate,
            title=f'Antarctic sites clustering map ({col})'
        )
        outputs.append(out)
    return outputs



def run_pipeline(
    accum_csv='accum.csv',
    data_sources_csv='data_sources.csv',
    distance='cosine',
    standardize='zscore_per_site',
    max_layers=10,
    output_csv='cluster_results.csv',
    make_maps=True,
    map_prefix='cluster_map',
    annotate=True,
):
    """
    推荐默认：
    - distance='cosine'
    - standardize='zscore_per_site'

    如果你想保留绝对 accumulation 大小信息：
    - distance='euclidean'
    - standardize=None
    """
    X_df, loc_df = load_and_align_data(accum_csv, data_sources_csv)
    X = X_df.values
    X = standardize_sites(X, mode=standardize)

    partitions, num_clusters = hierarchical_clustering(
        X,
        distance=distance,
        max_layers=max_layers,
        verbose=True
    )

    if partitions.size == 0:
        print('没有得到有效聚类结果。')
        return None, None, None

    if partitions.ndim == 1:
        partitions = partitions.reshape(-1, 1)

    result_df = pd.DataFrame(partitions, index=X_df.index)
    result_df.columns = [f'Level_{i}' for i in range(result_df.shape[1])]
    result_df = result_df.merge(loc_df, left_index=True, right_index=True, how='left')

    result_df.to_csv(output_csv, encoding='utf-8-sig')
    print(f'聚类结果已保存到: {output_csv}')
    print(f'每层聚类数: {num_clusters}')

    map_paths = []
    if make_maps:
        map_paths = save_all_cluster_maps(result_df, output_prefix=map_prefix, annotate=annotate)

    return result_df, num_clusters, map_paths


if __name__ == '__main__':
    result_df, num_clusters, map_paths = run_pipeline(
        accum_csv='dataset/icecore/chem/chem.csv',
        data_sources_csv='dataset/icecore/data_sources.csv',
        distance='euclidean',
        standardize=None,
        max_layers=10,
        output_csv='cluster/chem/chem_cluster.csv',
        make_maps=False,
        # map_prefix='cluster_map_euc_ori',
        annotate=True,
    )
    print('输出图片:', map_paths)
