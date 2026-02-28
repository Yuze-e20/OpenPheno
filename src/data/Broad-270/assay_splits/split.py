import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def check_split_quality(df_train, df_val, df_test):
    """检查划分质量"""
    # 1. 检查 Assay Type 分布
    all_types = set(df_train['ASSAY_TYPE']) | set(df_val['ASSAY_TYPE']) | set(df_test['ASSAY_TYPE'])
    train_counts = df_train['ASSAY_TYPE'].value_counts().reindex(list(all_types), fill_value=0)
    test_counts = df_test['ASSAY_TYPE'].value_counts().reindex(list(all_types), fill_value=0)
    
    contingency_test = pd.DataFrame({'Train': train_counts, 'Test': test_counts})
    contingency_test = contingency_test.loc[~(contingency_test==0).all(axis=1)]
    
    # 如果某一列全是0，chi2会报错，加个保护
    if contingency_test.shape[0] < 2:
        p_type_test = 0 # 无法计算，视为失败
    else:
        chi2_test, p_type_test, _, _ = stats.chi2_contingency(contingency_test)
    
    # 2. 检查 Sample Size 分布
    try:
        _, p_size = stats.kruskal(df_train['Total_Samples'], df_val['Total_Samples'], df_test['Total_Samples'])
    except ValueError:
        p_size = 0

    # 3. 检查 Positive Rate 分布
    try:
        _, p_rate = stats.kruskal(df_train['Positive_Rate'], df_val['Positive_Rate'], df_test['Positive_Rate'])
    except ValueError:
        p_rate = 0
    
    # 判定标准：P > 0.05
    is_good = (p_type_test > 0.05) and (p_size > 0.05) and (p_rate > 0.05)
    
    return is_good, p_type_test, p_size, p_rate

def split_assays_robust(meta_path, label_path, output_prefix="split", max_attempts=500):
    # --- 1. 读取并检查数据 ---
    print(f"正在读取 Meta 数据: {meta_path}")
    df_meta = pd.read_csv(meta_path)
    
    print("-" * 30)
    print("CSV 列名检查:")
    print(df_meta.columns.tolist())
    print("-" * 30)
    
    # 关键列名检查 (根据你的报错，这里最容易出问题)
    required_cols = ['PUMA_ASSAY_ID', 'ASSAY_TYPE', 'Total_Samples', 'Positive_Count']
    missing_cols = [c for c in required_cols if c not in df_meta.columns]
    if missing_cols:
        raise ValueError(f"❌ 错误: Meta文件中缺少关键列: {missing_cols}。请检查CSV表头！")

    print(f"正在读取 Label Matrix 头部: {label_path}")
    df_labels_head = pd.read_csv(label_path, nrows=0) 
    
    # --- 2. 数据清洗 ---
    label_cols = [c for c in df_labels_head.columns if c != 'smiles']
    df_meta['PUMA_ASSAY_ID'] = df_meta['PUMA_ASSAY_ID'].astype(str)
    
    # 过滤
    df_meta = df_meta[df_meta['PUMA_ASSAY_ID'].isin(label_cols)].copy()
    df_meta = df_meta.drop_duplicates(subset=['PUMA_ASSAY_ID'])
    
    print(f"有效 Assay 数: {len(df_meta)}")
    if len(df_meta) < 10:
        raise ValueError("❌ 错误: 有效 Assay 数量过少，无法进行 Train/Val/Test 划分。")

    # --- 3. 鲁棒的特征工程 (防止分层报错) ---
    df_meta['Positive_Rate'] = df_meta['Positive_Count'] / df_meta['Total_Samples']
    
    # 分箱
    n_bins = 3
    df_meta['Size_Bin'] = pd.qcut(df_meta['Total_Samples'].rank(method='first'), q=n_bins, labels=['S_Low', 'S_Mid', 'S_High'])
    df_meta['Rate_Bin'] = pd.qcut(df_meta['Positive_Rate'].rank(method='first'), q=n_bins, labels=['R_Low', 'R_Mid', 'R_High'])

    # 构建初始分层键
    df_meta['Strata'] = (df_meta['ASSAY_TYPE'].astype(str) + "_" + 
                         df_meta['Size_Bin'].astype(str) + "_" + 
                         df_meta['Rate_Bin'].astype(str))

    # === 核心修复: 三级回退策略 ===
    # 逻辑：如果某个组样本少于3个（无法分给Train/Val/Test各一个），就合并到更大的组
    
    # 第一轮检查：详细分层
    counts = df_meta['Strata'].value_counts()
    rare_indices = counts[counts < 3].index
    # 回退到 Level 2: 仅按 Assay Type
    df_meta.loc[df_meta['Strata'].isin(rare_indices), 'Strata'] = df_meta.loc[df_meta['Strata'].isin(rare_indices), 'ASSAY_TYPE']
    
    # 第二轮检查：Assay Type
    counts = df_meta['Strata'].value_counts()
    rare_indices = counts[counts < 4].index
    # 回退到 Level 3: "Other" (大杂烩，确保绝对不报错)
    if len(rare_indices) > 0:
        print(f"警告: 发现极少样本的 Assay Type {rare_indices.tolist()}，将其归类为 'Mixed_Strata'")
        df_meta.loc[df_meta['Strata'].isin(rare_indices), 'Strata'] = 'Mixed_Strata'
        
    # 最终检查 "Mixed_Strata" 是否也少于3个 (极罕见情况)
    # 如果 Mixed_Strata 只有1或2个，那就强制不分层，随机分配
    mixed_count = len(df_meta[df_meta['Strata'] == 'Mixed_Strata'])
    if 0 < mixed_count < 3:
         # 实在没办法，把这些行随便并入一个存在的组，比如数量最多的组
         most_common_strata = df_meta['Strata'].mode()[0]
         df_meta.loc[df_meta['Strata'] == 'Mixed_Strata', 'Strata'] = most_common_strata

    print("分层构建完成，Strata 分布如下 (Top 5):")
    print(df_meta['Strata'].value_counts().head())

    # --- 4. 自动搜索最佳随机种子 ---
    print(f"\n开始搜索最佳随机种子 (Max attempts: {max_attempts})...")
    
    best_seed = None
    best_metrics = (-1, -1, -1)
    train_meta, val_meta, test_meta = None, None, None
    
    for seed in range(max_attempts):
        try:
            # 这里的 stratify 应该非常安全了
            train_val_meta, test_split = train_test_split(
                df_meta, test_size=0.2, stratify=df_meta['Strata'], random_state=seed
            )
            
            # 第二次划分 (Val)
            # 注意：Train/Val 划分时，某些 Strata 在 train_val_meta 中可能只剩 <2 个了
            # 所以这里需要再次动态检查，如果报错就退化
            try:
                train_split, val_split = train_test_split(
                    train_val_meta, test_size=0.125, stratify=train_val_meta['Strata'], random_state=seed
                )
            except ValueError:
                # 如果第二次分层失败，尝试仅按 ASSAY_TYPE
                try:
                    train_split, val_split = train_test_split(
                        train_val_meta, test_size=0.125, stratify=train_val_meta['ASSAY_TYPE'], random_state=seed
                    )
                except ValueError:
                    # 如果还失败，完全随机
                    train_split, val_split = train_test_split(
                        train_val_meta, test_size=0.125, random_state=seed
                    )
            
            # 检查质量
            is_good, p_type, p_size, p_rate = check_split_quality(train_split, val_split, test_split)
            
            current_score = p_type + p_size + p_rate
            best_score = best_metrics[0] + best_metrics[1] + best_metrics[2]
            
            # 更新最佳记录
            if current_score > best_score:
                best_metrics = (p_type, p_size, p_rate)
                best_seed = seed
                train_meta, val_meta, test_meta = train_split, val_split, test_split
            
            if is_good:
                print(f"✅ 找到完美种子: {seed}")
                print(f"   P-values -> Type: {p_type:.4f}, Size: {p_size:.4f}, Rate: {p_rate:.4f}")
                break
                
        except Exception as e:
            if seed == 0: # 只打印第一次报错，避免刷屏
                print(f"⚠️ Seed {seed} 划分失败 (后续错误隐藏): {e}")
            continue
            
    if train_meta is None:
        raise RuntimeError("❌ 严重错误: 所有随机种子尝试均失败，请检查数据量是否太小或类别极度不平衡。")

    if best_seed is not None and not is_good:
        print(f"⚠️ 未找到完美种子，使用最佳结果 (Seed {best_seed})")
        print(f"   Best P-values -> Type: {best_metrics[0]:.4f}, Size: {best_metrics[1]:.4f}, Rate: {best_metrics[2]:.4f}")

    # --- 5. 保存结果 ---
    print("\n正在读取完整 Label Matrix 并保存结果...")
    df_labels = pd.read_csv(label_path)
    
    # 获取列名列表
    train_cols = ['smiles'] + train_meta['PUMA_ASSAY_ID'].tolist()
    val_cols = ['smiles'] + val_meta['PUMA_ASSAY_ID'].tolist()
    test_cols = ['smiles'] + test_meta['PUMA_ASSAY_ID'].tolist()
    
    # 再次过滤，确保列名在 label matrix 中存在
    valid_cols = set(df_labels.columns)
    train_cols = [c for c in train_cols if c in valid_cols]
    val_cols = [c for c in val_cols if c in valid_cols]
    test_cols = [c for c in test_cols if c in valid_cols]

    df_labels[train_cols].to_csv(f"{output_prefix}_train.csv", index=False)
    df_labels[val_cols].to_csv(f"{output_prefix}_val.csv", index=False)
    df_labels[test_cols].to_csv(f"{output_prefix}_test.csv", index=False)
    
    print(f"文件保存完毕: {output_prefix}_train.csv, {output_prefix}_val.csv, {output_prefix}_test.csv")
    
    return df_meta, train_meta, val_meta, test_meta

def evaluate_split(df_all, df_train, df_val, df_test):
    print("\n" + "="*40)
    print("最终划分结果评价指标")
    print("="*40)
    
    datasets = {'Train': df_train, 'Val': df_val, 'Test': df_test}
    
    # 1. Assay Type
    print("\n[1. Assay Type Distribution (%)]")
    type_dist = pd.DataFrame()
    for name, df in datasets.items():
        type_dist[name] = df['ASSAY_TYPE'].value_counts(normalize=True) * 100
    print(type_dist.round(2))
    
    # 2. Sample Size
    print("\n[2. Average Sample Size]")
    size_stats = pd.DataFrame()
    for name, df in datasets.items():
        size_stats.loc['Mean', name] = df['Total_Samples'].mean()
        size_stats.loc['Median', name] = df['Total_Samples'].median()
    print(size_stats.round(1))
    
    s_stat, s_p = stats.kruskal(df_train['Total_Samples'], df_val['Total_Samples'], df_test['Total_Samples'])
    print(f"Kruskal-Wallis P-value (Size): {s_p:.4f}")
    
    # 3. Positive Rate
    print("\n[3. Average Positive Rate (%)]")
    rate_stats = pd.DataFrame()
    for name, df in datasets.items():
        rate_stats.loc['Mean', name] = df['Positive_Rate'].mean() * 100
    print(rate_stats.round(3))
    
    r_stat, r_p = stats.kruskal(df_train['Positive_Rate'], df_val['Positive_Rate'], df_test['Positive_Rate'])
    print(f"Kruskal-Wallis P-value (Rate): {r_p:.4f}")

# --- 执行代码 ---
meta_file = '/mnt/petrelfs/zhengqiaoyu.p/SunYuze/bioactivity_prediction/data/Broad-270/assay_meta/assay_meta_expanded.csv'
label_file = '/mnt/petrelfs/zhengqiaoyu.p/SunYuze/bioactivity_prediction/data/Broad-270/label/compound_assay_matrix_270.csv'

try:
    all_meta, train_meta, val_meta, test_meta = split_assays_robust(meta_file, label_file)
    evaluate_split(all_meta, train_meta, val_meta, test_meta)
except Exception as e:
    import traceback
    traceback.print_exc()