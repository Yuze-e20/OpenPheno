import pandas as pd
import sys

def check_scaffold_split(scaffold_file, train_file, test_file):
    """
    检查数据集是否按照 Scaffold 进行划分。
    
    参数:
    scaffold_file: 包含 SMILES, murcko_scaffold, cluster_id 的 CSV 文件路径
    train_file: 训练集 CSV 文件路径
    test_file: 测试集 CSV 文件路径
    """
    
    print("正在加载数据...")
    try:
        # 读取 CSV 文件
        # 注意：实际使用时可能需要根据文件具体情况调整 sep (分隔符) 或 encoding
        df_scaffold = pd.read_csv(scaffold_file)
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 1. 预处理列名和数据
    # 假设 Scaffold 文件中的 SMILES 列名为 'SMILES'
    # 假设 Train/Test 文件中的 SMILES 列名为 'Compound SMILES' (根据你的描述)
    
    scaffold_smiles_col = 'SMILES'
    cluster_id_col = 'cluster_id'
    data_smiles_col = 'Compound SMILES'

    # 检查列是否存在
    if scaffold_smiles_col not in df_scaffold.columns or cluster_id_col not in df_scaffold.columns:
        print(f"错误: Scaffold 文件中缺少 '{scaffold_smiles_col}' 或 '{cluster_id_col}' 列")
        return
    if data_smiles_col not in df_train.columns:
        print(f"错误: Train 文件中缺少 '{data_smiles_col}' 列")
        return
    if data_smiles_col not in df_test.columns:
        print(f"错误: Test 文件中缺少 '{data_smiles_col}' 列")
        return

    # 建立 SMILES 到 Cluster ID 的映射字典
    # 使用 strip() 去除可能的空白字符，确保匹配准确
    # 如果 SMILES 格式不标准（如 Canonical 形式不同），可能需要 RDKit 进行标准化，这里假设字符串完全一致
    smiles_to_cluster = pd.Series(
        df_scaffold[cluster_id_col].values, 
        index=df_scaffold[scaffold_smiles_col]
    ).to_dict()

    print(f"Scaffold 映射表加载完毕，共包含 {len(smiles_to_cluster)} 个化合物。")

    # --- 步骤 1: 检查覆盖率 (是否所有化合物都有记载) ---
    print("\n--- 步骤 1: 检查化合物是否在 Scaffold 表中有记录 ---")
    
    def check_coverage(df, name):
        missing_smiles = []
        found_clusters = set()
        
        for smiles in df[data_smiles_col]:
            # 简单的清洗，防止空值
            if pd.isna(smiles):
                continue
            
            if smiles in smiles_to_cluster:
                found_clusters.add(smiles_to_cluster[smiles])
            else:
                missing_smiles.append(smiles)
        
        return missing_smiles, found_clusters

    train_missing, train_clusters = check_coverage(df_train, "Train")
    test_missing, test_clusters = check_coverage(df_test, "Test")

    if train_missing:
        print(f"警告: Train 集中有 {len(train_missing)} 个 SMILES 未在 Scaffold 表中找到对应的 Cluster ID。")
        # print(f"前5个缺失的 SMILES: {train_missing[:5]}")
    else:
        print("通过: Train 集中所有 SMILES 均已找到对应的 Cluster ID。")

    if test_missing:
        print(f"警告: Test 集中有 {len(test_missing)} 个 SMILES 未在 Scaffold 表中找到对应的 Cluster ID。")
        # print(f"前5个缺失的 SMILES: {test_missing[:5]}")
    else:
        print("通过: Test 集中所有 SMILES 均已找到对应的 Cluster ID。")

    if train_missing or test_missing:
        print("注意: 由于存在未找到 ID 的化合物，后续的重叠检查可能不完全准确（仅基于已找到 ID 的部分）。")

    # --- 步骤 2: 检查 Cluster ID 是否重叠 ---
    print("\n--- 步骤 2: 检查 Train 和 Test 的 Cluster ID 是否重叠 ---")
    
    intersection = train_clusters.intersection(test_clusters)
    
    print(f"Train 集包含 {len(train_clusters)} 个唯一的 Cluster ID")
    print(f"Test 集包含 {len(test_clusters)} 个唯一的 Cluster ID")
    
    if len(intersection) > 0:
        print(f"❌ 失败: 检测到数据泄漏！Train 和 Test 共有 {len(intersection)} 个相同的 Cluster ID。")
        print("这意味着这不是一个严格的 Scaffold Split，或者同一个骨架被分到了不同的集合中。")
        print(f"重叠的 Cluster ID 示例: {list(intersection)[:10]}")
    else:
        print("✅ 成功: Train 和 Test 之间没有重叠的 Cluster ID。")
        print("这份数据集是按照 Scaffold (Cluster ID) 进行严格划分的。")

# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    # 请在这里替换为你实际的文件路径
    scaffold_csv_path = '/mnt/petrelfs/zhengqiaoyu.p/SunYuze/bioactivity_prediction_cleaned/data/rawdata/clustered_compounds_209.csv'  # 包含 cluster_id 的文件
    train_csv_path = '/mnt/petrelfs/zhengqiaoyu.p/SunYuze/bioactivity_prediction_cleaned/data/ChEMBL-209/splits/train1.csv'             # 训练集
    test_csv_path = '/mnt/petrelfs/zhengqiaoyu.p/SunYuze/bioactivity_prediction_cleaned/data/ChEMBL-209/splits/test1.csv'               # 测试集

    # 运行检查
    # 如果你没有实际文件想测试代码，请确保路径指向存在的文件
    # 这里为了演示，我注释掉了函数调用，你需要取消注释并填入真实路径
    
    check_scaffold_split(scaffold_csv_path, train_csv_path, test_csv_path)
    
    print("请在代码末尾修改文件路径并调用 check_scaffold_split 函数。")