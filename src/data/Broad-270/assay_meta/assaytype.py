import pandas as pd
import sys
import os

def analyze_csv_column(file_path, column_name, encoding='utf-8'):
    """
    读取CSV文件并统计指定列的元素出现次数。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在。")
        return

    try:
        # 2. 读取CSV文件
        # keep_default_na=False 防止pandas自动将空字符串转换为NaN，保持原始数据格式
        df = pd.read_csv(file_path, encoding=encoding, keep_default_na=False)
        
        # 3. 检查列名是否存在
        if column_name not in df.columns:
            print(f"错误: 列名 '{column_name}' 在文件中未找到。")
            print(f"可用的列名有: {list(df.columns)}")
            return

        # 4. 获取指定列的数据
        column_data = df[column_name]

        # 5. 统计不重复元素及其计数 (value_counts 是 pandas 的核心功能)
        # dropna=False 表示我们也想统计空值（NaN/None）出现的次数
        value_counts = column_data.value_counts(dropna=False)

        # 6. 打印结果
        print(f"\n文件: {file_path}")
        print(f"目标列: {column_name}")
        print(f"不重复元素总数: {len(value_counts)}")
        print("-" * 40)
        print(f"{'元素内容':<25} | {'出现次数':<10}")
        print("-" * 40)

        for value, count in value_counts.items():
            # 处理显示格式，如果内容太长则截断
            display_value = str(value)
            if len(display_value) > 22:
                display_value = display_value[:20] + "..."
            
            print(f"{display_value:<25} | {count:<10}")
        
        print("-" * 40)

        # 可选：将结果保存到新文件
        # output_file = f"{column_name}_counts.csv"
        # value_counts.to_csv(output_file, header=['Count'])
        # print(f"统计结果已保存至: {output_file}")

    except UnicodeDecodeError:
        print("错误: 文件编码格式不对，尝试使用 'gbk' 或 'latin1' 编码重试。")
        # 递归尝试一次 GBK 编码
        if encoding == 'utf-8':
            print("正在尝试使用 GBK 编码重新读取...")
            analyze_csv_column(file_path, column_name, encoding='gbk')
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    csv_file = '/mnt/petrelfs/zhengqiaoyu.p/SunYuze/bioactivity_prediction/data/Broad-270/assay_meta/assay_meta_expanded.csv'
    target_col = 'ASSAY_TYPE'
    analyze_csv_column(csv_file, target_col)