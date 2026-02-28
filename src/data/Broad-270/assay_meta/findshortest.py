import pandas as pd
import sys

def find_shortest_assay_pandas(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 定义我们需要关注的列
        target_columns = ['ASSAY_NAME', 'ASSAY_DESC', 'OBS_NAME', 'ASSAY_TYPE', 'READOUT_TYPE']
        
        # 检查列是否存在，防止报错
        missing_cols = [col for col in target_columns if col not in df.columns]
        if missing_cols:
            print(f"错误: CSV文件中缺少以下列: {missing_cols}")
            return

        # 填充空值为字符串空值，防止计算长度出错
        df_filled = df[target_columns].fillna('')

        # 计算每一行的总长度
        # apply方法将每一行的这些列拼接起来，然后计算长度
        df['total_length'] = df_filled.apply(lambda row: sum(len(str(val)) for val in row), axis=1)

        # 找到长度最小的行的索引
        min_length_index = df['total_length'].idxmin()
        
        # 获取对应的行数据
        shortest_row = df.loc[min_length_index]
        
        print("-" * 30)
        print("计算完成。")
        print(f"最短总长度: {shortest_row['total_length']}")
        print(f"对应的 ASSAY_NAME: {shortest_row['ASSAY_NAME']}")
        print("-" * 30)
        
        return shortest_row['ASSAY_NAME']

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")

# 使用示例：将 'your_data.csv' 替换为你的实际文件路径
if __name__ == "__main__":
    # 请在这里修改你的文件名
    csv_file_path = '/mnt/petrelfs/zhengqiaoyu.p/SunYuze/bioactivity_prediction_cleaned/data/Broad-270/assay_meta/assay_meta_expanded.csv' 
    find_shortest_assay_pandas(csv_file_path)