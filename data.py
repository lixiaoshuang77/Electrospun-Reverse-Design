import pandas as pd

def merge_and_sort_csv():
    # 1. 定义要合并的文件列表
    file_names = ['/mnt/d/MyDesktop/论文/code/MatMCL/datasets/table/mech/train.csv', '/mnt/d/MyDesktop/论文/code/MatMCL/datasets/table/mech/val.csv', '/mnt/d/MyDesktop/论文/code/MatMCL/datasets/table/mech/test.csv']
    df_list = []

    # 2. 逐个读取 CSV 文件并添加到列表中
    for file in file_names:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"成功读取: {file}")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {file}，请确保它在当前目录中。")
            return

    # 3. 将三个数据框合并为一个
    merged_df = pd.concat(df_list, ignore_index=True)

    # 4. 按照 'ID' 列从小到大进行排序
    # 注意：确保 'ID' 列的数据类型为数值型，如果是字符串，排序规则会有所不同
    merged_df = merged_df.sort_values(by='ID', ascending=True)

    # 5. 重置索引（为了让合并后的行号保持连续）
    merged_df = merged_df.reset_index(drop=True)

    # 6. 将处理后的数据保存为新的 CSV 文件
    output_filename = 'merged_sorted_data.csv'
    merged_df.to_csv(output_filename, index=False)
    
    print(f"\n合并与排序完成！")
    print(f"合并后的总行数: {len(merged_df)}")
    print(f"数据已保存为: {output_filename}")

if __name__ == "__main__":
    merge_and_sort_csv()