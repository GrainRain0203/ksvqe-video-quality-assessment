import argparse

import pandas as pd
import numpy as np


def generate_contrastive_labels(input_csv_path, output_txt_path, group_size=7):
    # 1. 读取原始的 CSV 文档（确保 MOS 分对应准确）
    df = pd.read_csv(input_csv_path)

    # 获取总行数
    num_rows = len(df)

    # 2. 生成标签：每 7 个视频属于同一个 ID (0, 1, 2...)
    # np.arange(num_rows // group_size + 1) 生成 0, 1, 2...
    # np.repeat(..., group_size) 将每个数字重复 7 次
    # [:num_rows] 截取到与数据行数完全一致，防止多余
    labels = np.repeat(np.arange(num_rows // group_size + 1), group_size)[:num_rows]

    # 3. 按照你训练集的格式拼装新的 DataFrame（文件路径, 标签1, 标签2, MOS分）
    new_df = pd.DataFrame({
        'filename': df.iloc[:, 0],  # 第一列是视频路径
        'label1': labels,  # 插入的对比学习标签列1
        'label2': labels,  # 插入的对比学习标签列2
        'score': df.iloc[:, 1]  # 原本的 MOS 分数
    })

    # 4. 保存为 txt 格式（逗号分隔，不要表头和行索引）
    new_df.to_csv(output_txt_path, index=False, header=False, sep=',')
    print(f"处理完成！成功保存至 {output_txt_path}")
    print(f"共处理 {num_rows} 个视频，划分为 {labels[-1] + 1} 个类别。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a two-column MOS CSV to the KSVQE four-column manifest.")
    parser.add_argument("input_csv", help="CSV containing filename and MOS columns")
    parser.add_argument("output_txt", help="Destination four-column manifest")
    parser.add_argument("--group-size", type=int, default=7)
    args = parser.parse_args()
    generate_contrastive_labels(args.input_csv, args.output_txt, args.group_size)
