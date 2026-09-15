import pandas as pd


def convert_csv_to_txt(csv_filepath, txt_filepath):
    print(f"正在读取 {csv_filepath} ...")

    # 1. 读取 CSV 文件
    try:
        # 假设你的文件名是 original_videos_MOS_for_YouTube_UGC_dataset.xlsx
        # 指定 sheet_name='MOS' 因为真实的分数在这个特定的表里
        df = pd.read_excel(csv_filepath, sheet_name='MOS')
    except FileNotFoundError:
        print(f"报错：找不到文件 {csv_filepath}")
        return

    # 2. 提取所有不重复的视频类别 (category)
    # 并自动生成 类别名 -> 数字ID 的映射字典
    unique_categories = df['category'].unique()
    category_to_id = {cat: idx for idx, cat in enumerate(unique_categories)}

    print(f"\n✅ 成功提取到 {len(unique_categories)} 个视频类别，映射关系如下：")
    for cat, idx in category_to_id.items():
        print(f"  - {cat} : {idx}")

    # 3. 逐行提取所需数据并写入 txt
    print(f"\n正在生成 {txt_filepath} ...")
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            # 获取视频名称 (YouTube-UGC的视频通常是.mkv或.mp4，这里统一加上.mp4作为示范)
            vid_name = str(row['vid']) + ".mp4"

            # 获取对应的类别 ID
            cat_id = category_to_id[row['category']]

            # 获取整个视频的真实 MOS 分 (对应 'MOS full' 列)
            mos_score = float(row['MOS full'])

            # 格式化为四列: 视频名, 类别ID, 类别ID, MOS分 (保留9位小数对齐你的格式)
            line = f"{vid_name},{cat_id},{cat_id},{mos_score:.9f}\n"
            f.write(line)

    print(f"\n🎉 转换完成！共处理了 {len(df)} 个视频。")
    print(f"📁 文件已保存为: {txt_filepath}")

    # 预览前5行
    print("\n--- 转换后的 txt 文件前 5 行预览 ---")
    with open(txt_filepath, 'r') as f:
        for _ in range(5):
            print(f.readline().strip())


# ==========================================
# 运行配置区
# ==========================================
if __name__ == '__main__':
    # 你的源 CSV 文件名 (根据你上传的文件，原名应该是下面这个)
    input_csv = 'original_videos_MOS_for_YouTube_UGC_dataset.xlsx'

    # 你想生成的测试集 TXT 文件名
    output_txt = 'test_youtube_ugc.txt'

    convert_csv_to_txt(input_csv, output_txt)