import pandas as pd


def convert_konvid_to_txt(csv_filepath, txt_filepath):
    print(f"正在读取 {csv_filepath} ...")

    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"报错：找不到文件 {csv_filepath}")
        return

    print(f"\n正在生成 {txt_filepath} ...")

    with open(txt_filepath, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            # ========== 核心修复 ==========
            # 先转成 int 去掉自动生成的 .0，再转成字符串拼接后缀
            vid_name = str(int(row['flickr_id'])) + ".mp4"
            # ==============================

            mos_score = float(row['mos'])
            line = f"{vid_name},0,0,{mos_score:.9f}\n"
            f.write(line)

    print(f"\n🎉 转换完成！请检查 txt 中是否已经没有 .0 了！")


if __name__ == '__main__':
    input_csv = 'KoNViD_1k_mos.csv'  # 替换成你的csv路径
    output_txt = 'test_konvid_1k.txt'  # 替换成你要生成的txt路径
    convert_konvid_to_txt(input_csv, output_txt)