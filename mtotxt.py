import scipy.io as sio


def live_vqc_mat_to_txt(mat_filepath, txt_filepath):
    print(f"正在读取 {mat_filepath} ...")
    try:
        # 1. 加载 MATLAB 格式的 .mat 文件
        mat_data = sio.loadmat(mat_filepath)
    except FileNotFoundError:
        print(f"报错：找不到文件 {mat_filepath}，请确认路径是否正确！")
        return

    try:
        # 2. 提取视频名和 MOS 分数
        # 注意：MATLAB 的 cell 数组到 Python 里会有多层嵌套，需要 [0][0] 来解包提取纯字符串和浮点数
        video_names = [item[0][0] for item in mat_data['video_list']]
        mos_scores = [item[0] for item in mat_data['mos']]
    except KeyError:
        print("报错：文件内容格式不对，找不到 'video_list' 或 'mos'，请确认这是官方的 data.mat！")
        return

    # 3. 按照你的 DataLoader 格式 (四列: path, 0, 0, mos) 写入 txt
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        for name, score in zip(video_names, mos_scores):
            # 格式化：视频名, 对比标签1(0), 对比标签2(0), MOS分(保留9位小数以对齐你之前的格式)
            line = f"{name},0,0,{score:.9f}\n"
            f.write(line)

    # 4. 打印成功信息和预览
    print(f"\n✅ 转换成功！共处理了 {len(video_names)} 个视频。")
    print(f"✅ 文件已保存为: {txt_filepath}\n")

    print("--- 转换后的 txt 文件前 5 行预览 ---")
    with open(txt_filepath, 'r') as f:
        for _ in range(5):
            print(f.readline().strip())


# ==========================================
# 在这里修改你的文件路径
# ==========================================
if __name__ == '__main__':
    # 你下载的 LIVE-VQC 官方 mat 文件的路径
    input_mat_file = 'data.mat'

    # 你想要生成的测试集 txt 标注文件名称
    output_txt_file = 'test_live_vqc.txt'

    live_vqc_mat_to_txt(input_mat_file, output_txt_file)