# KSVQE Reproduction Runbook

本手册记录当前仓库实际可用的训练、验证、跨库测试和推理路径。命令均假设当前目录为仓库根目录；数据放在被 Git 忽略的 `data/` 中。

## 1. Environment

```bash
conda create -n ksvqe python=3.8 -y
conda activate ksvqe
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

检查关键导入：

```bash
python -c "import torch, cv2, decord, timm, einops; print(torch.__version__, torch.cuda.is_available())"
```

当前 `Trainer` 明确使用 CUDA。至少准备一块可用 NVIDIA GPU；论文相近设置在 RTX 4090 上训练显存约 18 GiB，challenge-style 设置约 15 GiB，测试约 11.7 GiB。数值来自项目报告，实际占用随 PyTorch、CUDA、分辨率和 batch size 变化。

## 2. Required Checkpoints

```text
pretrained_weights/
├── clip/ViT-B-16.pt
├── CONTRIQUE_checkpoint25.tar
├── KSVQE_techniqual_pretrainonLSVQ.pth
└── swin_tiny_patch244_window877_kinetics400_1k.pth

handoff_fullstack_inference/final_weights/
└── KSVQE_paper7class_head_val-ltest_s_finetuned.pth
```

代码默认使用仓库相对路径。部署时也可设置：

```bash
export KSVQE_CLIP_WEIGHTS=/path/to/ViT-B-16.pt
export KSVQE_CONTRIQUE_WEIGHTS=/path/to/CONTRIQUE_checkpoint25.tar
export KSVQE_SWIN_WEIGHTS=/path/to/swin_tiny_patch244_window877_kinetics400_1k.pth
```

Windows PowerShell 使用 `$env:KSVQE_CLIP_WEIGHTS = "..."` 形式。

检查 3D Swin checkpoint：

```bash
python check_ksvqe_swin_load.py --mode inspect
```

## 3. Dataset Layout and Manifest

建议目录：

```text
data/
├── KVQ-dataset/{Train,Validation,Test}/
├── LIVE-VQC/videos/
├── KoNViD-1k/videos/
└── YouTube-UGC/original_videos_h264/
```

四列 manifest：

```text
relative/path.mp4,cls_label,dis_label,mos
```

无标签推理也必须保留四列，可填占位值：

```text
uploads/example.mp4,0,0,0
```

从两列 filename/MOS CSV 生成分组标签：

```bash
python labeltotxt.py input_mos.csv output_4col.txt --group-size 7
```

生成固定 seed 的 reference-level 80/20 split：

```bash
python scripts/make_kvq_paper_split.py \
  --dataset-root data/KVQ-dataset \
  --out-dir dataset_csv/KVQ_paper_split \
  --seed 42 \
  --train-ratio 0.8
```

脚本默认每连续七行是一组。请先检查源标注排序；如果官方数据组织变化，不应直接套用。

## 4. Main Configurations

| Config | Purpose |
|---|---|
| `config/Kwai_KSVQE.yml` | challenge-style 7-class training |
| `config/Kwai_KSVQE_paper7class.yml` | reference 80/20 paper-like training |
| `config/Kwai_KSVQE_test.yml` | KVQ validation/inference |
| `config/Kwai_KSVQE_livevqc.yml` | LIVE-VQC evaluation |
| `config/Kwai_KSVQE_konvid.yml` | KoNViD-1k evaluation |
| `config/eval_youtube_7class.yml` | YouTube-UGC with challenge-style model |
| `config/eval_youtube_paper7class.yml` | YouTube-UGC with paper-like model |
| `config/Kwai_KSVQE_no_contrast*.yml` | loss ablation configs |

运行前检查 `anno_file`、`data_prefix`、`load_path` 和 `test_load_path`。跨库 YAML 默认指向保留下来的 paper-like fine-tuned checkpoint；若要复现 challenge-style 表格，必须替换成对应 challenge-style checkpoint。

## 5. Training

单卡 challenge-style：

```bash
python train.py --o config/Kwai_KSVQE.yml --gpu_id 0 -r checkpoint/
```

单卡 paper-like：

```bash
python train.py --o config/Kwai_KSVQE_paper7class.yml --gpu_id 0 -r checkpoint_paper7class/
```

多卡：

```bash
torchrun --nproc_per_node=4 --master_port=3332 train_ddp.py \
  --o config/Kwai_KSVQE.yml \
  --gpu_id 0,1,2,3 \
  -r checkpoint_ddp/
```

配置中的主参数：50 epochs、2.5 warm-up epochs、AdamW、`lr=3e-5`、`wd=0.05`、EMA 0.999。paper-like run 的 `batch_size=8`、`frame_interval=2`；challenge-style 的 `batch_size=4`、`frame_interval=4`。

日志里 normal/EMA 的 `model-n`/`model-s` 显示文案存在历史歧义。选择模型时检查实际 checkpoint 文件名、评测命令和完整 SRCC/PLCC/KRCC/RMSE，不要只根据打印标签判断。

## 6. Validation

```bash
python test.py --o config/Kwai_KSVQE_test.yml --gpu_id 0 --mode val
```

`--mode val` 要求 MOS 有效，输出 SRCC、PLCC、KRCC、RMSE。预测会先做 mean/std 线性对齐；不同数据集 MOS 标度不同，主要比较相关系数。

跨库示例：

```bash
python test.py --o config/Kwai_KSVQE_livevqc.yml --gpu_id 0 --mode val
python test.py --o config/Kwai_KSVQE_konvid.yml --gpu_id 0 --mode val
python test.py --o config/eval_youtube_paper7class.yml --gpu_id 0 --mode val
```

## 7. Unlabeled Inference

复制一份 test YAML，修改：

```yaml
data:
  val:
    args:
      anno_file: path/to/inference_4col.txt
      data_prefix: path/to/video/root
```

运行：

```bash
python test.py --o path/to/inference.yml --gpu_id 0 --mode test
```

输出：

```text
output.txt
```

当前 test 分支把 raw score 裁剪到 `[-2.5, 2.5]` 后线性映射到 `[1, 5]`。这只是展示用启发式标度，不要用它替代带标签的正式验证。

## 8. Common Diagnostics

### Checkpoint keys

加载器会：

- 读取顶层 `state_dict` 或直接参数字典；
- 去掉 `module.`；
- 将 `technical_backbone.*` 映射为 `KSVQE_backbone.*`；
- 将 `technical_head.*` 映射为 `KSVQE_head.*`；
- 打印 missing/unexpected key 数量和样例。

若 missing keys 很多，不要继续训练后假定预训练生效。先检查 checkpoint 来源、2D/3D 类型与模块命名。

### Video decoding

Decord 失败后会回退到 OpenCV。建议部署前：

1. 用 ffprobe 或 OpenCV 检查时长、帧数和 codec；
2. 统一转码到稳定的 H.264/MP4 profile；
3. 单独记录无法解码的视频，不要吞掉异常；
4. 用一条短视频先完成端到端 smoke test。

### OOM

优先减小 batch size；验证 loader 已固定 batch size 1。不要先改 fragment size 或 clip length，否则会改变实验协议。

## 9. Result Provenance

公开表格位于 `docs/results/`。KSVQE 数字大部分由 raw logs 复核；challenge-style KoNViD 日志不完整，因此该行被标为 consolidated-report evidence。BRISQUE 和网页人工失真数字只来源于团队演示，当前源码不能复现。
