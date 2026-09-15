# KSVQE Inference Handoff

该目录保留最终微调权重，供服务端推理使用。原先复制在本目录中的源码、配置和辅助预训练权重与仓库根目录完全重复，公开 Git 仓库通过 `.gitignore` 排除这些副本；本地文件未删除。

## Canonical files

- 推理代码：仓库根目录 `test.py`、`trainer.py`、`models/`、`datasets/`
- 配置：`config/Kwai_KSVQE_test.yml`
- 依赖：`requirements.txt`
- 辅助权重：`pretrained_weights/`
- 最终权重：`handoff_fullstack_inference/final_weights/KSVQE_paper7class_head_val-ltest_s_finetuned.pth`

## Input manifest

每行必须有四个英文逗号分隔字段：

```text
relative/video.mp4,cls_label,dis_label,mos
```

无标签视频可使用占位值：

```text
uploads/example.mp4,0,0,0
```

## Run

从仓库根目录执行：

```bash
python test.py --o config/Kwai_KSVQE_test.yml --gpu_id 0 --mode test
```

先在 YAML 中把 `data.val.args.anno_file` 和 `data.val.args.data_prefix` 指向服务端生成的 manifest 与上传目录。结果写入 `output.txt`，格式为视频名和映射到 `[1,5]` 的展示分数。

生产服务建议使用常驻 GPU worker 预加载模型，而不是每个 HTTP 请求启动一次 Python 进程。还应加入上传格式校验、转码、超时、任务隔离、错误日志和模型版本字段。
