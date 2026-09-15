# Project Summary — KSVQE Video Quality Assessment

本文档面向简历撰写与技术面试。内容按“代码可证实”“日志/报告可证实”“团队材料记录但当前源码缺失”三类证据整理，避免把第三方算法或团队其他成员工作写成个人原创。

## 1. 项目一句话简介

基于官方 KVQ/KSVQE 代码复现无参考短视频质量评价模型，完成视频采样与多视图预处理、训练与权重适配、KVQ 及三套公开视频库评测、推理交付和误差分析，并在团队系统中与 BRISQUE 方案进行对比展示。

## 2. 项目背景与目标

短视频质量同时受压缩、噪声、模糊、增强处理、内容语义和时空变化影响，单帧或单一低层统计特征难以完整刻画。项目目标不是提出新的 KSVQE 网络，而是把论文与开源实现落到可训练、可测试、可分析、可交付的工程流程，并回答三个问题：

1. 官方 KSVQE 在现有 GPU 与 KVQ 数据组织下能否稳定复现？
2. 不同训练划分在 KVQ 域内和 LIVE-VQC、KoNViD-1k、YouTube-UGC 跨库场景下表现如何？
3. 深度视频质量模型与传统 BRISQUE 对人工失真的响应是否一致，出现反常趋势时该如何解释？

## 3. 完整技术流程

```text
视频文件 + 四列标注
  → Decord 解码（失败时 OpenCV fallback）
  → UnifiedFrameSampler 统一时序采样
  → CLIP resize view + 9×9 technical fragments + original fragments
  → CLIP 语义分支 / CONTRIQUE 失真分支 / 3D Swin 技术分支
  → 语义与失真特征交互、调制和融合
  → VQAHead 回归视频质量分数
  → 有标签：SRCC / PLCC / KRCC / RMSE
  → 无标签：裁剪并线性映射到 [1, 5]，写入 output.txt
```

真实调用链为：

```text
train.py 或 test.py
  → trainer.py::Trainer
  → datasets.ViewDecompositionDataset_KVQ
  → models.model.VQA_Network
  → models.backbones.KSVQE_model.KSVQE
  → VQAHead
```

前端部分在团队演示中通过“上传视频—后端生成四列 manifest—调用 `test.py`—读取 `output.txt`—展示曲线/报告”的方式衔接。当前工作区只保留了 KSVQE 推理交接包、结题展示和结果截图，没有前端、后端接口或 BRISQUE 源码，因此无法从代码进一步确认路由、任务队列或进程调用细节。

## 4. 模型与算法

### KSVQE

本项目调用的是第三方 KSVQE 模型，不是个人原创网络。实际代码包含：

- **Semantic branch**：CLIP ViT-B/16 提取图像语义特征，并在若干关键帧上执行质量相关区域选择。
- **Distortion representation**：冻结的 CONTRIQUE 编码器提取失真表征，训练时参与失真对比损失。
- **Technical branch**：3D Swin Transformer 对时空 fragments 建模。
- **Feature interaction**：语义、失真与技术特征通过 cross/self attention、adapter 和 modulation 模块交互。
- **Regression head**：将融合后的 768 维特征映射为质量分数。

主训练目标为：

```text
L = L_PLCC + 0.3 × L_distortion_contrastive + 0.0 × L_rank
```

优化器使用 AdamW，学习率 `3e-5`、权重衰减 `0.05`，配置 2.5 epoch warm-up、cosine decay 和 `0.999` EMA。

### BRISQUE

团队材料记录的 BRISQUE 实验为：使用 `pyiqa` 对均匀抽取的视频帧计算 BRISQUE 特征/分数，对视频内帧结果做平均，并使用 scikit-learn RBF-SVR 拟合视频 MOS。该实现源码和原始日志未保存在当前仓库，因此只能把它当作团队实验记录，不能声称当前代码可复现。

## 5. 数据处理流程

### 标注格式

每行四列：

```text
video_path,cls_label,dis_label,mos
```

`cls_label`/`dis_label` 被送入 KSVQE 的失真对比学习分支，`mos` 用于质量回归和验证指标。

### 抽帧与输入

- 主入口使用 Decord，按全部 clip 的索引并集一次性解码，避免重复读取同一帧。
- Decord 异常时切换 OpenCV；本地日志中确实出现过 H.264 解码错误。
- 主实验每个 clip 取 32 帧；challenge-style 间隔为 4，paper-like 间隔为 2。
- 训练每个视频取 1 个 clip，验证取 3 个 clip 并平均预测。
- technical view 使用 `9 × 9` 个 `32 × 32` fragments，最终空间尺寸为 `288 × 288`。
- CLIP view 使用 CLIP 均值方差归一化；technical fragment 使用 ImageNet 风格的像素均值方差。
- 数据项还保留 `ori_fragment`、`frame_inds`、`original_shape`、`dis_label` 等，供不同分支使用。

### KVQ 划分

paper-like 脚本合并 Train/Validation/Test，假设每连续 7 行属于同一参考内容，以位置 `0..6` 生成近似 7 类标签，再以 seed 42 按参考组随机划分 80%/20%，得到 480/120 个参考组。由于官方处理工作流标签缺失，这只是“original + six positions/QP-like bins”的近似，不是精确的官方失真分类。

### 人工失真

团队网页材料明确出现过 compression、blur、sharpen、noise、fog、brightness/saturation 控件及不同 level；保留下来的示例 level 包括 compression 10、blur 12、sharpen 10、fog 50/100、noise 30。当前工作区没有生成这些失真的代码，因此不能确认核大小、噪声方差、雾化公式、颜色空间、编码器参数或 level 的物理单位。

`scripts/analyze_kvq_distortions.py` 也不是人工失真生成器。它分析 KVQ 已有七视频组的文件大小和抽帧统计，包括亮度、标准差、Laplacian 方差、高通标准差、Sobel 复杂度、熵、色彩度和块效应，然后执行内部 z-score/PCA/K-means 流程。

## 6. 实验设计

### KSVQE 两套协议

1. **Challenge split / 7-class approximation**：沿用 KVQ challenge 的训练验证组织，batch size 4，32 帧，interval 4。
2. **Reference 80/20 / paper-like 7-class**：以参考内容为组重建固定 seed 的 80/20 划分，batch size 8，32 帧，interval 2。

两套模型分别在 KVQ、LIVE-VQC、KoNViD-1k、YouTube-UGC 上测试，比较域内精度与跨库泛化。

### BRISQUE 报告实验

团队材料记录了 LIVE-VQC 官方验证子集 117 个视频上的三组实验：默认抽帧/default SVR、固定 32 帧/default SVR、固定 32 帧/GridSearchCV。当前仓库没有对应代码和 split 文件，结果只用于说明团队对比过程。

### 评价指标

- **SRCC**：排序一致性，适合衡量质量预测的单调关系。
- **PLCC**：线性相关性；代码在验证时先将预测均值/标准差线性对齐到 MOS。
- **KRCC**：另一种秩相关指标，对成对次序更直观。
- **RMSE**：对齐后预测与 MOS 的绝对误差。

不同数据库 MOS 标度不同，因此不能直接横向比较 RMSE 数值；跨库分析主要看相关系数。

## 7. 最重要的实验结果

| 训练协议 | 数据集 | SRCC | PLCC | KRCC | RMSE |
|---|---|---:|---:|---:|---:|
| Challenge split | KVQ | **0.8657** | **0.8679** | 0.6793 | 0.3063 |
| Challenge split | LIVE-VQC | 0.5010 | 0.5372 | 0.3509 | 16.4111 |
| Challenge split | KoNViD-1k | 0.4667 | 0.4789 | — | — |
| Challenge split | YouTube-UGC | 0.6383 | 0.6357 | 0.4525 | 0.5524 |
| Reference 80/20 | KVQ | 0.8372 | 0.8456 | 0.6471 | 0.3170 |
| Reference 80/20 | LIVE-VQC | **0.5859** | **0.5988** | 0.4117 | 15.2799 |
| Reference 80/20 | KoNViD-1k | **0.5432** | **0.5601** | 0.3822 | 0.6011 |
| Reference 80/20 | YouTube-UGC | **0.7038** | **0.7110** | 0.5099 | 0.4920 |

主要结论：

- Challenge split 在 KVQ 上达到 `0.8657/0.8679` SRCC/PLCC，接近论文报告的 `0.867/0.869`。
- Reference 80/20 相比 challenge split，KVQ SRCC/PLCC 分别下降 `0.0285/0.0223`，但 LIVE-VQC 提升 `0.0849/0.0616`、KoNViD-1k 提升 `0.0765/0.0812`、YouTube-UGC 提升 `0.0654/0.0754`。
- 该对比同时改变了数据划分、frame interval 和 batch size，不能把跨库提升单独归因于某一个因素。
- 训练曲线中 challenge-style run 约在 epoch 28 达到 `SRCC 0.8677 / PLCC 0.8629`，paper-like run 约在 epoch 16 达到 `0.8361 / 0.8465`；后续回落提示需要 early stopping/checkpoint selection。
- KoNViD challenge-split 的原始日志只运行到 16/1200，`0.4667/0.4789` 来自实验汇总而非完整原始日志，公开结果表已标注。

## 8. 有代表性的异常现象与踩坑

### 8.1 KSVQE 对 blur/fog 的演示趋势与直觉相反

同一条团队演示视频中：

- clean：KSVQE 45.7，BRISQUE-quality 47.6；
- blur 12：KSVQE 47.9（+2.2），BRISQUE-quality 41.2（-6.4）；
- fog 100：KSVQE 52.7（+7.0），BRISQUE-quality 30.5（-17.1）。

BRISQUE 的显示分数按材料说明由原生“越低越好”的失真分反向映射为“越高越好”的质量分，所以其下降方向符合直觉。KSVQE 升高则可能来自：

1. CLIP 语义/显著区域分支在模糊或雾化后改变了区域选择；
2. 训练时的处理工作流与网页合成失真分布不一致；
3. 原始回归输出到网页 0–100 分的校准可能放大或改变局部差异；
4. 单视频内容、曝光和纹理结构造成偶然响应；
5. 模型同时衡量语义内容与技术质量，不等价于单纯锐度检测器。

这些只能作为解释假设。因为只有一条视频的结果、且缺少失真代码和 raw scores，不能下结论说“KSVQE 普遍偏爱强雾或模糊”。正确的后续实验应固定编码链，在多视频、多 seed、多强度下记录原始模型输出并做置信区间和单调性统计。

### 8.2 权重结构与键名不一致

预训练/微调权重可能包含 `state_dict`，也可能直接是参数字典；DP/DDP 权重还可能带 `module.`。旧版键名为 `technical_backbone`/`technical_head`，当前网络用 `KSVQE_backbone`/`KSVQE_head`。解决方式是在加载前统一剥离前缀并重映射键名，打印 missing/unexpected keys 抽样检查，而不是只依赖 `strict=False` 静默通过。

### 8.3 2D/3D Swin 权重混用

调试脚本专门区分 2D inflation 与 3D checkpoint loading。最终主干使用 `swin_tiny_patch244_window877_kinetics400_1k.pth`，并保留 checkpoint inspection 工具验证结构。

### 8.4 CLIP 权重“存在但未被使用”

旧代码默认调用下载函数并从用户缓存读取，即使仓库已有 `ViT-B-16.pt`。现已改为优先读取仓库相对路径，并支持 `KSVQE_CLIP_WEIGHTS` 覆盖；不存在时才下载官方文件。

### 8.5 服务器绝对路径和跨平台问题

原配置混有 `/data/wyr/...`、上游 `/data2/luyt/...` 和本地 Windows 路径。已把公开代码、脚本和主要 YAML 改成仓库相对路径，数据统一放在被 Git 忽略的 `data/` 下。

### 8.6 视频解码不稳定

LIVE-VQC 日志出现过 H.264 解码错误。代码提供 OpenCV fallback，但异常分支会把整段视频读入内存，并对短视频重复最后一帧；空视频仍可能失败。面试中应承认这仍是可改进点：需要显式校验帧数、区分可恢复/不可恢复错误、记录失败样本并避免 silent padding。

### 8.7 指标和 checkpoint 标签易混淆

`train.py` 继承的打印文案把两个 best tuple 标成 `model-n`/`model-s`，与 normal/EMA 的实际变量名不够直观；而每个指标的 best 又可能来自不同 epoch。汇报时必须以 checkpoint 文件、评测日志和完整 metric tuple 为准，不能只截取控制台“best”字样。

### 8.8 小验证集调参反而降低 PLCC

团队 BRISQUE 实验中，32 帧 + GridSearchCV 的 PLCC 从默认 SVR 的 `0.5476` 降到 `0.4848`，而 SRCC 基本不变（`0.5108`→`0.5110`）。可能是 117 视频规模下超参数选择不稳定、优化目标与 PLCC 不完全一致或划分泄漏/方差问题。没有代码和 fold 记录，不能进一步定性。

## 9. 我个人的实际贡献

根据个人报告和团队分工，可稳妥表述为：

- 阅读 KSVQE/KVQ 论文与官方代码，拆解语义、失真、3D 时空建模和回归头之间的调用关系。
- 在 RTX 4090 / Python 3.8 环境中完成依赖、数据路径、训练配置与 GPU 显存适配，跑通 KSVQE 训练、验证和测试。
- 处理预训练与微调 checkpoint 的包装格式、DP/DDP 前缀和历史键名不一致问题，并验证 2D/3D Swin、CLIP、CONTRIQUE 权重的实际加载情况。
- 构建/整理四列视频 manifest、7 类近似标签和固定 seed 的 reference-level 80/20 split，完成 KVQ 域内与三套外部数据库的评测。
- 对两套训练协议、normal/EMA checkpoint、训练曲线和跨库泛化差异做结果核验与误差分析。
- 整理 `handoff_fullstack_inference` 推理交接材料，使团队后端可以按 manifest → `test.py` → `output.txt` 方式集成模型。

不应写成个人贡献的部分：BRISQUE 模块开发与调参、网页前后端、AI 评语接口、整体 UI 设计。它们属于团队项目背景，个人面试时应明确分工。

## 10. 技术栈

### 当前源码真实使用

- **Language / runtime**：Python 3.8、CUDA、PowerShell/Bash
- **Deep learning**：PyTorch、torchvision、timm、einops、THOP
- **Vision/video**：OpenCV、eva-decord（以 `decord` 导入）、PIL、scikit-video
- **Data/science**：NumPy、SciPy、pandas、Matplotlib
- **Model/token utilities**：PyYAML、tqdm、ftfy、regex
- **Legacy auxiliary path**：PyTorchVideo / SlowFast（`SlowFast_features.py`，非 KSVQE 主流程）
- **Engineering**：YAML configuration、Git LFS、CLI scripts

### 团队材料中出现但当前源码缺失

- `pyiqa` BRISQUE
- scikit-learn `SVR` / `GridSearchCV`
- Web UI 与后端推理调用（具体 Flask/Streamlit/Gradio/FastAPI 框架无法由文件证明）

### 未发现真实使用证据

TensorBoard、Flask、Streamlit、Gradio。不要仅为丰富简历而列入。

## 11. 可写进简历的 4 条 bullet

- 基于 PyTorch 复现 KSVQE 无参考视频质量评价模型，完成 CLIP 语义分支、CONTRIQUE 失真表征与 3D Swin 时空分支的代码梳理、预训练权重适配及端到端训练/推理流程落地。
- 搭建 Decord/OpenCV 视频解码与多视图采样 pipeline，按 32 帧 clip 构造 CLIP resize view 和 9×9 spatial fragments，并建立四列标注、7 类近似标签及固定 seed 的 reference-level 数据划分。
- 在 KVQ 上取得 `0.8657 SRCC / 0.8679 PLCC`，并完成 LIVE-VQC、KoNViD-1k、YouTube-UGC 跨库评测；分析另一数据划分带来的域内下降与跨库 SRCC `+0.0654～+0.0849` 的变化及其混杂因素。
- 排查并修复 checkpoint 包装、DP/DDP 前缀、历史模块键名、2D/3D Swin 和 CLIP 本地权重未命中等问题，整理可移植配置、推理交接包和 Git-LFS 权重管理方案。

如果岗位特别强调系统集成，可以在不冒领的前提下补充：“与团队协作将 KSVQE 推理以 manifest → CLI → result file 方式交付给网页后端，并参与 KSVQE/BRISQUE 差异结果分析。”

## 12–13. 面试官最可能追问的 15 个问题与回答思路

### Q1. 你复现了 KSVQE 的哪些部分，哪些是原作者工作？

回答思路：KSVQE 网络思想和主干代码来自论文/官方仓库；自己的工作是读懂分支、适配数据与配置、权重加载、训练评测、跨库实验、调试和推理交付，不声称提出网络。

### Q2. KSVQE 为什么需要 CLIP、CONTRIQUE 和 3D Swin 三类特征？

回答思路：CLIP提供高层语义与显著内容线索；CONTRIQUE偏低层自然图像/失真表征；3D Swin建模 fragments 的空间与时间变化。质量评价既与内容相关，也与技术失真和时序一致性相关。

### Q3. 一个视频如何变成模型输入？

回答思路：从四列 manifest 取路径；统一采样 32 帧 clip；Decord 按索引并集解码；生成 CLIP resize view、288×288 technical fragment view 与 original fragment；分别归一化后连同帧索引、失真标签送入网络。

### Q4. 9×9 fragments 是什么，为什么不用整帧缩放？

回答思路：从不同空间网格抽局部块并拼接，尽量保留局部失真而降低整帧高分辨率计算量；整帧缩放可能抹平压缩块、噪声和锐度细节。语义分支仍用 resize view 补充全局信息。

### Q5. 训练损失为什么使用 PLCC loss 和 contrastive loss？

回答思路：PLCC loss直接优化预测与 MOS 的线性相关；失真对比损失利用近似失真标签约束表征。主配置权重是 1 与 0.3；rank loss 做成可配置但主实验为 0。

### Q6. 为什么同时看 SRCC 和 PLCC？

回答思路：SRCC关心排序，适合质量评价的单调关系；PLCC关心线性拟合。一个模型可能排序正确但尺度不准，所以二者需要同时报告；KRCC/RMSE补充成对次序和绝对误差。

### Q7. 验证时为什么要对预测做线性对齐？

回答思路：网络 raw score 与不同数据库 MOS 的均值/方差未必同尺度。代码用 mean/std 线性对齐后算 PLCC/RMSE；SRCC 对单调线性变换不敏感。要说明这不是 test 模式 [1,5] 映射。

### Q8. paper-like split 为什么不是严格论文复现？

回答思路：缺少官方处理工作流标签，只能按每7行的位置近似赋类；同时重建了 reference-level 80/20 split，且 frame interval/batch size 变化。因此它用于受控程度有限的协议比较，不应标成 exact reproduction。

### Q9. 为什么域内性能下降、跨库性能反而提升？

回答思路：reference-level split 减少同内容泄漏/记忆的可能性，可能促使泛化；更密集的时间采样也可能有帮助。但多因素同时改变，所以只能报告相关现象，下一步要逐变量 ablation。

### Q10. checkpoint 加载遇到过什么问题？

回答思路：state dict 包装不同、`module.` 前缀、旧模块名映射、2D/3D Swin不兼容、CLIP从缓存而非仓库读取。通过结构检查、key normalization、missing/unexpected 抽样和文件 hash 确认实际加载。

### Q11. EMA 起什么作用？

回答思路：以 0.999 对参数做指数滑动平均，降低单步更新噪声，通常验证更稳定；但当前日志命名容易混淆，选择 checkpoint 时按文件和完整指标核验。

### Q12. BRISQUE 与 KSVQE 的本质差别是什么？

回答思路：BRISQUE基于自然场景统计，通常逐帧、偏低层且不直接建模语义/时间；KSVQE结合语义、失真与3D时空特征。展示分数还做了方向反转，比较前必须统一“越高越好”的定义。

### Q13. 怎么解释 blur/fog 后 KSVQE 分数反而上升？

回答思路：先限定为单视频现象；给出分布外合成失真、语义区域选择、校准与内容依赖等假设；然后提出多视频多强度、raw output、置信区间和单调性检验，而不是强行解释为模型规律。

### Q14. 如果把项目部署成服务，你会改什么？

回答思路：把 CLI 包装成常驻 GPU worker，避免每请求加载模型；上传后生成 manifest；增加转码/解码校验、批处理、超时与失败队列；输出 raw/calibrated score 和模型版本；用异步任务给前端轮询。

### Q15. 你会怎样继续改进实验可信度？

回答思路：锁定同一 reference split；只改变一个变量；保存每视频 raw prediction；至少3个 seed；报告均值/方差与置信区间；使用标准非线性 logistic mapping；检查内容泄漏；补齐 BRISQUE 源码、失真生成参数和多视频强度曲线。

## 14. 与“视觉大模型 / 多模态算法岗位”的关联

### 直接相关

- 使用 CLIP ViT 作为语义视觉编码器，理解预训练视觉表征如何服务下游质量任务。
- 将语义 token、局部质量区域、低层失真表征与 3D 时空特征进行 cross-attention / modulation / fusion。
- 处理多来源预训练权重、冻结/微调策略、分支学习率和 checkpoint 兼容问题。
- 分析语义表征与传统低层质量特征在分布外人工失真上的冲突，这与多模态/大模型下游可靠性分析相通。

### 更偏传统 CV / 视频算法

- 视频解码、抽帧、clip 采样、局部 fragment 拼接、归一化与显存控制。
- 3D Swin 时空建模、无参考 VQA、MOS 回归和跨数据库泛化。
- BRISQUE 自然场景统计、OpenCV 图像统计、人工失真与质量曲线。
- SRCC/PLCC/KRCC/RMSE、数据划分、early stopping 与误差分析。

### 不应过度包装

该项目没有训练通用视觉大模型，也没有文本-视频指令数据、生成式模型、LLM agent、VLM 对话或多模态检索。更准确的定位是：**预训练视觉模型在视频质量任务中的融合与微调 + 扎实的视频/CV工程和实验能力**。
