# KSVQE 实验指标汇总

本文档用于展示实验结果，重点记录本项目两个主要复现模型在 KVQ 本库测试和跨库测试中的指标。

## 1. 命名说明

先说明一下 7 类标签的来源：KSVQE 原论文中有一个失真对比学习分支，这个分支需要知道“当前视频属于哪一种处理/失真模式”来做对比学习。但作者开源代码和公开数据中没有给出清晰、可直接使用的对比学习类别标签，也没有完整公开论文中可能使用的 processing pattern 标签。因此，本项目不是随便弄了 7 个标签，而是在公开信息有限的情况下，为了让对比学习分支能够正常训练，自己构造了“原始视频 + 6 个 QP bin”的 7 类近似失真标签。


`7class` 指使用 7 类近似失真标签训练的复现模型。这里的 7 类由“原始视频 + 6 个 QP bin”构成，用来近似表示不同失真/压缩程度；训练划分采用原 challenge split，主要配置为 batch size 4、frame interval 4。

`paper7class` 指仍然使用同一套 7 类近似失真标签，但训练划分和部分训练配置更接近论文设置：按 reference content 重新做 80/20 划分，batch size 调整为 8，frame interval 调整为 2。名字里的 `paper` 只表示“paper-like setting”，即更接近论文口径，并不等同于论文官方完整设置；名字里的 `7class` 仍然表示使用 7 类近似失真标签。

需要注意：KSVQE reported 使用的是官方 processing pattern 标签及作者原始训练协议，而本项目的 `7class` 和 `paper7class` 使用的是公开信息下构造的近似 7 类标签，因此跨库结果与论文报告结果存在差距是正常的。

## 2. 主表：SRCC / PLCC

SRCC 反映预测质量排序与真实 MOS 排序的一致性，PLCC 反映预测分数与真实 MOS 的线性相关性，二者都是越高越好。

| 模型版本 | 失真标签 | 训练协议 | KVQ 本库测试 | LIVE-VQC | KoNViD-1k | YouTube-UGC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 7class | 原始视频 + 6 个 QP bin | challenge split，batch size 4，frame interval 4 | 0.8657 / 0.8679 | 0.5010 / 0.5372 | 0.4667 / 0.4789 | 0.6383 / 0.6357 |
| paper7class | 原始视频 + 6 个 QP bin | reference 80/20 split，batch size 8，frame interval 2 | 0.8372 / 0.8456 | 0.5859 / 0.5988 | 0.5432 / 0.5601 | 0.7038 / 0.7110 |
| KSVQE reported | 官方 processing pattern 标签 | 论文原始设置 | 0.867 / 0.869 | 0.720 / 0.768 | 0.650 / 0.661 | 0.742 / 0.764 |

表中每个单元格格式为 `SRCC / PLCC`。

## 3. 本库测试详细指标

本库测试指在 KVQ 数据集内进行训练/测试划分后的评估。`7class` 对应原 challenge split 的 KVQ test；`paper7class` 对应按 reference content 重划分后的 KVQ paper split。

| 模型版本 | 测试集 | SRCC | PLCC | KRCC | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 7class | KVQ test | 0.8657 | 0.8679 | 0.6793 | 0.3063 |
| paper7class | KVQ paper split | 0.8372 | 0.8456 | 0.6471 | 0.3170 |

本库测试中，`7class` 的 SRCC/PLCC 更高，说明原 challenge split 下模型能较好学习 KVQ 数据分布。`paper7class` 本库指标略低，但它的训练划分更接近论文 reference-level 评估口径，因此更适合与跨库泛化结果一起观察。

## 4. 跨库测试详细指标

跨库测试指使用在 KVQ 上训练得到的模型，直接在外部视频质量评价数据集上测试，包括 LIVE-VQC、KoNViD-1k 和 YouTube-UGC。

| 模型版本 | 测试集 | SRCC | PLCC | KRCC | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 7class | LIVE-VQC | 0.5010 | 0.5372 | 0.3509 | 16.4111 |
| 7class | KoNViD-1k | 0.4667 | 0.4789 | 0.3247 | 0.6542 |
| 7class | YouTube-UGC | 0.6383 | 0.6357 | 0.4525 | 0.5524 |
| paper7class | LIVE-VQC | 0.5859 | 0.5988 | 0.4117 | 15.2799 |
| paper7class | KoNViD-1k | 0.5432 | 0.5601 | 0.3822 | 0.6011 |
| paper7class | YouTube-UGC | 0.7038 | 0.7110 | 0.5099 | 0.4920 |

跨库结果中，`paper7class` 在 LIVE-VQC、KoNViD-1k、YouTube-UGC 三个外部数据集上均优于 `7class`，说明更接近论文的 reference-level 划分和采样配置提升了当前记录下的跨库泛化能力。

## 5. 相对变化汇总

下表展示 `paper7class` 相比 `7class` 的 SRCC/PLCC 变化。

| 测试集 | SRCC 变化 | PLCC 变化 | 结论 |
| --- | ---: | ---: | --- |
| KVQ 本库 | -0.0285 | -0.0223 | 本库指标略低 |
| LIVE-VQC | +0.0849 | +0.0616 | 跨库提升明显 |
| KoNViD-1k | +0.0765 | +0.0812 | 跨库提升明显 |
| YouTube-UGC | +0.0654 | +0.0754 | 跨库提升明显 |

## 6. 结论

- `7class` 在 KVQ 本库测试上取得最高复现指标，SRCC/PLCC 达到 0.8657/0.8679，说明当前复现流程能够在 KVQ 数据分布内学习到较稳定的质量排序关系。
- `paper7class` 的 KVQ 本库指标略低，但在 LIVE-VQC、KoNViD-1k 和 YouTube-UGC 上均优于 `7class`，说明 paper-like 配置更有利于当前跨库泛化。
- 与 KSVQE reported 相比，本项目复现结果仍存在差距，主要原因可能包括失真标签定义不同、数据划分不完全一致、训练资源和 checkpoint 选择策略不同。
- 跨库展示时建议优先讲 SRCC 和 PLCC；RMSE 会受到不同数据集 MOS 分数尺度影响，例如 LIVE-VQC 是 0-100 分量级，而 KVQ、KoNViD-1k、YouTube-UGC 更接近 1-5 或归一化量级，不适合直接横向比较。
