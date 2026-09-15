# Checkpoints

Model files in this directory are tracked with Git LFS. They are not bundled through ordinary Git objects.

| File | Size (MiB) | Purpose |
|---|---:|---|
| `clip/ViT-B-16.pt` | 334.58 | CLIP ViT-B/16 semantic backbone |
| `CONTRIQUE_checkpoint25.tar` | 107.02 | CONTRIQUE distortion representation |
| `KSVQE_techniqual_pretrainonLSVQ.pth` | 121.43 | KSVQE technical branch pretrained on LSVQ |
| `swin_tiny_patch244_window877_kinetics400_1k.pth` | 121.51 | 3D Swin initialization used by KSVQE |
| `swin_tiny_patch4_window7_224.pth` | 109.05 | 2D Swin-Tiny initialization retained for legacy experiments |
| `swin_base_patch4_window12_384.pth` | 348.44 | Swin-Base initialization retained for legacy experiments |

The fine-tuned project checkpoint is kept at `handoff_fullstack_inference/final_weights/KSVQE_paper7class_head_val-ltest_s_finetuned.pth` (626.18 MiB). Duplicate copies inside the handoff bundle are ignored; the canonical pretrained copies above remain tracked.

Before redistributing third-party weights, confirm that each upstream model's license and terms permit redistribution. If not, remove the corresponding LFS pointer from the public remote and replace it with an official download instruction; do not silently relabel third-party weights as project-authored assets.
