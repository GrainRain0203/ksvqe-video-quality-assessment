


#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p log checkpoint_ddp
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 --master_port=3332 train_ddp.py --o config/Kwai_KSVQE.yml --gpu_id 0,1,2,3 -r checkpoint_ddp/ > log/Kwai_KSVQE_ddp_loadpretrained.log 2>&1 &
