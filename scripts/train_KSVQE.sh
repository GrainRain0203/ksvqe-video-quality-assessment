
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p log checkpoint
nohup python -u train.py --o config/Kwai_KSVQE.yml -r checkpoint/ --gpu_id 0,1 > log/Kwai_KSVQE.log 2>&1 &
