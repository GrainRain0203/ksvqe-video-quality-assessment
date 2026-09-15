
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python test.py --o config/Kwai_KSVQE_test.yml --gpu_id 0 --mode val
