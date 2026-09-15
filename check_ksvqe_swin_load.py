import argparse
import traceback
from pathlib import Path

import torch

from models.backbones.KSVQE_model import KSVQE

REPO_ROOT = Path(__file__).resolve().parent


def inspect_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    print(f"checkpoint type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"top-level keys: {list(ckpt.keys())}")
        for candidate in ("state_dict", "model"):
            if candidate in ckpt and isinstance(ckpt[candidate], dict):
                keys = list(ckpt[candidate].keys())
                print(f"{candidate} key count: {len(keys)}")
                print(f"{candidate} first keys: {keys[:10]}")
    return ckpt


def try_build(path, pretrained2d):
    print("\n" + "=" * 80)
    print(f"Trying KSVQE(pretrained={path!r}, pretrained2d={pretrained2d})")
    print("=" * 80)
    try:
        model = KSVQE(
            pretrained=path,
            pretrained2d=pretrained2d,
            num_samples=1,
            CLIP_location=8,
            cls_use=True,
            tuning_stage=1,
            a1=1,
            a2=2,
            frozen_stages=-1,
        )
        print("SUCCESS: KSVQE instantiated.")
        print(f"parameter count: {sum(p.numel() for p in model.parameters())}")
    except Exception:
        print("FAILED:")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--swin",
        default=str(REPO_ROOT / "pretrained_weights" / "swin_tiny_patch244_window877_kinetics400_1k.pth"),
        help="Path to the Swin checkpoint.",
    )
    parser.add_argument(
        "--mode",
        choices=["inspect", "3d", "2d", "both"],
        default="both",
        help="inspect only reads keys; 3d uses load_swin; 2d uses inflate_weights.",
    )
    args = parser.parse_args()

    inspect_checkpoint(args.swin)

    if args.mode in ("3d", "both"):
        try_build(args.swin, pretrained2d=False)
    if args.mode in ("2d", "both"):
        try_build(args.swin, pretrained2d=True)


if __name__ == "__main__":
    main()
