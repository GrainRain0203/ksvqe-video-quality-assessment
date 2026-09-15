import argparse
import hashlib
import os
import re
from pathlib import Path


CLIP_MODEL_NAME = "ViT-B/16"
CLIP_FILENAME = "ViT-B-16.pt"
CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_contains_clip_loader(root):
    clip_py = root / "models" / "backbones" / "clip" / "clip.py"
    clip_backbone_py = root / "models" / "backbones" / "CLIP_backbone.py"
    result = {
        "clip_py": clip_py,
        "clip_backbone_py": clip_backbone_py,
        "has_vit_b16_url": False,
        "uses_download": False,
        "default_cache": None,
    }

    if clip_py.exists():
        text = clip_py.read_text(encoding="utf-8", errors="replace")
        result["has_vit_b16_url"] = CLIP_FILENAME in text and CLIP_SHA256 in text
        cache_match = re.search(r"os\.path\.expanduser\(\"([^\"]+)\"\)", text)
        if cache_match:
            result["default_cache"] = cache_match.group(1)

    if clip_backbone_py.exists():
        text = clip_backbone_py.read_text(encoding="utf-8", errors="replace")
        result["uses_download"] = "clip._download(url)" in text

    return result


def load_state_dict(path):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect .pth/.tar checkpoints") from exc

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"{path} does not look like a state_dict checkpoint")
    return obj


def summarize_checkpoint(path):
    state_dict = load_state_dict(path)
    keys = list(state_dict.keys())
    norm_keys = [k.removeprefix("module.") for k in keys]
    clip_visual = [k for k in norm_keys if "CLIP_tool.visual" in k]
    clip_adapter = [k for k in norm_keys if "CLIP_tool.adapter_layer" in k]
    semantic_adapter = [k for k in norm_keys if "semantic_adapter" in k]
    distortion_adapter = [k for k in norm_keys if "dist_adapter" in k or "distortion_adapter" in k]
    swin = [k for k in norm_keys if "layers." in k or "patch_embed" in k]

    return {
        "total_keys": len(keys),
        "clip_visual_keys": len(clip_visual),
        "clip_adapter_keys": len(clip_adapter),
        "semantic_adapter_keys": len(semantic_adapter),
        "distortion_adapter_keys": len(distortion_adapter),
        "swin_like_keys": len(swin),
        "clip_visual_sample": clip_visual[:5],
        "clip_adapter_sample": clip_adapter[:5],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check how this KSVQE repo uses CLIP pretraining and whether checkpoints contain CLIP keys."
    )
    parser.add_argument("--root", default=".", help="Project root. Default: current directory.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint path to inspect. Can be passed more than once.",
    )
    parser.add_argument(
        "--clip-cache",
        default=os.path.join(os.path.expanduser("~"), ".cache", "clip", CLIP_FILENAME),
        help="Expected local CLIP cache file.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cache_path = Path(args.clip_cache).expanduser()

    print("== Source check ==")
    source = source_contains_clip_loader(root)
    print(f"CLIP source: {source['clip_py']}")
    print(f"CLIP backbone source: {source['clip_backbone_py']}")
    print(f"{CLIP_MODEL_NAME} URL/hash in source: {source['has_vit_b16_url']}")
    print(f"CLIP_backbone calls clip._download(url): {source['uses_download']}")
    print(f"Default cache in source: {source['default_cache']}")

    print("\n== Local CLIP cache check ==")
    print(f"Expected cache file: {cache_path}")
    if cache_path.is_file():
        digest = sha256_file(cache_path)
        print(f"Exists: yes")
        print(f"Size: {cache_path.stat().st_size} bytes")
        print(f"SHA256 matches OpenAI CLIP {CLIP_MODEL_NAME}: {digest == CLIP_SHA256}")
        print(f"SHA256: {digest}")
    else:
        print("Exists: no")
        print("Note: if training succeeded, check the same path under the server user account.")

    if args.checkpoint:
        print("\n== Checkpoint key check ==")
    for ckpt in args.checkpoint:
        ckpt_path = Path(ckpt).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = root / ckpt_path
        print(f"\nCheckpoint: {ckpt_path}")
        if not ckpt_path.is_file():
            print("Exists: no")
            continue
        try:
            summary = summarize_checkpoint(ckpt_path)
        except RuntimeError as exc:
            print(f"Could not inspect checkpoint: {exc}")
            continue
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
