import argparse
import random
from pathlib import Path


SOURCES = (
    ("Train", "Train/train_data_4col_7class.txt"),
    ("Validation", "Validation/val_truth_4col_7class.txt"),
    ("Test", "Test/test_MOS_7class.txt"),
)


def read_groups(dataset_root, group_size):
    groups = []
    for split_dir, ann_rel in SOURCES:
        ann_path = dataset_root / ann_rel
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")

        lines = [line.strip() for line in ann_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) % group_size != 0:
            raise ValueError(f"{ann_path} has {len(lines)} rows, not divisible by {group_size}")

        for start in range(0, len(lines), group_size):
            group = []
            for pos, line in enumerate(lines[start : start + group_size]):
                parts = line.split(",")
                if len(parts) != 4:
                    raise ValueError(f"Expected 4 comma-separated fields, got: {line}")
                filename, _, _, mos = parts
                cls_label = str(pos)
                dis_label = str(pos)
                group.append(f"{split_dir}/{filename},{cls_label},{dis_label},{mos}")
            groups.append(group)

    return groups


def write_lines(path, groups):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [line for group in groups for line in group]
    path.write_text("\n".join(flat) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Create the paper-style 80/20 reference-content KVQ split with 7-class labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/KVQ-dataset"),
        help="KVQ-dataset root containing Train, Validation, and Test folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset_csv/KVQ_paper_split"),
        help="Output directory for generated annotation files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--group-size", type=int, default=7)
    args = parser.parse_args()

    groups = read_groups(args.dataset_root, args.group_size)
    rng = random.Random(args.seed)
    rng.shuffle(groups)

    train_count = int(round(len(groups) * args.train_ratio))
    train_groups = groups[:train_count]
    val_groups = groups[train_count:]

    write_lines(args.out_dir / "train_80ref_7class.txt", train_groups)
    write_lines(args.out_dir / "val_20ref_7class.txt", val_groups)
    write_lines(args.out_dir / "all_7class_rootpaths.txt", groups)

    print(f"groups: total={len(groups)} train={len(train_groups)} val={len(val_groups)}")
    print(f"videos: total={len(groups) * args.group_size} train={len(train_groups) * args.group_size} val={len(val_groups) * args.group_size}")
    print(f"wrote: {args.out_dir / 'train_80ref_7class.txt'}")
    print(f"wrote: {args.out_dir / 'val_20ref_7class.txt'}")


if __name__ == "__main__":
    main()
