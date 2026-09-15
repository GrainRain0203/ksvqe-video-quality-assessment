import argparse
import re
from pathlib import Path

import pandas as pd


H264_SUFFIX_RE = re.compile(r"^(.*)_crf_\d+_ss_\d+_t_[0-9.]+\.mp4$")


def canonical_vid(filename):
    match = H264_SUFFIX_RE.match(filename)
    if match:
        return match.group(1)
    return Path(filename).stem


def main():
    parser = argparse.ArgumentParser(
        description="Build a 4-column YouTube-UGC annotation file from the official MOS workbook."
    )
    parser.add_argument(
        "--mos-xlsx",
        required=True,
        type=Path,
        help="Path to original_videos_MOS_for_YouTube_UGC_dataset.xlsx.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory containing YouTube-UGC videos. When provided, only existing matched videos are written.",
    )
    parser.add_argument(
        "--out-file",
        required=True,
        type=Path,
        help="Output 4-column txt: filename, cls_label, dis_label, MOS full.",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.mos_xlsx, sheet_name="MOS")
    required = {"vid", "category", "MOS full"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in MOS sheet: {sorted(missing)}")

    category_to_id = {
        category: idx for idx, category in enumerate(sorted(df["category"].dropna().unique()))
    }
    rows_by_vid = {
        str(row["vid"]): row
        for _, row in df.iterrows()
        if pd.notna(row["vid"]) and pd.notna(row["MOS full"])
    }

    lines = []
    unmatched_videos = []
    if args.video_dir is not None:
        for video_path in sorted(args.video_dir.glob("*.mp4")):
            vid = canonical_vid(video_path.name)
            row = rows_by_vid.get(vid)
            if row is None:
                unmatched_videos.append(video_path.name)
                continue
            label = category_to_id[row["category"]]
            lines.append(
                f"{video_path.name},{label},{label},{float(row['MOS full']):.9f}"
            )
    else:
        for vid, row in rows_by_vid.items():
            label = category_to_id[row["category"]]
            lines.append(f"{vid}.mp4,{label},{label},{float(row['MOS full']):.9f}")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"MOS rows: {len(rows_by_vid)}")
    if args.video_dir is not None:
        print(f"video files: {len(list(args.video_dir.glob('*.mp4')))}")
        print(f"matched videos written: {len(lines)}")
        print(f"unmatched video files skipped: {len(unmatched_videos)}")
        if unmatched_videos:
            print("unmatched sample:", unmatched_videos[:10])
    else:
        print(f"rows written: {len(lines)}")
    print(f"category_to_id: {category_to_id}")
    print(f"wrote: {args.out_file}")


if __name__ == "__main__":
    main()
