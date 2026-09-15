import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_annotation(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_no} expected 4 columns, got {len(parts)}")
            rows.append(
                {
                    "filename": parts[0],
                    "cls_label": parts[1],
                    "dis_label": parts[2],
                    "mos": float(parts[3]),
                    "mos_text": parts[3],
                }
            )
    return rows


def safe_float(value, default=np.nan):
    try:
        if value in (None, "", "N/A"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_fps(text):
    if not text or text == "0/0":
        return np.nan
    if "/" in text:
        a, b = text.split("/", 1)
        a = safe_float(a)
        b = safe_float(b)
        if b and not np.isnan(a) and not np.isnan(b):
            return a / b
        return np.nan
    return safe_float(text)


def ffprobe_metadata(video_path):
    if shutil.which("ffprobe") is None:
        return {}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,bit_rate,nb_frames,duration",
        "-show_entries",
        "format=duration,bit_rate,size",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except Exception:
        return {}
    if result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    duration = safe_float(stream.get("duration"))
    if np.isnan(duration):
        duration = safe_float(fmt.get("duration"))
    bit_rate = safe_float(stream.get("bit_rate"))
    if np.isnan(bit_rate):
        bit_rate = safe_float(fmt.get("bit_rate"))
    return {
        "codec": stream.get("codec_name", ""),
        "width": safe_float(stream.get("width")),
        "height": safe_float(stream.get("height")),
        "fps": parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "duration": duration,
        "bit_rate": bit_rate,
        "nb_frames": safe_float(stream.get("nb_frames")),
        "format_size": safe_float(fmt.get("size")),
    }


def colorfulness(rgb):
    rg = rgb[:, :, 0].astype(np.float32) - rgb[:, :, 1].astype(np.float32)
    yb = 0.5 * (rgb[:, :, 0].astype(np.float32) + rgb[:, :, 1].astype(np.float32)) - rgb[
        :, :, 2
    ].astype(np.float32)
    return float(
        math.sqrt(float(np.var(rg)) + float(np.var(yb)))
        + 0.3 * math.sqrt(float(np.mean(rg) ** 2) + float(np.mean(yb) ** 2))
    )


def blockiness(gray):
    gray = gray.astype(np.float32)
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return np.nan
    vert = np.abs(np.diff(gray, axis=1))
    horiz = np.abs(np.diff(gray, axis=0))
    v_boundary = vert[:, 7::8]
    h_boundary = horiz[7::8, :]
    v_non = np.delete(vert, np.arange(7, vert.shape[1], 8), axis=1)
    h_non = np.delete(horiz, np.arange(7, horiz.shape[0], 8), axis=0)
    boundary = np.mean(v_boundary) + np.mean(h_boundary)
    non_boundary = np.mean(v_non) + np.mean(h_non)
    return float(boundary / (non_boundary + 1e-6))


def frame_stats_from_bgr(frame, resize_width=320):
    import cv2

    h, w = frame.shape[:2]
    if resize_width and w > resize_width:
        new_h = max(1, int(h * resize_width / w))
        frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    highpass = gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 1.0)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).reshape(-1)
    prob = hist / (hist.sum() + 1e-6)
    entropy = -float(np.sum(prob * np.log2(prob + 1e-12)))

    return {
        "brightness_mean": float(np.mean(gray)),
        "brightness_std": float(np.std(gray)),
        "sharpness_lap_var": float(np.var(lap)),
        "noise_highpass_std": float(np.std(highpass)),
        "complexity_sobel_mean": float(np.mean(np.hypot(sobel_x, sobel_y))),
        "entropy": entropy,
        "colorfulness": colorfulness(rgb),
        "blockiness": blockiness(gray),
    }


def average_frame_stats(frame_stats):
    feats = defaultdict(list)
    for stats in frame_stats:
        for key, value in stats.items():
            feats[key].append(value)
    return {key: float(np.nanmean(values)) for key, values in feats.items() if values}


FRAME_FEATURE_KEYS = {
    "brightness_mean",
    "brightness_std",
    "sharpness_lap_var",
    "noise_highpass_std",
    "complexity_sobel_mean",
    "entropy",
    "colorfulness",
    "blockiness",
}


def frame_features_opencv(video_path, sample_frames=8, resize_width=320):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required for --with-frame-features") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"frame_error": "opencv_cannot_open"}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        indices = np.linspace(0, max(frame_count - 1, 0), sample_frames).astype(int)
    else:
        indices = np.arange(sample_frames)

    feats = defaultdict(list)
    for idx in indices:
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        for key, value in frame_stats_from_bgr(frame, resize_width=resize_width).items():
            feats[key].append(value)

    cap.release()
    result = {key: float(np.nanmean(values)) for key, values in feats.items() if values}
    if not result:
        return {"frame_error": "opencv_no_frames_read"}
    return result


def frame_features_decord(video_path, sample_frames=8, resize_width=320):
    try:
        import cv2
        import decord
        from decord import VideoReader, cpu
    except Exception as exc:
        return {"frame_error": f"decord_import_failed: {exc}"}

    try:
        decord.bridge.set_bridge("native")
        reader = VideoReader(str(video_path), ctx=cpu(0))
        nframes = len(reader)
        if nframes <= 0:
            return {"frame_error": "decord_zero_frames"}
        indices = np.linspace(0, max(nframes - 1, 0), sample_frames).astype(int)
        frames = []
        for idx in indices:
            try:
                frames.append(reader[int(idx)].asnumpy())
            except Exception:
                continue
    except Exception as exc:
        return {"frame_error": f"decord_read_failed: {exc}"}

    stats = []
    for rgb in frames:
        if rgb is None:
            continue
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        stats.append(frame_stats_from_bgr(bgr, resize_width=resize_width))
    result = average_frame_stats(stats)
    if not result:
        return {"frame_error": "decord_no_frames_read"}
    return result


def frame_features_ffmpeg(video_path, duration=np.nan, sample_frames=8, resize_width=320):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required for --with-frame-features") from exc
    if shutil.which("ffmpeg") is None:
        return {"frame_error": "ffmpeg_not_found"}

    if np.isnan(duration) or duration <= 0:
        meta = ffprobe_metadata(video_path)
        duration = safe_float(meta.get("duration"))
    if np.isnan(duration) or duration <= 0:
        timestamps = [0]
    else:
        timestamps = np.linspace(0.05 * duration, 0.95 * duration, sample_frames)

    stats = []
    first_error = ""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, timestamp in enumerate(timestamps):
            out_png = tmp_dir / f"frame_{i:03d}.png"
            vf = f"yadif=deint=interlaced,scale={resize_width}:-2"
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{float(timestamp):.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                vf,
                "-y",
                str(out_png),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except subprocess.TimeoutExpired:
                if not first_error:
                    first_error = "ffmpeg_timeout"
                continue
            if result.returncode != 0 or not out_png.exists():
                if not first_error:
                    stderr = (result.stderr or result.stdout or "").strip()
                    first_error = stderr[:300] if stderr else f"ffmpeg_returncode_{result.returncode}"
                continue
            frame = cv2.imread(str(out_png), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            stats.append(frame_stats_from_bgr(frame, resize_width=resize_width))
    result = average_frame_stats(stats)
    if not result:
        return {"frame_error": first_error or "ffmpeg_no_frames_extracted"}
    return result


def frame_features(video_path, duration=np.nan, sample_frames=8, resize_width=320, reader="ffmpeg"):
    if reader == "opencv":
        return frame_features_opencv(
            video_path, sample_frames=sample_frames, resize_width=resize_width
        )
    if reader == "decord":
        return frame_features_decord(
            video_path, sample_frames=sample_frames, resize_width=resize_width
        )
    return frame_features_ffmpeg(
        video_path,
        duration=duration,
        sample_frames=sample_frames,
        resize_width=resize_width,
    )


def zscore(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    means = np.nanmean(matrix, axis=0)
    inds = np.where(np.isnan(matrix))
    matrix[inds] = np.take(means, inds[1])
    stds = np.std(matrix, axis=0)
    stds[stds < 1e-8] = 1.0
    return (matrix - means) / stds


def pca2(matrix):
    x = zscore(matrix)
    x = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def kmeans(matrix, k, seed=42, max_iter=100):
    rng = np.random.default_rng(seed)
    x = zscore(matrix)
    n = x.shape[0]
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cid in range(k):
            members = x[labels == cid]
            if len(members):
                centers[cid] = members.mean(axis=0)
            else:
                centers[cid] = x[rng.integers(0, n)]
    inertia = float(((x - centers[labels]) ** 2).sum())
    return labels, inertia


def silhouette_score_fast(matrix, labels, max_points=600, seed=42):
    x = zscore(matrix)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    if len(x) > max_points:
        idx = rng.choice(len(x), size=max_points, replace=False)
        x = x[idx]
        labels = labels[idx]
    dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i in range(len(x)):
        same = labels == labels[i]
        same[i] = False
        a = np.mean(dist[i, same]) if np.any(same) else 0.0
        b = min(np.mean(dist[i, labels == other]) for other in unique if other != labels[i])
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 1e-8 else 0.0)
    return float(np.mean(scores))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_outputs(out_dir, group_rows, feature_matrix, labels_by_k, metrics):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    coords = pca2(feature_matrix)
    for k, labels in labels_by_k.items():
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=18, cmap="tab20")
        plt.title(f"KVQ group workflow clusters, k={k}")
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")
        plt.colorbar(scatter, label="cluster")
        plt.tight_layout()
        plt.savefig(out_dir / f"pca_clusters_k{k}.png", dpi=180)
        plt.close()

    plt.figure(figsize=(8, 5))
    ks = [m["k"] for m in metrics]
    plt.plot(ks, [m["inertia"] for m in metrics], marker="o", label="inertia")
    plt.xlabel("k")
    plt.ylabel("inertia")
    plt.twinx()
    plt.plot(ks, [m["silhouette"] for m in metrics], marker="s", color="tab:orange", label="silhouette")
    plt.ylabel("silhouette")
    plt.title("Cluster count diagnostics")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_count_diagnostics.png", dpi=180)
    plt.close()

    positions = list(range(7))
    mos_means = []
    size_means = []
    for pos in positions:
        vals_mos = [float(row[f"pos{pos + 1}_mos"]) for row in group_rows]
        vals_size = [float(row[f"pos{pos + 1}_size"]) for row in group_rows]
        mos_means.append(np.nanmean(vals_mos))
        size_means.append(np.nanmean(vals_size) / 1024.0)
    plt.figure(figsize=(8, 5))
    plt.plot([p + 1 for p in positions], mos_means, marker="o")
    plt.xlabel("position in 7-video group")
    plt.ylabel("mean MOS")
    plt.title("Mean MOS by group position")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_mos_by_position.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot([p + 1 for p in positions], size_means, marker="o")
    plt.xlabel("position in 7-video group")
    plt.ylabel("mean file size (KB)")
    plt.title("Mean file size by group position")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_size_by_position.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze KVQ 7-video groups and build pseudo distortion workflow labels."
    )
    parser.add_argument("--train-ann", default="data/KVQ-dataset/Train/train_data_4col_7class.txt")
    parser.add_argument("--train-root", default="data/KVQ-dataset/Train")
    parser.add_argument("--val-ann", default="data/KVQ-dataset/Validation/val_truth_4col_7class.txt")
    parser.add_argument("--val-root", default="data/KVQ-dataset/Validation")
    parser.add_argument("--test-ann", default="data/KVQ-dataset/Test/test_MOS_7class.txt")
    parser.add_argument("--test-root", default="data/KVQ-dataset/Test")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--out-dir", default="analysis/kvq_distortion_clusters")
    parser.add_argument("--group-size", type=int, default=7)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--pseudo-k", type=int, default=10)
    parser.add_argument("--with-frame-features", action="store_true")
    parser.add_argument("--frame-reader", choices=["ffmpeg", "opencv", "decord"], default="decord")
    parser.add_argument("--include-mos-in-clustering", action="store_true")
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    split_specs = [
        ("train", Path(args.train_ann), Path(args.train_root)),
        ("val", Path(args.val_ann), Path(args.val_root)),
        ("test", Path(args.test_ann), Path(args.test_root)),
    ]
    split_specs = [spec for spec in split_specs if spec[0] in requested_splits]

    per_video = []
    split_offsets = {}
    frame_feature_success = 0
    frame_feature_failed = 0
    frame_failure_reasons = defaultdict(int)
    start_time = time.time()
    for split, ann_path, root in split_specs:
        if not ann_path.exists():
            print(f"skip missing annotation: {ann_path}")
            continue
        split_offsets[split] = len(per_video)
        for local_idx, row in enumerate(read_annotation(ann_path)):
            if args.limit_groups and len(per_video) >= args.limit_groups * args.group_size:
                break
            video_path = root / row["filename"]
            if not video_path.exists():
                video_path = root / row["filename"].split("/", 1)[-1]
            item = {
                "split": split,
                "local_index": local_idx,
                "global_index": len(per_video),
                "group_id": len(per_video) // args.group_size,
                "position": local_idx % args.group_size,
                "filename": row["filename"],
                "video_path": str(video_path),
                "mos": row["mos"],
                "mos_text": row["mos_text"],
                "size": os.path.getsize(video_path) if video_path.exists() else np.nan,
            }
            item.update(ffprobe_metadata(video_path))
            if args.with_frame_features and video_path.exists():
                current_video_no = len(per_video) + 1
                if args.progress_every and current_video_no % args.progress_every == 1:
                    elapsed = time.time() - start_time
                    print(
                        f"feature progress: videos={current_video_no - 1}, "
                        f"success={frame_feature_success}, failed={frame_feature_failed}, "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                features = frame_features(
                        video_path,
                        duration=safe_float(item.get("duration")),
                        sample_frames=args.sample_frames,
                        reader=args.frame_reader,
                    )
                if any(key in features for key in FRAME_FEATURE_KEYS):
                    frame_feature_success += 1
                    item.update(features)
                else:
                    frame_feature_failed += 1
                    frame_failure_reasons[features.get("frame_error", "unknown")] += 1
                    item.update(features)
            per_video.append(item)

    if args.with_frame_features:
        print(
            "frame feature extraction:",
            f"success={frame_feature_success}",
            f"failed={frame_feature_failed}",
        )
        if frame_failure_reasons:
            print("frame failure reasons:")
            for reason, count in sorted(frame_failure_reasons.items(), key=lambda kv: -kv[1])[:8]:
                print(f"  {count}: {reason}")
        if frame_feature_success == 0:
            raise RuntimeError(
                "No frame features were extracted. Check that the updated script is "
                "synced and ffmpeg/opencv can decode the videos."
            )

    video_fields = sorted({key for row in per_video for key in row.keys()})
    write_csv(out_dir / "per_video_features.csv", per_video, video_fields)

    groups = []
    for gid in sorted({row["group_id"] for row in per_video}):
        members = [row for row in per_video if row["group_id"] == gid]
        if len(members) != args.group_size:
            continue
        members = sorted(members, key=lambda row: row["position"])
        base = members[0]
        group = {"group_id": gid, "split": base["split"], "original_filename": base["filename"]}
        excluded_numeric_keys = {
            "local_index",
            "global_index",
            "group_id",
            "position",
            "nb_frames",
        }
        if not args.include_mos_in_clustering:
            excluded_numeric_keys.add("mos")
        numeric_keys = [
            key
            for key in video_fields
            if key not in {"split", "filename", "video_path", "codec", "mos_text"}
            and key not in excluded_numeric_keys
            and isinstance(base.get(key), (int, float, np.floating))
        ]
        for pos, row in enumerate(members, 1):
            group[f"pos{pos}_filename"] = row["filename"]
            group[f"pos{pos}_mos"] = row["mos"]
            group[f"pos{pos}_size"] = row["size"]
        for pos, row in enumerate(members[1:], 2):
            for key in numeric_keys:
                value = safe_float(row.get(key))
                original = safe_float(base.get(key))
                group[f"p{pos}_{key}"] = value
                group[f"p{pos}_{key}_delta"] = value - original
                group[f"p{pos}_{key}_ratio"] = value / (original + 1e-6)
        groups.append(group)

    group_fields = sorted({key for row in groups for key in row.keys()})
    write_csv(out_dir / "group_features.csv", groups, group_fields)

    feature_keys = [
        key
        for key in group_fields
        if key not in {"group_id", "split", "original_filename"}
        and not key.endswith("_filename")
        and (args.include_mos_in_clustering or "_mos" not in key)
    ]
    feature_matrix = np.array(
        [[safe_float(row.get(key)) for key in feature_keys] for row in groups],
        dtype=np.float64,
    )

    labels_by_k = {}
    metrics = []
    for k in range(args.k_min, args.k_max + 1):
        labels, inertia = kmeans(feature_matrix, k, seed=args.seed)
        sil = silhouette_score_fast(feature_matrix, labels, seed=args.seed)
        labels_by_k[k] = labels
        metrics.append({"k": k, "inertia": inertia, "silhouette": sil})

    write_csv(out_dir / "cluster_count_metrics.csv", metrics, ["k", "inertia", "silhouette"])

    pseudo_k = args.pseudo_k
    if pseudo_k not in labels_by_k:
        labels_by_k[pseudo_k], _ = kmeans(feature_matrix, pseudo_k, seed=args.seed)

    cluster_rows = []
    selected_labels = labels_by_k[pseudo_k]
    for row, label in zip(groups, selected_labels):
        out = dict(row)
        out["workflow_cluster"] = int(label)
        cluster_rows.append(out)
    write_csv(out_dir / f"group_clusters_k{pseudo_k}.csv", cluster_rows, sorted({k for r in cluster_rows for k in r}))

    cluster_summary = []
    for cid in range(pseudo_k):
        members = [row for row in cluster_rows if row["workflow_cluster"] == cid]
        if not members:
            continue
        summary = {"workflow_cluster": cid, "count": len(members)}
        for pos in range(1, args.group_size + 1):
            summary[f"pos{pos}_mos_mean"] = float(np.mean([float(row[f"pos{pos}_mos"]) for row in members]))
            summary[f"pos{pos}_size_mean"] = float(np.mean([float(row[f"pos{pos}_size"]) for row in members]))
        cluster_summary.append(summary)
    write_csv(
        out_dir / f"cluster_summary_k{pseudo_k}.csv",
        cluster_summary,
        sorted({k for r in cluster_summary for k in r}),
    )

    group_index_by_id = {row["group_id"]: i for i, row in enumerate(groups)}

    # Pseudo labels: original is 0; processed labels are 1 + cluster * 6 + qp_position.
    for split, ann_path, _root in split_specs:
        rows = read_annotation(ann_path)
        if split not in split_offsets:
            continue
        out_path = out_dir / f"{split}_pseudo{pseudo_k}_4col.txt"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            for idx, row in enumerate(rows):
                gid = (split_offsets[split] + idx) // args.group_size
                if gid not in group_index_by_id:
                    continue
                pos = idx % args.group_size
                if pos == 0:
                    label = 0
                else:
                    group_idx = group_index_by_id[gid]
                    label = 1 + int(selected_labels[group_idx]) * 6 + (pos - 1)
                f.write(f"{row['filename']},{label},{label},{row['mos_text']}\n")

    plot_outputs(out_dir, groups, feature_matrix, labels_by_k, metrics)
    print(f"wrote analysis outputs to: {out_dir}")
    print("important files:")
    print(f"  {out_dir / 'per_video_features.csv'}")
    print(f"  {out_dir / 'group_features.csv'}")
    print(f"  {out_dir / 'cluster_count_metrics.csv'}")
    print(f"  {out_dir / f'group_clusters_k{pseudo_k}.csv'}")
    print(f"  {out_dir / f'cluster_summary_k{pseudo_k}.csv'}")


if __name__ == "__main__":
    main()
