#!/usr/bin/env python3
"""Prepare frame or Codec assets for the Mage-VL C++ demo."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_DEMO_CANVASES = 16
CANVAS_SIZE = 384
PATCH_SIZE = 16
MERGE_SIZE = 2
BLOCK_SIZE = PATCH_SIZE * MERGE_SIZE
PATCH_GRID = CANVAS_SIZE // PATCH_SIZE
BLOCK_GRID = CANVAS_SIZE // BLOCK_SIZE
PATCHES_PER_CANVAS = PATCH_GRID * PATCH_GRID
BLOCKS_PER_CANVAS = BLOCK_GRID * BLOCK_GRID
ROTARY_DIM = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Mage-VL frame-mode or Codec-mode video assets."
    )
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("output_dir", type=Path, help="output asset directory")
    parser.add_argument(
        "--mode",
        choices=("frame", "codec"),
        default="frame",
        help="frame: complete frames; codec: importance-selected Patch canvases",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=MAX_DEMO_CANVASES,
        help="frame mode: uniformly sampled count; 0 decodes every frame",
    )
    parser.add_argument("--groups", type=int, default=4, help="Codec groups")
    parser.add_argument(
        "--frames-per-group", type=int, default=32, help="Codec candidates per group"
    )
    parser.add_argument(
        "--canvases-per-group", type=int, default=4, help="Codec canvases per group"
    )
    parser.add_argument(
        "--codec-score",
        choices=("official", "proxy"),
        default="official",
        help=(
            "official: call the public cv-preinfer bitcost/readiness pipeline; "
            "proxy: retain the legacy decoded-frame-difference heuristic"
        ),
    )
    parser.add_argument(
        "--cv-preinfer-bin",
        default="cv-preinfer",
        help="cv-preinfer executable from codec-video-prep (official Codec mode)",
    )
    parser.add_argument(
        "--format", choices=("jpg", "png"), default="jpg", help="frame mode format"
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95, help="JPEG quality in [1, 100]"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing generated assets in output_dir",
    )
    args = parser.parse_args()

    if args.num_frames < 0:
        parser.error("--num-frames must be 0 or a positive integer")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if args.groups <= 0 or args.frames_per_group <= 1:
        parser.error("--groups must be positive and --frames-per-group must exceed 1")
    if args.canvases_per_group <= 0:
        parser.error("--canvases-per-group must be positive")
    if args.groups * args.canvases_per_group > MAX_DEMO_CANVASES:
        parser.error(
            f"Codec output exceeds the ctx4096 limit of {MAX_DEMO_CANVASES} canvases"
        )
    return args


def open_video(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    return capture


def count_frames(video_path):
    capture = open_video(video_path)
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def video_info(video_path):
    capture = open_video(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if total_frames <= 0:
        total_frames = count_frames(video_path)
    if total_frames <= 0:
        raise RuntimeError(f"video contains no decodable frame: {video_path}")
    return total_frames, fps


def uniform_indices(total_frames, requested_frames):
    output_count = min(total_frames, requested_frames)
    if output_count <= 0:
        return []
    if output_count == 1:
        return [total_frames // 2]
    denominator = output_count - 1
    return [
        (index * (total_frames - 1) + denominator // 2) // denominator
        for index in range(output_count)
    ]


def prepare_output_dir(output_dir, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_names = {
        "decode_metadata.json",
        "codec_metadata.json",
        "cv_preinfer_meta.json",
        "src_patch_position.npy",
        "patch_positions.npy",
        "rotary_pos_emb.fp16.bin",
    }
    generated = [
        path
        for path in output_dir.iterdir()
        if path.is_file()
        and (path.suffix.lower() in SUPPORTED_SUFFIXES or path.name in generated_names)
    ]
    if generated and not overwrite:
        raise RuntimeError(
            f"output directory already contains {len(generated)} generated asset(s); "
            "use --overwrite to replace them"
        )
    if overwrite:
        for path in generated:
            path.unlink()


def decode_selected_frames(video_path, selected_indices):
    capture = open_video(video_path)
    frames = []
    selected_position = 0
    source_index = 0
    while selected_position < len(selected_indices):
        ok, frame = capture.read()
        if not ok:
            break
        if source_index == selected_indices[selected_position]:
            frames.append(frame)
            selected_position += 1
        source_index += 1
    capture.release()
    if len(frames) != len(selected_indices):
        raise RuntimeError(
            "decoder reached end of stream before all selected frames were read "
            f"({len(frames)}/{len(selected_indices)})"
        )
    return frames


def write_image(path, image, image_format, jpeg_quality):
    parameters = []
    if image_format == "jpg":
        parameters = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    if not cv2.imwrite(str(path), image, parameters):
        raise RuntimeError(f"failed to write image: {path}")


def prepare_frame_assets(args, video_path, output_dir, total_frames, fps):
    decode_all = args.num_frames == 0
    if decode_all:
        capture = open_video(video_path)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        selected_indices = list(range(len(frames)))
    else:
        selected_indices = uniform_indices(total_frames, args.num_frames)
        frames = decode_selected_frames(video_path, selected_indices)

    if not frames:
        raise RuntimeError(f"video contains no decodable frame: {video_path}")
    output_files = []
    for output_index, frame in enumerate(frames, 1):
        output_path = output_dir / f"frame_{output_index:06d}.{args.format}"
        write_image(output_path, frame, args.format, args.jpeg_quality)
        output_files.append(output_path.name)

    metadata = {
        "asset_format": "mage_vl_frame_assets_v1",
        "mode": "frame",
        "source_video": str(video_path),
        "source_fps": fps if fps > 0 else None,
        "source_frame_count": total_frames,
        "sampling": "all" if decode_all else "uniform",
        "selected_source_indices": "all" if decode_all else selected_indices,
        "saved_frames": len(output_files),
        "image_files": output_files,
    }
    (output_dir / "decode_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return len(output_files)


def probe_compressed_frames(video_path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("Codec mode requires ffprobe in PATH")
    command = [
        ffprobe,
        "-v",
        "error",
        "-flags2",
        "+export_mvs",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=key_frame,best_effort_timestamp_time,pkt_size,pict_type:frame_side_data_list",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    output = []
    for frame in payload.get("frames", []):
        side_data = frame.get("side_data_list", [])
        output.append(
            {
                "pkt_size": int(frame.get("pkt_size", 0) or 0),
                "pict_type": frame.get("pict_type", "?"),
                "key_frame": bool(int(frame.get("key_frame", 0) or 0)),
                "timestamp": float(frame.get("best_effort_timestamp_time", 0.0) or 0.0),
                "has_motion_vector_side_data": any(
                    item.get("side_data_type") == "Motion vectors" for item in side_data
                ),
            }
        )
    return output


def block_mean(values):
    if values.shape != (CANVAS_SIZE, CANVAS_SIZE):
        raise RuntimeError(f"unexpected score map shape: {values.shape}")
    return values.reshape(BLOCK_GRID, BLOCK_SIZE, BLOCK_GRID, BLOCK_SIZE).mean(
        axis=(1, 3)
    )


def full_canvas_positions(frame_id):
    positions = []
    for block_h in range(BLOCK_GRID):
        for block_w in range(BLOCK_GRID):
            for merge_h in range(MERGE_SIZE):
                for merge_w in range(MERGE_SIZE):
                    positions.append(
                        [
                            frame_id,
                            block_h * MERGE_SIZE + merge_h,
                            block_w * MERGE_SIZE + merge_w,
                        ]
                    )
    return positions


def positions_to_rotary(positions):
    positions = np.asarray(positions, dtype=np.float32)
    section_sizes = (8, 12, 12)
    inverse_frequencies = [
        1.0 / (10000.0 ** (np.arange(size, dtype=np.float32) / size))
        for size in section_sizes
    ]
    frequencies = np.concatenate(
        [
            positions[:, axis : axis + 1] * inverse_frequencies[axis][None, :]
            for axis in range(3)
        ],
        axis=1,
    )
    return np.concatenate([frequencies, frequencies], axis=1).astype(np.float16)


def assemble_codec_group(
    frames, frame_ids, packet_sizes, target_canvases, score_mode, score_grids=None
):
    resized = [
        cv2.resize(frame, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_LINEAR)
        for frame in frames
    ]
    canvases = [resized[0].copy()]
    positions = [full_canvas_positions(frame_ids[0])]
    if target_canvases == 1:
        return canvases, positions, [(frame_ids[0], frame_ids[0])]

    if score_mode == "proxy":
        residual_grids = []
        gray_previous = cv2.cvtColor(resized[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        for frame in resized[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            residual_grids.append(block_mean(np.abs(gray - gray_previous)))
            gray_previous = gray
        residual_stack = np.stack(residual_grids, axis=0)
        residual_scale = max(float(np.percentile(residual_stack, 99.0)), 1e-6)
        residual_stack = np.clip(residual_stack / residual_scale, 0.0, 1.0)
        packet_values = np.log1p(np.asarray(packet_sizes[1:], dtype=np.float32))
        packet_scale = max(float(np.percentile(packet_values, 99.0)), 1e-6)
        packet_values = np.clip(packet_values / packet_scale, 0.0, 1.0)
    elif score_grids is None or len(score_grids) != len(frames):
        raise RuntimeError("official Codec score grids are missing or misaligned")

    ranked = []
    for frame_position in range(1, len(resized)):
        if score_mode == "proxy":
            score_grid = 0.85 * residual_stack[frame_position - 1] + 0.15 * packet_values[
                frame_position - 1
            ]
        else:
            score_grid = score_grids[frame_position]
        for block_h in range(BLOCK_GRID):
            for block_w in range(BLOCK_GRID):
                ranked.append(
                    (
                        float(score_grid[block_h, block_w]),
                        frame_position,
                        block_h,
                        block_w,
                    )
                )
    ranked.sort(
        key=lambda item: (-item[0], frame_ids[item[1]], item[2], item[3])
    )
    required = (target_canvases - 1) * BLOCKS_PER_CANVAS
    selected = ranked[:required]
    if len(selected) < required:
        raise RuntimeError("not enough Codec Patch candidates to fill canvases")
    selected.sort(key=lambda item: (frame_ids[item[1]], item[2], item[3]))

    canvas_ranges = [(frame_ids[0], frame_ids[0])]
    for canvas_index in range(target_canvases - 1):
        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
        canvas_positions = []
        source_ids = []
        begin = canvas_index * BLOCKS_PER_CANVAS
        for destination_block in range(BLOCKS_PER_CANVAS):
            _, source_position, source_block_h, source_block_w = selected[
                begin + destination_block
            ]
            destination_h = destination_block // BLOCK_GRID
            destination_w = destination_block % BLOCK_GRID
            source_y = source_block_h * BLOCK_SIZE
            source_x = source_block_w * BLOCK_SIZE
            destination_y = destination_h * BLOCK_SIZE
            destination_x = destination_w * BLOCK_SIZE
            canvas[
                destination_y : destination_y + BLOCK_SIZE,
                destination_x : destination_x + BLOCK_SIZE,
            ] = resized[source_position][
                source_y : source_y + BLOCK_SIZE,
                source_x : source_x + BLOCK_SIZE,
            ]
            source_frame_id = frame_ids[source_position]
            source_ids.append(source_frame_id)
            for merge_h in range(MERGE_SIZE):
                for merge_w in range(MERGE_SIZE):
                    canvas_positions.append(
                        [
                            source_frame_id,
                            source_block_h * MERGE_SIZE + merge_h,
                            source_block_w * MERGE_SIZE + merge_w,
                        ]
                    )
        canvases.append(canvas)
        positions.append(canvas_positions)
        canvas_ranges.append((min(source_ids), max(source_ids)))
    return canvases, positions, canvas_ranges


def resolve_executable(command):
    path = Path(command).expanduser()
    if path.is_file():
        return str(path.resolve())
    resolved = shutil.which(command)
    if resolved:
        return resolved
    raise RuntimeError(
        "official Codec mode requires cv-preinfer from codec-video-prep. "
        "Install it in a separate preprocessing environment with "
        "`pip install codec-video-prep==0.2.5`, or pass --cv-preinfer-bin."
    )


def prepare_official_codec_assets(args, video_path, output_dir, total_frames, fps):
    """Run Microsoft's public cv-preinfer and add RKNN-demo RoPE assets."""
    cv_preinfer = resolve_executable(args.cv_preinfer_bin)
    candidate_count = args.groups * args.frames_per_group
    expected_canvases = args.groups * args.canvases_per_group
    if total_frames < candidate_count:
        raise RuntimeError(
            f"official Codec mode needs {candidate_count} sampled frames for the "
            f"fixed {args.groups}x{args.frames_per_group} adaptation; video has "
            f"{total_frames}"
        )

    with tempfile.TemporaryDirectory(prefix="mage_cv_preinfer_") as temporary:
        raw_dir = Path(temporary) / "raw"
        command = [
            cv_preinfer,
            "--video",
            str(video_path),
            "--out_dir",
            str(raw_dir),
            "--num_sampled_frames",
            str(candidate_count),
            # Official Mage-VL uses readiness grouping. The adapted ctx4096 model
            # needs exactly four groups, so only grouping is fixed; bitcost scoring,
            # keyframe avoidance and canvas packing remain cv-preinfer-native.
            "--grouping_mode",
            "fixed",
            "--group_size",
            str(args.frames_per_group),
            "--images_per_group",
            str(args.canvases_per_group),
            "--patch",
            str(PATCH_SIZE),
            "--block_size",
            str(MERGE_SIZE),
            "--max_pixels",
            str(CANVAS_SIZE * CANVAS_SIZE),
            "--min_group_frames",
            str(min(8, args.frames_per_group)),
            "--max_group_frames",
            str(args.frames_per_group),
            "--avoid_keyframes",
            "--canvas_format",
            "png",
            "--bitcost_grid",
            "adaptive",
            "--decode_backend",
            "cv_reader_pixels",
            "--thread_type",
            "auto",
            "--thread_count",
            "1",
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-4000:]
            raise RuntimeError(f"cv-preinfer failed with code {result.returncode}: {detail}")

        meta_path = raw_dir / "meta.json"
        position_path = raw_dir / "src_patch_position.npy"
        if not meta_path.is_file() or not position_path.is_file():
            raise RuntimeError("cv-preinfer output is missing meta.json or positions")
        official_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        canvas_names = official_meta.get("canvas_files") or []
        if len(canvas_names) != expected_canvases:
            raise RuntimeError(
                f"cv-preinfer produced {len(canvas_names)} canvases; adapted model "
                f"requires exactly {expected_canvases}"
            )
        positions = np.load(position_path, allow_pickle=False).astype(np.int32)
        if positions.shape != (expected_canvases * PATCHES_PER_CANVAS, 3):
            raise RuntimeError(
                f"unexpected cv-preinfer position shape {positions.shape}; expected "
                f"({expected_canvases * PATCHES_PER_CANVAS}, 3)"
            )

        canvas_files = []
        canvas_shapes = []
        for index, name in enumerate(canvas_names, 1):
            source = raw_dir / name
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot read cv-preinfer canvas: {source}")
            height, width = image.shape[:2]
            if height % PATCH_SIZE or width % PATCH_SIZE:
                raise RuntimeError(f"canvas is not Patch16 aligned: {image.shape}")
            if (height // PATCH_SIZE) * (width // PATCH_SIZE) != PATCHES_PER_CANVAS:
                raise RuntimeError(
                    f"canvas {name} has {(height // PATCH_SIZE) * (width // PATCH_SIZE)} "
                    f"patches; Patch576 Vision requires {PATCHES_PER_CANVAS}"
                )
            output_name = f"canvas_{index:06d}.png"
            shutil.copy2(source, output_dir / output_name)
            canvas_files.append(output_name)
            canvas_shapes.append([height, width])

        np.save(output_dir / "src_patch_position.npy", positions, allow_pickle=False)
        np.save(output_dir / "patch_positions.npy", positions, allow_pickle=False)
        rotary = positions_to_rotary(positions).reshape(
            expected_canvases, PATCHES_PER_CANVAS, ROTARY_DIM
        )
        (output_dir / "rotary_pos_emb.fp16.bin").write_bytes(rotary.tobytes())
        (output_dir / "cv_preinfer_meta.json").write_text(
            json.dumps(official_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    metadata = {
        "asset_format": "mage_vl_codec_assets_v2_cv_preinfer",
        "mode": "codec",
        "source_video": str(video_path),
        "source_fps": fps if fps > 0 else None,
        "source_frame_count": total_frames,
        "codec_backend": "codec-video-prep/cv-preinfer",
        "score_source": "official_block_bitcost",
        "grouping_mode": "fixed_for_patch576_ctx4096",
        "groups": args.groups,
        "frames_per_group": args.frames_per_group,
        "canvases_per_group": args.canvases_per_group,
        "canvas_count": expected_canvases,
        "canvas_shapes": canvas_shapes,
        "patch_size": PATCH_SIZE,
        "merge_size": MERGE_SIZE,
        "patches_per_canvas": PATCHES_PER_CANVAS,
        "vision_tokens_per_canvas": PATCHES_PER_CANVAS // (MERGE_SIZE * MERGE_SIZE),
        "total_vision_tokens": expected_canvases
        * PATCHES_PER_CANVAS
        // (MERGE_SIZE * MERGE_SIZE),
        "canvas_files": canvas_files,
        "src_patch_position": "src_patch_position.npy",
        "rotary_pos_emb": "rotary_pos_emb.fp16.bin",
        "rotary_shape": [expected_canvases, PATCHES_PER_CANVAS, ROTARY_DIM],
        "rotary_dtype": "float16_little_endian",
        "official_metadata": "cv_preinfer_meta.json",
    }
    (output_dir / "codec_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return expected_canvases


def prepare_codec_assets(args, video_path, output_dir, total_frames, fps):
    if args.codec_score == "official":
        return prepare_official_codec_assets(
            args, video_path, output_dir, total_frames, fps
        )

    candidate_count = args.groups * args.frames_per_group
    if total_frames < candidate_count:
        raise RuntimeError(
            f"Codec mode needs {candidate_count} source frames, video has {total_frames}"
        )
    selected_indices = uniform_indices(total_frames, candidate_count)
    frames = decode_selected_frames(video_path, selected_indices)
    compressed_frames = probe_compressed_frames(video_path)
    packet_sizes = [
        compressed_frames[index]["pkt_size"] if index < len(compressed_frames) else 0
        for index in selected_indices
    ]
    all_canvases = []
    all_positions = []
    all_ranges = []
    for group_index in range(args.groups):
        begin = group_index * args.frames_per_group
        end = begin + args.frames_per_group
        canvases, positions, ranges = assemble_codec_group(
            frames[begin:end],
            selected_indices[begin:end],
            packet_sizes[begin:end],
            args.canvases_per_group,
            args.codec_score,
            None,
        )
        all_canvases.extend(canvases)
        all_positions.extend(positions)
        all_ranges.extend(ranges)

    canvas_files = []
    for index, canvas in enumerate(all_canvases, 1):
        output_path = output_dir / f"canvas_{index:06d}.png"
        write_image(output_path, canvas, "png", 100)
        canvas_files.append(output_path.name)

    positions = np.asarray(all_positions, dtype=np.int32)
    if positions.shape != (len(all_canvases), PATCHES_PER_CANVAS, 3):
        raise RuntimeError(f"unexpected Codec position shape: {positions.shape}")
    flat_positions = positions.reshape(-1, 3)
    np.save(output_dir / "src_patch_position.npy", flat_positions, allow_pickle=False)
    np.save(output_dir / "patch_positions.npy", flat_positions, allow_pickle=False)
    rotary = positions_to_rotary(flat_positions).reshape(
        len(all_canvases), PATCHES_PER_CANVAS, ROTARY_DIM
    )
    (output_dir / "rotary_pos_emb.fp16.bin").write_bytes(rotary.tobytes())

    selected_probe = [
        compressed_frames[index] if index < len(compressed_frames) else {}
        for index in selected_indices
    ]
    metadata = {
        "asset_format": "mage_vl_codec_assets_v1",
        "mode": "codec",
        "source_video": str(video_path),
        "source_fps": fps if fps > 0 else None,
        "source_frame_count": total_frames,
        "candidate_sampling": "uniform_fixed_128_for_patch576_ctx4096",
        "selected_source_indices": selected_indices,
        "groups": args.groups,
        "frames_per_group": args.frames_per_group,
        "canvases_per_group": args.canvases_per_group,
        "canvas_count": len(all_canvases),
        "canvas_size": [CANVAS_SIZE, CANVAS_SIZE],
        "patch_size": PATCH_SIZE,
        "merge_size": MERGE_SIZE,
        "patches_per_canvas": PATCHES_PER_CANVAS,
        "importance_signal": "0.85*decoded_residual+0.15*ffprobe_frame_pkt_size",
        "score_backend": "legacy_proxy",
        "mpp_colmv": False,
        "bitcost_precision": "frame_packet_size_proxy_not_block_bitcost",
        "ffmpeg_motion_vector_side_data_frames": sum(
            bool(item.get("has_motion_vector_side_data")) for item in selected_probe
        ),
        "canvas_files": canvas_files,
        "canvas_frame_ranges": [list(item) for item in all_ranges],
        "src_patch_position": "src_patch_position.npy",
        "rotary_pos_emb": "rotary_pos_emb.fp16.bin",
        "rotary_shape": [len(all_canvases), PATCHES_PER_CANVAS, ROTARY_DIM],
        "rotary_dtype": "float16_little_endian",
    }
    (output_dir / "codec_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return len(all_canvases)


def main():
    args = parse_args()
    try:
        video_path = args.video.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        if not video_path.is_file():
            raise RuntimeError(f"input video does not exist: {video_path}")
        prepare_output_dir(output_dir, args.overwrite)
        total_frames, fps = video_info(video_path)
        if args.mode == "codec":
            count = prepare_codec_assets(args, video_path, output_dir, total_frames, fps)
            print(f"Prepared {count} Codec canvas(es) in: {output_dir}")
            print("Run rknn_mage_vl_demo with --input-mode codec.")
        else:
            count = prepare_frame_assets(args, video_path, output_dir, total_frames, fps)
            print(f"Decoded {count} frame(s) to: {output_dir}")
            print("Run rknn_mage_vl_demo with --input-mode frame.")
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
