import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def patchify_pair(first_path, second_path, patch_size=16, image_size=384):
    frames = []
    for image_path in (first_path, second_path):
        image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
        pixels = np.asarray(image, dtype=np.float32)
        frames.append((pixels - 127.5) / 127.5)

    pixel_values = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    merge_size = 2
    patches = pixel_values.reshape(
        1,
        2,
        3,
        image_size // merge_size // patch_size,
        merge_size,
        patch_size,
        image_size // merge_size // patch_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(
        image_size // patch_size * image_size // patch_size,
        3 * 2 * patch_size * patch_size,
    ).contiguous()


def prepare_dataset(source_paths, output_dir, samples=20):
    if samples <= 0:
        raise ValueError("--samples must be positive")

    output_dir = Path(output_dir).expanduser().resolve()
    images_dir = output_dir / "images"
    inputs_dir = output_dir / "model_inputs"
    images_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(path).expanduser().resolve() for path in source_paths]
    if len(source_paths) < 2:
        raise ValueError("At least two calibration images are required")
    for image_path in source_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Calibration image not found: {image_path}")

    copied_paths = []
    for index in range(samples * 2):
        source_path = source_paths[index % len(source_paths)]
        target_path = images_dir / f"{index:03d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, target_path)
        copied_paths.append(target_path)

    grid_thw = torch.tensor([[1, 24, 24]], dtype=torch.int64)
    index_rows = []
    for index in range(samples):
        flatten_patches = patchify_pair(
            copied_paths[index * 2], copied_paths[index * 2 + 1]
        )
        sample_path = inputs_dir / f"sample_{index:03d}.pkl"
        sample = {
            "args": (flatten_patches,),
            "kwargs": {"grid_thw": grid_thw.clone(), "return_dict": True},
        }
        with sample_path.open("wb") as output_file:
            pickle.dump(sample, output_file)
        index_rows.append({"sample": str(sample_path.relative_to(output_dir))})

    dataset_path = output_dir / "model_inputs.json"
    dataset_path.write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
    print(f"GRQ samples: {len(index_rows)} -> {dataset_path}")
    return dataset_path
