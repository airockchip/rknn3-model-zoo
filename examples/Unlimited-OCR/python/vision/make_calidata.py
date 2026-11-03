import argparse
import json
import os
import pickle
import sys

import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from py_utils.cali_data import capture_module_input


def add_model_code_to_path(model_path):
    if os.path.isdir(model_path):
        sys.path.append(os.path.abspath(model_path))


class UnlimitedOCRVisionStack(torch.nn.Module):
    """Vision part used by Unlimited-OCR.

    The original model has three vision-side compute blocks:
      1. sam_model: high-resolution SAM-ViT feature extractor.
      2. vision_model: CLIP-L/14 tower, using sam_model output as patch_embeds.
      3. projector: linear 2048 -> 1280 projector after concatenating CLIP and SAM features.

    image_newline and view_seperator are vision-side parameters used when flattening
    global/local image features into the token sequence.
    """

    def __init__(self, unlimited_ocr_model):
        super().__init__()
        self.sam_model = unlimited_ocr_model.sam_model
        self.vision_model = unlimited_ocr_model.vision_model
        self.projector = unlimited_ocr_model.projector
        self.image_newline = unlimited_ocr_model.image_newline
        self.view_seperator = unlimited_ocr_model.view_seperator

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def _encode_one(self, pixel_values):
        sam_features = self.sam_model(pixel_values)
        clip_features = self.vision_model(pixel_values, sam_features)
        features = torch.cat(
            (clip_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)),
            dim=-1,
        )
        return self.projector(features)

    def forward(self, images, images_spatial_crop):
        all_image_features = []
        for image, crop_shape in zip(images, images_spatial_crop):
            patches = image[0]
            image_ori = image[1]
            width_crop_num = int(crop_shape[0].item())
            height_crop_num = int(crop_shape[1].item())
            images_in_this_batch = []

            if torch.sum(patches).item() != 0:
                local_features = self._encode_one(patches)
                global_features = self._encode_one(image_ori)

                _, hw, n_dim = global_features.shape
                h = w = int(hw**0.5)
                global_features = global_features.view(h, w, n_dim)
                global_features = torch.cat(
                    [
                        global_features,
                        self.image_newline[None, None, :].expand(h, 1, n_dim),
                    ],
                    dim=1,
                ).view(-1, n_dim)

                _, hw2, n_dim2 = local_features.shape
                h2 = w2 = int(hw2**0.5)
                local_features = (
                    local_features.view(
                        height_crop_num,
                        width_crop_num,
                        h2,
                        w2,
                        n_dim2,
                    )
                    .permute(0, 2, 1, 3, 4)
                    .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                )
                local_features = torch.cat(
                    [
                        local_features,
                        self.image_newline[None, None, :].expand(
                            height_crop_num * h2, 1, n_dim2
                        ),
                    ],
                    dim=1,
                ).view(-1, n_dim2)

                images_in_this_batch.append(
                    torch.cat(
                        [local_features, global_features, self.view_seperator[None, :]],
                        dim=0,
                    )
                )
            else:
                for img_idx in range(image_ori.shape[0]):
                    global_features = self._encode_one(image_ori[img_idx : img_idx + 1])
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)
                    global_features = global_features.view(h, w, n_dim)
                    global_features = torch.cat(
                        [
                            global_features,
                            self.image_newline[None, None, :].expand(h, 1, n_dim),
                        ],
                        dim=1,
                    ).view(-1, n_dim)
                    images_in_this_batch.append(
                        torch.cat([global_features, self.view_seperator[None, :]], dim=0)
                    )

            if images_in_this_batch:
                all_image_features.append(torch.cat(images_in_this_batch, dim=0))

        if len(all_image_features) == 1:
            return all_image_features[0]
        return all_image_features


def load_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def normalize_transform(mean, std):
    from torchvision import transforms

    transform_pipelines = [transforms.ToTensor()]
    if mean is not None or std is not None:
        if mean is None:
            mean = [0.0] * len(std)
        if std is None:
            std = [1.0] * len(mean)
        transform_pipelines.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_pipelines)


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=32, image_size=640):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    return processed_images, target_aspect_ratio


def resolve_image_path(datapath, item):
    base_dir = os.path.dirname(os.path.abspath(datapath))
    if "image" in item and os.path.isabs(item["image"]):
        return item["image"]
    if "image_path" in item and "image" in item:
        return os.path.join(base_dir, item["image_path"], item["image"])
    if "image" in item:
        return os.path.join(base_dir, item["image"])
    if "images" in item and item["images"]:
        image_path = item["images"][0]
        if os.path.isabs(image_path):
            return image_path
        return os.path.join(base_dir, image_path)
    raise KeyError("dataset item must contain image/image_path or images")


def build_vision_inputs(image, base_size=1024, image_size=640, mode="base"):
    if mode not in ("base", "gundam"):
        raise ValueError(f"Unsupported vision preprocessing mode: {mode}")

    transform = normalize_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    pad_color = tuple(int(x * 255) for x in (0.5, 0.5, 0.5))

    images_list = []
    images_crop_list = []
    images_spatial_crop = []

    if mode == "gundam":
        if image.size[0] <= image_size and image.size[1] <= image_size:
            crop_ratio = (1, 1)
            images_crop_raw = []
        else:
            images_crop_raw, crop_ratio = dynamic_preprocess(image, image_size=image_size)

        global_view = ImageOps.pad(image, (base_size, base_size), color=pad_color)
        images_list.append(transform(global_view).float())

        width_crop_num, height_crop_num = crop_ratio
        images_spatial_crop.append([width_crop_num, height_crop_num])
        if width_crop_num > 1 or height_crop_num > 1:
            for crop in images_crop_raw:
                images_crop_list.append(transform(crop).float())

        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, base_size, base_size), dtype=torch.float32)
    else:
        if image_size <= 640:
            image = image.resize((image_size, image_size))
        global_view = ImageOps.pad(image, (image_size, image_size), color=pad_color)
        images_list.append(transform(global_view).float())
        images_spatial_crop.append([1, 1])
        images_crop = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)

    images_ori = torch.stack(images_list, dim=0)
    images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
    return [(images_crop, images_ori)], images_spatial_crop


def save_sample(export_datapath, sample_name, sample):
    export_dir = os.path.dirname(os.path.abspath(export_datapath))
    path_dir = os.path.join(export_dir, "model_inputs")
    os.makedirs(path_dir, exist_ok=True)
    with open(os.path.join(path_dir, sample_name), "wb") as f:
        pickle.dump(sample, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="baidu/Unlimited-OCR",
        help="Unlimited-OCR model path",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="../../../../datasets/OmniDocBench/llm/dataset.json",
        help="dataset json path",
    )
    parser.add_argument(
        "--export_datapath",
        type=str,
        default="../../data/vision/model_inputs.json",
        help="output calibration index json path",
    )
    parser.add_argument("--base_size", type=int, default=1024)
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="image size for preprocessing; defaults to 1024 in base mode and 640 in gundam mode",
    )
    parser.add_argument(
        "--mode",
        choices=["base", "gundam"],
        default="base",
        help="vision preprocessing mode: base disables crop, gundam enables dynamic crops",
    )
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="deprecated alias for --mode base",
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()
    if args.no_crop:
        args.mode = "base"
    if args.image_size is None:
        args.image_size = 1024 if args.mode == "base" else 640

    add_model_code_to_path(args.model_path)

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    vision_stack = UnlimitedOCRVisionStack(model.model).eval()

    with open(args.datapath, "r", encoding="utf-8") as f:
        datasets = json.load(f)
    if args.max_samples > 0:
        datasets = datasets[: args.max_samples]

    info = []
    for idx, data in enumerate(tqdm(datasets)):
        image_path = resolve_image_path(args.datapath, data)
        image = load_image(image_path)
        images, images_spatial_crop = build_vision_inputs(
            image,
            base_size=args.base_size,
            image_size=args.image_size,
            mode=args.mode,
        )

        with torch.inference_mode():
            sample = capture_module_input(
                vision_stack,
                lambda: vision_stack(images=images, images_spatial_crop=images_spatial_crop),
            )

        sample_name = "sample_{}".format(idx)
        save_sample(args.export_datapath, sample_name, sample)
        info.append({"sample": "model_inputs/{}".format(sample_name)})

    os.makedirs(os.path.dirname(os.path.abspath(args.export_datapath)), exist_ok=True)
    with open(args.export_datapath, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
