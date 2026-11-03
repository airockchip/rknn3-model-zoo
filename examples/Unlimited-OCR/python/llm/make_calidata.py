import argparse
import json
import math
import os
import pickle
import sys

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from py_utils.cali_data import capture_module_input

import importlib.util

VISION_MAKE_CALIDATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../vision/make_calidata.py")
)
_vision_spec = importlib.util.spec_from_file_location(
    "unlimited_ocr_vision_make_calidata", VISION_MAKE_CALIDATA
)
_vision_module = importlib.util.module_from_spec(_vision_spec)
_vision_spec.loader.exec_module(_vision_module)
build_vision_inputs = _vision_module.build_vision_inputs
load_image = _vision_module.load_image


IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4
BOS_ID = 0


def default_image_size(mode):
    return 1024 if mode == "base" else 640


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


def normalize_prompt(text):
    text = text or "document parsing."
    if IMAGE_TOKEN not in text:
        return f"{IMAGE_TOKEN}{text}"
    return text


def encode_text(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def make_image_token_ids(mode, crop_shape, base_size, image_size):
    width_crop_num, height_crop_num = int(crop_shape[0]), int(crop_shape[1])
    if mode == "gundam":
        num_queries_base = math.ceil((base_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
        tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]) * num_queries_base
        tokenized_image += [IMAGE_TOKEN_ID]

        if width_crop_num > 1 or height_crop_num > 1:
            num_queries = math.ceil((image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
            tokenized_image += ([IMAGE_TOKEN_ID] * (num_queries * width_crop_num) + [IMAGE_TOKEN_ID]) * (
                num_queries * height_crop_num
            )
        return tokenized_image

    num_queries = math.ceil((image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
    tokenized_image = ([IMAGE_TOKEN_ID] * num_queries + [IMAGE_TOKEN_ID]) * num_queries
    tokenized_image += [IMAGE_TOKEN_ID]
    return tokenized_image


def build_model_inputs(tokenizer, image_path, prompt, mode, base_size, image_size):
    image = load_image(image_path)
    images, images_spatial_crop = build_vision_inputs(
        image,
        base_size=base_size,
        image_size=image_size,
        mode=mode,
    )

    text_splits = normalize_prompt(prompt).split(IMAGE_TOKEN)
    tokenized_str = []
    images_seq_mask = []

    for text_sep, crop_shape in zip(text_splits, images_spatial_crop):
        tokenized_sep = encode_text(tokenizer, text_sep)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        tokenized_image = make_image_token_ids(mode, crop_shape, base_size, image_size)
        tokenized_str += tokenized_image
        images_seq_mask += [True] * len(tokenized_image)

    tokenized_sep = encode_text(tokenizer, text_splits[-1])
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    tokenized_str = [BOS_ID] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    input_ids = torch.LongTensor(tokenized_str).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool).unsqueeze(0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images,
        "images_seq_mask": images_seq_mask,
        "images_spatial_crop": images_spatial_crop,
        "use_cache": True,
    }


def save_sample(export_datapath, sample_name, sample):
    export_dir = os.path.dirname(os.path.abspath(export_datapath))
    path_dir = os.path.join(export_dir, "model_inputs")
    os.makedirs(path_dir, exist_ok=True)
    with open(os.path.join(path_dir, sample_name), "wb") as f:
        pickle.dump(sample, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="baidu/Unlimited-OCR")
    parser.add_argument("--datapath", type=str, default="../../../../datasets/OmniDocBench/llm/dataset.json")
    parser.add_argument("--export_datapath", type=str, default="../../data/llm/model_inputs.json")
    parser.add_argument("--mode", choices=["base", "gundam"], default="base")
    parser.add_argument("--base_size", type=int, default=1024)
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="defaults to 1024 in base mode and 640 in gundam mode",
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    args = parser.parse_args()

    if args.image_size is None:
        args.image_size = default_image_size(args.mode)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    model.config.sliding_window = None

    with open(args.datapath, "r", encoding="utf-8") as f:
        datasets = json.load(f)
    if args.max_samples > 0:
        datasets = datasets[: args.max_samples]

    info = []
    for idx, data in enumerate(tqdm(datasets)):
        image_path = resolve_image_path(args.datapath, data)
        prompt = data.get("input", "<image>document parsing.")
        inputs = build_model_inputs(
            tokenizer,
            image_path=image_path,
            prompt=prompt,
            mode=args.mode,
            base_size=args.base_size,
            image_size=args.image_size,
        )

        with torch.inference_mode():
            sample = capture_module_input(
                model.model,
                lambda: model.generate(**inputs, max_new_tokens=args.max_new_tokens),
            )

        sample_name = f"sample_{idx}"
        save_sample(args.export_datapath, sample_name, sample)
        info.append({"sample": f"model_inputs/{sample_name}"})

    os.makedirs(os.path.dirname(os.path.abspath(args.export_datapath)), exist_ok=True)
    with open(args.export_datapath, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
