BATCH_SIZE = 384

from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import torch
from torch import Tensor
import numpy as np

from prompt import prompt  # pyrefly:ignore

from diffusers.models import AutoencoderKL
from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
import torchvision.transforms.functional as TF


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=0, help="Local Rank of Script")
parser.add_argument("--world_size", type=int, default=1, help="How Many Total")
args = parser.parse_args()


def serialize(mean, short_caption, long_caption):
    record = {
        "latent": mean,
        "short_caption": short_caption,
        "long_caption": long_caption,
    }

    return record


vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
).to(device="cuda")

model_name = "Qwen/Qwen3-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
qwen = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
)

transform_latent = v2.Compose(
    [
        v2.RGB(),
        v2.Resize(256),
        v2.CenterCrop(256),
        v2.ToImage(),
        v2.ToDtype(torch.bfloat16, scale=True),
    ]  # pyrefly:ignore
)


transform_qwen = v2.Compose(
    [
        v2.RGB(),
        v2.Resize(512),
        v2.CenterCrop(512),
        v2.ToImage(),
        v2.ToDtype(torch.bfloat16, scale=False),
    ]  # pyrefly:ignore
)


def preprocess(batch) -> dict[str, list[Tensor] | list[str]]:
    latent_tensor: list[Tensor] = [transform_latent(img) for img in batch["jpg"]]
    qwen_tensor: list[Tensor] = [transform_qwen(img) for img in batch["jpg"]]
    caption: list[str] = batch.get("blip2_caption")
    return {"latent_img": latent_tensor, "img": qwen_tensor, "caption": caption}


ds: IterableDataset = load_dataset(  # pyrefly:ignore
    "common-canvas/commoncatalog-cc-by", streaming=True, split="train"
)

cols_to_keep = ["latent_img", "img", "caption"]
cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]  # pyrefly:ignore

ds = ds.map(
    preprocess,
    batched=True,
    remove_columns=cols_to_remove,
)

from datasets.distributed import split_dataset_by_node

ds = split_dataset_by_node(ds, rank=args.rank, world_size=args.world_size)

dataloader = DataLoader(
    ds,  # pyrefly:ignore
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=None,
)

import time
from datetime import timedelta
from array_record.python import array_record_module  # pyrefly:ignore
import pickle

path = f"common_canvas_{args.rank}.array_record"
writer = array_record_module.ArrayRecordWriter(path, "group_size:1")

total = 14581672 // args.world_size // BATCH_SIZE
record_count = 0
last_10_times = []
for i, data in enumerate(dataloader):
    start_time = time.time()
    latent_tensor = data["latent_img"]
    qwen_tensor = data["img"]
    captions = data["caption"]

    latent_inp = latent_tensor.to(device=vae.device)
    imgs = qwen_tensor

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for img in imgs
    ]
    inputs = processor.apply_chat_template(  # pyrefly:ignore
        messages,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        # truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(qwen.device)

    with torch.inference_mode():
        meanlogvar = vae._encode(latent_inp)  # B,32,32,32
        mean = meanlogvar[:, :16, :, :]  # B,16,32,32
        mean = mean.view(torch.int16)

        generated_ids = qwen.generate(
            **inputs,
            do_sample=True,
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            max_new_tokens=300,
        )

    mean_np = mean.detach().cpu().numpy()

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(  # pyrefly:ignore
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    for b in range(mean_np.shape[0]):
        record = serialize(mean_np[b], captions[b], output_text[b])
        writer.write(pickle.dumps(record))
        record_count += 1

    percentage = 1 - ((i + 1) / total)
    last_10_times.append(time.time() - start_time)
    if len(last_10_times) > 10:
        last_10_times.pop(0)
    seconds = int(np.mean(last_10_times) * total * percentage)
    print(f"{i+1}/{total} | Estimated time remaining: {timedelta(seconds=seconds)}")

writer.close()
print(f"Number of records written to array_record file {path} :" f" {record_count}")
