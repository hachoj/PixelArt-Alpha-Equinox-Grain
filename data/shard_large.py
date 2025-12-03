BATCH_SIZE = 384

from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import torch
from torch import Tensor
from torch.amp import autocast

from prompt import prompt  # pyrefly:ignore

from diffusers.models import AutoencoderKL
from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
import torchvision.transforms.functional as TF


vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
).to(device="cuda")

model_name = "Qwen/Qwen3-VL-4B-Instruct"

processor = AutoProcessor.from_pretrained(model_name)
qwen = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
)

transform = v2.Compose(
    [
        v2.RGB(),
        v2.Resize(512),
        v2.CenterCrop(512),
        v2.ToImage(),
        v2.ToDtype(torch.bfloat16, scale=True),
    ]  # pyrefly:ignore
)


def preprocess(batch) -> dict[str, list[Tensor] | list[str]]:
    tensor: list[Tensor] = [transform(img) for img in batch["jpg"]]
    caption: list[str] = batch.get("blip2_caption")
    return {"img": tensor, "caption": caption}


ds: IterableDataset = load_dataset(
    "common-canvas/commoncatalog-cc-by", streaming=True, split="train"
)  # pyrefly:ignore

cols_to_keep = ["img", "caption"]
cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]  # pyrefly:ignore

ds = ds.map(
    preprocess,
    batched=True,
    remove_columns=cols_to_remove,
)

dataloader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)  # pyrefly:ignore

import time
import array_record

writer = array_record.ArrayRecordWriter(  #
    "ldm_dataset.arrayrecord",
    "group_size:1",  # or another group size config string
)

for i, data in enumerate(dataloader):
    start_time = time.time()
    img_tensor = data["img"]
    captions = data["caption"]

    # recaption
    imgs_cpu = img_tensor.float()
    pil_images = [TF.to_pil_image(img) for img in imgs_cpu]  # B

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for pil_img in pil_images
    ]
    inputs = processor.apply_chat_template(  # pyrefly:ignore
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(qwen.device)
    img_tensor = img_tensor.to(qwen.device)

    with torch.inference_mode():
        meanlogvar = vae._encode(img_tensor)  # B,32,64,64
        mean = meanlogvar[:, :16, :, :]  # B,16,64,64

        generated_ids = qwen.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_new_tokens=500,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(  # pyrefly:ignore
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(
            f"Batch thing {i+1}/{14581672//BATCH_SIZE} took {(time.time() - start_time):.4f} seconds."
        )
