import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig
import numpy as np
import torch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Instantiate VAE to get scaling factor
    print("Instantiating VAE...")
    # We put VAE on CPU as we only need the config/scaling factor, not the model weights for this
    vae = hydra.utils.instantiate(cfg.vae).to("cpu")
    scaling_factor = vae.config.scaling_factor
    print(f"Scaling factor: {scaling_factor}")

    print("Instantiating Dataloader...")
    dataloader = hydra.utils.instantiate(cfg.data)
    data_iterator = iter(dataloader)

    # Number of batches to estimate the mean over
    num_batches = 400
    print(f"Calculating mean over {num_batches} batches...")

    running_sum = None
    total_pixels = 0

    for i in range(num_batches):
        try:
            batch = next(data_iterator)
        except StopIteration:
            print("Data iterator exhausted.")
            break
        latents = batch["latent"]

        # Replicate train.py logic for loading latents
        # latents are stored as int16 representation of bfloat17
        latents_jax = jnp.array(latents, dtype=jnp.int16)
        latents_float = latents_jax.view(jnp.bfloat16) * scaling_factor

        # Convert to float32 for accumulation to avoid overflow/precision issues
        latents_f32 = latents_float.astype(jnp.float32)

        # Assuming shape [B, C, H, W] or similar. We want mean per channel (C).
        # We sum over all dimensions except the channel dimension.
        # In train.py validation noise is (8, 16, 32, 32), so C is dim 1.
        # If shape is (B, C, H, W), we sum over (0, 2, 3).

        # Let's verify shape dynamically
        if i == 0:
            print(f"Latent shape: {latents_f32.shape}")

        # Sum over Batch (0), Height (2), Width (3) - assuming (B, C, H, W)
        # If the shape is different, this needs adjustment.
        # Based on train.py: validation_noise = jax.random.normal(key, shape=(8, 16, 32, 32)...)
        # So we assume channel is index 1.
        batch_sum = jnp.sum(latents_f32, axis=(0, 2, 3))

        if running_sum is None:
            running_sum = batch_sum
        else:
            running_sum += batch_sum

        # Count pixels: B * H * W
        total_pixels += (
            latents_f32.shape[0] * latents_f32.shape[2] * latents_f32.shape[3]
        )

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_batches} batches...")

    mean = running_sum / total_pixels

    print("\nCalculation complete.")
    print(f"Total pixels processed per channel: {total_pixels}")
    print("VAE Mean (per channel):")
    print(mean)
    print(f"Global Mean: {jnp.mean(mean)}")


if __name__ == "__main__":
    main()
