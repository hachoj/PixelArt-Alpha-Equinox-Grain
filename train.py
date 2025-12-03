import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np

import hydra
from omegaconf import DictConfig
import wandb
import time

from models.utils import jax_to_torch
import diffrax
from einops import rearrange
import torch


def single_sample_fn(model, noise, label):
    def vector_field(t, y, args):
        model, label = args
        return model(y, t, label)

    term = diffrax.ODETerm(vector_field)

    solver = diffrax.Euler()

    num_steps = 24
    stepsize_controller = diffrax.ConstantStepSize()
    save_times = jnp.linspace(0.0, 1.0, 6)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=1.0 / num_steps,
        y0=noise,
        args=(model, label),
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(ts=save_times),
        max_steps=num_steps + 2,
    )
    return sol.ys


@eqx.filter_jit
def generate_samples(model, noise, labels):
    return jax.vmap(single_sample_fn, in_axes=(None, 0, 0))(model, noise, labels)


@eqx.filter_value_and_grad
def compute_grads(model, x_t, gt, t, labels):
    logits = jax.vmap(model)(x_t, t, labels)
    loss = optax.losses.l2_loss(logits, gt)
    return jnp.mean(loss)


@eqx.filter_jit
def step_model(model, optimizer, state, x_t, gt, t, labels):
    loss, grads = compute_grads(model, x_t, gt, t, labels)
    updates, new_state = optimizer.update(grads, state, model)

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


def train(
    model,
    optmizer,
    state,
    dataloader,
    vae,
    cfg,
    key,
):
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            config={  # optional; store hyperparameters
                "lr": cfg.train.lr,
                "batch_size": cfg.train.batch_size,
            },
            name=cfg.wandb.name,
        )
        wandb.define_metric("train_step")
        wandb.define_metric("train/*", step_metric="train_step")

    scaling_factor = vae.config.scaling_factor

    new_key, sub_key = jr.split(key)

    validation_noise = jax.random.normal(
        new_key, shape=(5, 16, 32, 32), dtype=jnp.bfloat16
    )
    validation_labels = jnp.array([0, 5, 10, 15, 1000])

    for step, batch in enumerate(dataloader):
        latent = batch["latent"]
        labels = batch["label"]

        B = latent.shape[0]

        X1 = jnp.array(latent).view(jnp.bfloat16) * scaling_factor

        new_key, sub_key = jr.split(sub_key)
        X0 = jax.random.normal(new_key, shape=X1.shape, dtype=X1.dtype)

        new_key, sub_key = jr.split(sub_key)
        eps = jax.random.normal(new_key, shape=[B])
        t = jnp.clip(eps, -7, 7)
        t = jax.nn.sigmoid(t)
        t_mult = t[:, None, None, None]

        Xt = t_mult * X1 + (1 - t_mult) * X0

        model, state, loss = step_model(model, optmizer, state, Xt, X1 - X0, t, labels)

        if cfg.wandb.enabled and (step + 1) % cfg.train.every_n_steps == 0:
            wandb.log(
                {
                    "train_step": step,
                    "train/loss": loss,
                }
            )
        if cfg.wandb.enabled and (step + 1) % cfg.train.every_n_image == 0:
            inference_model = eqx.tree_inference(model, value=True)
            generated_latents = generate_samples(
                inference_model, validation_noise, validation_labels
            )

            generated_latents = generated_latents / vae.config.scaling_factor
            generated_latents = jax.device_get(generated_latents)
            generated_latents = np.array(generated_latents, copy=True)
            generated_latents = (
                torch.from_numpy(generated_latents).to("cuda").to(dtype=torch.bfloat16)
            )

            # [B,T,C,H,W]
            generated_latents = generated_latents.view(-1, 16, 32, 32)
            with torch.inference_mode():
                decoded_images = vae.decode(generated_latents)[0]
            decoded_images = rearrange(
                decoded_images, "(b t) c h w -> c (b h) (t w)", b=5
            )

            decoded_images = (
                decoded_images.permute(1, 2, 0).cpu().clip(0, 1).float().numpy() * 255.0
            )

            wandb.log(
                {"train/image": wandb.Image(decoded_images, caption="Euler Solver")}
            )

            del generated_latents, decoded_images

        if step >= cfg.train.total_steps:
            break

    return model, state


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    key = jr.PRNGKey(cfg.train.seed)

    dataloader = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model, key=key)

    vae = hydra.utils.instantiate(cfg.vae).to("cuda")

    optimizer = hydra.utils.instantiate(cfg.optim)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train(model, optimizer, state, dataloader, vae, cfg, key)


if __name__ == "__main__":
    main()
