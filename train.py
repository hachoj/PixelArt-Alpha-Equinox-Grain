import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as jshard
import orbax.checkpoint
import equinox as eqx
import optax
import numpy as np
import grain.python as grain

import hydra
from omegaconf import DictConfig
import wandb
import time

import diffrax
from einops import rearrange
import torch


@eqx.filter_jit
def update_ema(model_ema, model_train, decay):
    mask = get_trainable_mask(model_ema)
    params_ema, static_ema = eqx.partition(model_ema, mask)
    params_train, _ = eqx.partition(model_train, mask)

    step_size = 1.0 - decay
    params_ema = optax.incremental_update(params_train, params_ema, step_size)

    return eqx.combine(params_ema, static_ema)


def get_trainable_mask(model):
    # First, get all possible trainingable params
    mask = jax.tree_util.tree_map(eqx.is_inexact_array, model)

    # Then mask out the trainable params you want frozen
    mask = eqx.tree_at(
        lambda m: [m.pos_embed.emb, m.time_proj.emb], mask, replace=(False, False)
    )
    return mask


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
def generate_samples(model, noise, labels, model_sharding, data_sharding):
    model = eqx.filter_shard(model, model_sharding)
    noise, labels = eqx.filter_shard((noise, labels), data_sharding)
    return jax.vmap(single_sample_fn, in_axes=(None, 0, 0))(model, noise, labels)


@eqx.filter_value_and_grad
def compute_grads(params, static, x_t, v, t, labels):
    model = eqx.combine(params, static)

    logits = jax.vmap(model)(x_t, t, labels)
    loss = optax.losses.l2_loss(logits, v)
    return jnp.mean(loss)


@eqx.filter_jit(donate="all")
def step_model(state, optimizer, x_t, v, t, labels, model_sharding, data_sharding):
    # you shard again here for XLA efficiency, it doesn't
    # actually divide the shards into smaller "sub-shards"

    # model, opt_state = state
    state = eqx.filter_shard(state, model_sharding)
    model_fp32, opt_state = state
    mask = get_trainable_mask(model_fp32)
    params_fp32, static_fp32 = eqx.partition(model_fp32, mask)

    # cast model weights to bf16
    params_bf16 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, params_fp32
    )
    static_bf16 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, static_fp32
    )

    x_t, v, t, labels = eqx.filter_shard((x_t, v, t, labels), data_sharding)
    x_t, v, t = (
        x_t.astype(jnp.bfloat16),
        v.astype(jnp.bfloat16),
        t.astype(jnp.bfloat16),
    )

    loss, grads = compute_grads(params_bf16, static_bf16, x_t, v, t, labels)

    # use bf16 grads to update fp32 weights since bf16 should be stable enough
    # without the use of grad scaling
    updates, opt_state = optimizer.update(grads, opt_state, params=params_fp32)
    params_fp32 = eqx.apply_updates(params_fp32, updates)

    model_fp32 = eqx.combine(params_fp32, static_fp32)

    return (model_fp32, opt_state), loss


def train(
    state,
    model_ema,
    optimizer,
    data_iterator,
    vae,
    cfg,
    model_sharding,
    data_sharding,
    key,
    checkpoint_manager,
    step_start=0,
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
        wandb.define_metric("image/*", step_metric="train_step")

    scaling_factor = cfg.train.scaling_factor
    # Computed from ~800k images
    latent_mean = jnp.array(
        [
            0.8951277,
            -0.27458745,
            -0.10973451,
            0.9663296,
            1.1391017,
            0.33827597,
            -0.13424845,
            -0.97946304,
            -0.41076022,
            0.8262475,
            1.139077,
            0.7410347,
            1.1904861,
            -1.3059448,
            -0.8135325,
            -0.3679146,
        ],
        dtype=jnp.float32,
    )

    key, sub_key = jr.split(key)

    validation_noise = jax.random.normal(key, shape=(1, 16, 32, 32), dtype=jnp.bfloat16)
    validation_noise = jnp.repeat(validation_noise, repeats=8, axis=0)
    validation_labels = jnp.array([0, 5, 10, 15, 200, 500, 750, 1000])

    # shard once before the loop since it's reused
    validation_noise = eqx.filter_shard(validation_noise, data_sharding)
    validation_labels = eqx.filter_shard(validation_labels, data_sharding)

    start_time = time.time()

    for step in range(step_start, cfg.train.total_steps):
        try:
            batch = next(data_iterator)
        except StopIteration:
            break

        latents = batch["latent"]
        labels = batch["label"]

        latents = jax.device_put(latents, data_sharding)
        labels = jax.device_put(labels, data_sharding)

        # with jax sharding, this still actually prints the global batch size
        B = latents.shape[0]

        key, sub_key = jr.split(key)

        # classifier free guidience
        probs = jnp.array([1 - cfg.train.cfg_p, cfg.train.cfg_p])
        mask = jr.choice(key, 2, shape=(B,), p=probs)

        labels = jnp.where(mask == 1, 1000, labels)

        # X1 inherit the sharding from latents
        X1 = jnp.array(latents, dtype=jnp.int16).view(jnp.bfloat16) * scaling_factor

        if not jnp.all(jnp.isfinite(X1)):
            print(f"Skipping step {step}: NaN detected in latents")
            continue

        X1 = X1 - latent_mean[None, :, None, None]

        key, sub_key = jr.split(key)
        # because jax sharding treats the shape as if it were on one GIGA-GPU
        # I have to reshard this
        X0 = jax.random.normal(key, shape=X1.shape, dtype=X1.dtype)
        X0 = jax.device_put(X0, data_sharding)

        # and the same idea for X0 is needed here
        key, sub_key = jr.split(key)
        eps = jax.random.normal(key, shape=[B])
        eps = jax.device_put(eps, data_sharding)

        # t inherits the sharding from eps
        t = jax.nn.sigmoid(eps)
        t_mult = t[:, None, None, None]

        Xt = t_mult * X1 + (1 - t_mult) * X0

        state, loss = step_model(
            state, optimizer, Xt, X1 - X0, t, labels, model_sharding, data_sharding
        )
        model, _opt_state = state

        if cfg.wandb.enabled and (step + 1) % cfg.train.every_n_steps == 0:
            wandb.log(
                {
                    "train_step": step + 1,
                    "train/loss": loss,
                }
            )
        if (step + 1) % cfg.train.every_n_ema == 0:
            model_ema = update_ema(model_ema, model, decay=cfg.train.ema_decay)
        if (step + 1) % cfg.train.every_n_checkpoint == 0:
            save_args = orbax.checkpoint.args.Composite(
                model=orbax.checkpoint.args.StandardSave(state),
                model_ema=orbax.checkpoint.args.StandardSave(model_ema),
                dataset=grain.PyGrainCheckpointSave(data_iterator),
            )
            checkpoint_manager.save(step + 1, args=save_args)
        if cfg.wandb.enabled and (step + 1) % cfg.train.every_n_image == 0:
            jax.block_until_ready(state)
            end_time = time.time()

            generated_latents = generate_samples(
                model_ema,
                validation_noise,
                validation_labels,
                model_sharding,
                data_sharding,
            )

            generated_latents = generated_latents + latent_mean[None, :, None, None]
            generated_latents = generated_latents / vae.config.scaling_factor
            generated_latents = jax.device_get(generated_latents)

            decode_start_time = time.time()
            generated_latents = np.array(generated_latents, copy=True)
            generated_latents = (
                torch.from_numpy(generated_latents).to("cpu").to(dtype=torch.bfloat16)
            )

            # [B,T,C,H,W]
            generated_latents = generated_latents.view(-1, 16, 32, 32)
            with torch.inference_mode():
                decoded_images = vae.decode(generated_latents)[0]
            decoded_images = rearrange(
                decoded_images,
                "(b t) c h w -> c (b h) (t w)",
                b=validation_noise.shape[0],
            )

            decoded_images = (
                decoded_images.permute(1, 2, 0).clip(0, 1).float().numpy() * 255.0
            )

            print(
                f"Time to decode latents fully on cpu: {(time.time() - decode_start_time):.4f} s."
            )
            if start_time is not None:
                wandb.log(
                    {f"train/{cfg.train.every_n_image} time": end_time - start_time}
                )

            wandb.log(
                {"image/examples": wandb.Image(decoded_images, caption="Euler Solver")}
            )
            start_time = time.time()

        if step >= cfg.train.total_steps:
            break

    return state


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    devices = jax.devices()
    mesh = jshard.Mesh(devices, axis_names=("data",))

    key = jr.PRNGKey(cfg.train.seed)

    ckpt_dir = os.path.abspath(cfg.train.checkpoint_dir)
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        ckpt_dir, options=options, item_names=("model", "dataset")
    )

    model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())
    data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("data"))

    dataloader = hydra.utils.instantiate(cfg.data)
    data_iterator = iter(dataloader)

    model = hydra.utils.instantiate(cfg.model, key=key)
    optimizer = hydra.utils.instantiate(cfg.optim)

    model_ema = jax.tree_util.tree_map(
        lambda x: jnp.copy(x) if eqx.is_inexact_array(x) else x, model
    )

    mask = get_trainable_mask(model)
    params, _ = eqx.partition(model, mask)
    opt_state = optimizer.init(params)

    # Place the VAE on CPU so it doesn't consume GPU VRAM.
    vae = hydra.utils.instantiate(cfg.vae).to("cpu")

    state = (model, opt_state)
    state = eqx.filter_shard(state, model_sharding)

    model_ema = eqx.filter_shard(model_ema, model_sharding)

    step_start = 0
    if checkpoint_manager.latest_step() is not None:
        print(
            f"Restoring previous trianing at step {checkpoint_manager.latest_step()}/{cfg.train.total_steps}"
        )
        restore_args = orbax.checkpoint.args.Composite(
            model=orbax.checkpoint.args.StandardRestore(item=state),
            dataset=grain.PyGrainCheckpointRestore(data_iterator),
        )
        restored = checkpoint_manager.restore(
            checkpoint_manager.latest_step(), args=restore_args
        )
        state = restored["model"]
        step_start = checkpoint_manager.latest_step()

    train(
        state,
        model_ema,
        optimizer,
        data_iterator,
        vae,
        cfg,
        model_sharding,
        data_sharding,
        key,
        checkpoint_manager,
        step_start,
    )


if __name__ == "__main__":
    main()
