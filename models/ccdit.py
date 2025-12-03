import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Float, Int
from einops import rearrange

from .dit_block import DiTBlock


class SinusoidalTimeEmbedding(eqx.Module):
    emb: Float[Array, "half_dim"]
    dim: int = eqx.field(static=True)
    half_dim: int = eqx.field(static=True)

    def __init__(self, dim: int):
        half_dim = dim // 2
        scale = jnp.log(10000.0) / (half_dim - 1)
        freqs = jnp.exp(jnp.arange(half_dim) * -scale)

        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "half_dim", half_dim)
        object.__setattr__(self, "emb", freqs)

    def __call__(self, t: Float[Array, ""]) -> Float[Array, "dim"]:
        # t: scalar
        emb = t * self.emb
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class DiT(eqx.Module):
    dit_blocks: list[DiTBlock]
    layer_norm: eqx.nn.LayerNorm
    patchify: eqx.nn.Conv2d
    cond_proj: eqx.nn.Embedding
    time_proj: SinusoidalTimeEmbedding
    linear_out: eqx.nn.Linear
    pos_embed: jax.Array
    p: int = eqx.field(static=True)
    image_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim,
        dim,
        cond_dim,
        time_dim,
        num_heads,
        mlp_ratio,
        num_blocks,
        patch_size,
        num_classes,
        image_size,
        key: PRNGKeyArray,
    ):
        key1, key2, key3, key4 = jr.split(key, 4)

        self.patchify = eqx.nn.Conv2d(
            in_dim,
            dim,
            kernel_size=[patch_size, patch_size],
            padding=[0, 0],
            stride=[patch_size, patch_size],
            key=key1,
        )

        self.cond_proj = eqx.nn.Embedding(num_classes, cond_dim, key=key2)
        self.time_proj = SinusoidalTimeEmbedding(time_dim)

        dit_keys = jr.split(key3, num_blocks)
        self.dit_blocks = [
            DiTBlock(dim, cond_dim, time_dim, num_heads, mlp_ratio, key=dit_keys[i])
            for i in range(num_blocks)
        ]

        self.layer_norm = eqx.nn.LayerNorm(dim)

        reshape_dim = in_dim * patch_size**2
        self.linear_out = eqx.nn.Linear(dim, reshape_dim, key=key4)
        self.p = patch_size

        N = (image_size // patch_size) ** 2
        self.image_size = image_size
        self.pos_embed = jnp.zeros((N, dim))

    def __call__(
        self,
        x: Float[Array, "in_dim height width"],
        t: Float[Array, ""],
        label: Int[Array, ""],
    ) -> Float[Array, "in_dim height width"]:
        _, H, W = x.shape
        assert (
            H == self.image_size and W == self.image_size
        ), "Passed in images must match initialiation"
        p = self.p
        h = H // p
        w = W // p

        time_embed = self.time_proj(t)
        class_embed = self.cond_proj(label)

        # [in_dim,H,W] -> [C,N]  N:=(H//P)(W//P)
        x = self.patchify(x)
        x = rearrange(x, "c h w -> (h w) c")

        x = x + self.pos_embed

        for block in self.dit_blocks:
            x = block(x, time_embed, class_embed)

        x = jax.vmap(self.layer_norm)(x)

        # [N,C] -> [N,p*p*in_dim]
        x = jax.vmap(self.linear_out)(x)
        x = rearrange(x, "(h w) (p1 p2 c) -> c (h p1) (w p2)", h=h, w=w, p1=p, p2=p)

        return x
