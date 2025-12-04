import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Float


class MHSA(eqx.Module):
    layer_norm: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention

    def __init__(self, dim, num_heads, key: PRNGKeyArray):
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads"
        self.layer_norm = eqx.nn.LayerNorm(
            dim, use_weight=False, use_bias=False, dtype=jnp.bfloat16
        )
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=dim, key=key, dtype=jnp.bfloat16
        )

    def __call__(
        self,
        x: Float[Array, "num_patches embed_dim"],
        gamma: Float[Array, "embed_dim"],
        beta: Float[Array, "embed_dim"],
        alpha: Float[Array, "embed_dim"],
    ) -> Float[Array, "num_patches embed_dim"]:

        residual = x
        x = jax.vmap(self.layer_norm)(x)
        x = x * (1 + gamma) + beta
        x = self.attention(x, x, x)
        return alpha * x + residual
