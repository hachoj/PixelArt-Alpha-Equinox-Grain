import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Float


class QKNormedAttention(eqx.Module):
    num_heads: int = eqx.field(static=True)
    query_size: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    q_norm: eqx.nn.RMSNorm
    k_norm: eqx.nn.RMSNorm

    def __init__(self, num_heads, dim, key, dtype):
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads"
        self.num_heads = num_heads
        self.query_size = dim // num_heads
        self.scale = self.query_size**-0.5

        key1, key2, key3, key4 = jr.split(key, 4)

        self.q_proj = eqx.nn.Linear(dim, dim, key=key1, dtype=dtype)
        self.k_proj = eqx.nn.Linear(dim, dim, key=key2, dtype=dtype)
        self.v_proj = eqx.nn.Linear(dim, dim, key=key3, dtype=dtype)
        self.out_proj = eqx.nn.Linear(dim, dim, key=key4, dtype=dtype)

        self.q_norm = eqx.nn.RMSNorm(dim, dtype=dtype)
        self.k_norm = eqx.nn.RMSNorm(dim, dtype=dtype)

    def __call__(
        self,
        query: Float[Array, "num_patches embed_dim"],
        key: Float[Array, "num_patches embed_dim"],
        value: Float[Array, "num_patches embed_dim"],
    ):
        q = jax.vmap(self.q_proj)(query)
        k = jax.vmap(self.k_proj)(key)
        v = jax.vmap(self.v_proj)(value)

        q = jax.vmap(self.q_norm)(q)
        k = jax.vmap(self.k_norm)(k)

        n, _ = q.shape
        q = q.reshape(n, self.num_heads, -1)
        k = k.reshape(n, self.num_heads, -1)
        v = v.reshape(n, self.num_heads, -1)

        attn_output = jax.nn.dot_product_attention(q, k, v, scale=self.scale)

        attn_output = attn_output.reshape(n, -1)
        out = jax.vmap(self.out_proj)(attn_output)
        return out


class MHSA(eqx.Module):
    layer_norm: eqx.nn.LayerNorm
    attention: QKNormedAttention

    def __init__(self, dim, num_heads, key: PRNGKeyArray):
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads"
        self.layer_norm = eqx.nn.LayerNorm(
            dim, use_weight=False, use_bias=False, dtype=jnp.bfloat16
        )
        self.attention = QKNormedAttention(
            num_heads=num_heads, dim=dim, key=key, dtype=jnp.bfloat16
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
