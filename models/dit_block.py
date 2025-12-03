import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Float
from einops import rearrange

from .mhsa import MHSA
from .mlp import MLP


class DiTBlock(eqx.Module):
    mlp: MLP
    attention: MHSA
    gamma_beta_1: eqx.nn.Linear
    gamma_beta_2: eqx.nn.Linear
    alpha_1: eqx.nn.Linear
    alpha_2: eqx.nn.Linear

    def __init__(
        self, dim, cond_dim, time_dim, num_heads, mlp_ratio, key: PRNGKeyArray
    ):
        key1, key2, key3, key4, key5, key6 = jr.split(key, 6)

        self.attention = MHSA(dim, num_heads, key=key1)
        self.mlp = MLP(dim, mlp_ratio, key=key2)
        self.gamma_beta_1 = eqx.nn.Linear(time_dim + cond_dim, dim * 4, key=key3)
        self.alpha_1 = eqx.nn.Linear(time_dim + cond_dim, dim * 2, key=key5)

        alpha_2_temp = eqx.nn.Linear(dim * 2, dim * 2, key=key6)
        gamma_beta_2_temp = eqx.nn.Linear(dim * 4, dim * 4, key=key4)

        alpha_zeros_w = jnp.zeros_like(alpha_2_temp.weight)
        alpha_zeros_b = jnp.zeros_like(alpha_2_temp.bias)  # pyrefly:ignore
        gamma_beta_zeros_w = jnp.zeros_like(gamma_beta_2_temp.weight)
        gamma_beta_zeros_b = jnp.zeros_like(gamma_beta_2_temp.bias)  # pyrefly:ignore

        self.gamma_beta_2 = eqx.nn.Linear(dim * 4, dim * 4, key=key4)
        self.gamma_beta_2 = eqx.nn.Linear(dim * 4, dim * 4, key=key4)

        self.alpha_2 = eqx.tree_at(
            lambda l: (l.weight, l.bias), alpha_2_temp, (alpha_zeros_w, alpha_zeros_b)
        )
        self.gamma_beta_2 = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            gamma_beta_2_temp,
            (gamma_beta_zeros_w, gamma_beta_zeros_b),
        )

    def __call__(
        self,
        x: Float[Array, "num_patches embed_dim"],
        t: Float[Array, "time_dim"],
        c: Float[Array, "cond_dim"],
    ) -> Float[Array, "num_patches embed_dim"]:

        cond = jnp.concatenate([t, c], axis=-1)

        gamma_betas = self.gamma_beta_1(cond)
        gamma_betas = jax.nn.silu(gamma_betas)
        gamma_betas = self.gamma_beta_2(gamma_betas)

        alphas = self.alpha_1(cond)
        alphas = jax.nn.silu(alphas)
        alphas = self.alpha_2(alphas)

        gamma1, beta1, gamma2, beta2 = jnp.split(gamma_betas, 4, axis=0)
        alpha1, alpha2 = jnp.split(alphas, 2, axis=0)

        x = self.attention(x, gamma1, beta1, alpha1)
        x = self.mlp(x, gamma2, beta2, alpha2)
        return x
