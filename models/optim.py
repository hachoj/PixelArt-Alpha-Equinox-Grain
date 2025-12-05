import optax


def build_optimizer(lr, max_grad):
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad), optax.adamw(learning_rate=lr)
    )
    return optimizer
