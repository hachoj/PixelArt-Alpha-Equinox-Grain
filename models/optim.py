import optax


def build_optimizer(lr):
    return optax.adamw(learning_rate=lr)
