import optax


def build_optimizer(lr, max_grad, warmup_steps, weight_decay, mask):
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0, end_value=lr, transition_steps=warmup_steps
            ),
            optax.constant_schedule(lr),
        ],
        boundaries=[warmup_steps],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad),
        optax.adamw(learning_rate=schedule, eps=1e-15, weight_decay=weight_decay, mask=mask),
    )
    return optimizer
