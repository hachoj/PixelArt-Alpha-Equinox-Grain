import optax


def build_optimizer(lr, max_grad, warmup_steps, weight_decay):
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0, end_value=lr, transition_steps=warmup_steps
            ),
            optax.constant_schedule(lr),
        ],
        boundaries=[warmup_steps],
    )
    return schedule
