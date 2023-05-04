# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/methods/01_vanilla.ipynb.

# %% ../../nbs/methods/01_vanilla.ipynb 3
from __future__ import annotations
from ..import_essentials import *
from .base import BaseCFModule
from ..utils import *

# %% auto 0
__all__ = ['VanillaCFConfig', 'VanillaCF']

# %% ../../nbs/methods/01_vanilla.ipynb 4
@auto_reshaping('x')
def _vanilla_cf(
    x: jnp.DeviceArray,  # `x` shape: (k,), where `k` is the number of features
    pred_fn: Callable[[jnp.DeviceArray], jnp.DeviceArray],  # y = pred_fn(x)
    n_steps: int,
    lr: float,  # learning rate for each `cf` optimization step
    lambda_: float,  #  loss = validity_loss + lambda_params * cost
    apply_fn: Callable
) -> jnp.DeviceArray:  # return `cf` shape: (k,)
    @jit
    def loss_fn_1(cf_y: Array, y_prime: Array):
        return jnp.mean(binary_cross_entropy(preds=cf_y, labels=y_prime))

    @jit
    def loss_fn_2(x: Array, cf: Array):
        return jnp.mean(optax.l2_loss(cf, x))

    @partial(jit, static_argnums=(2,))
    def loss_fn(
        cf: Array,  # `cf` shape: (k, 1)
        x: Array,  # `x` shape: (k, 1)
        pred_fn: Callable[[Array], Array],
    ):
        y_pred = pred_fn(x)
        y_prime = 1.0 - y_pred
        cf_y = pred_fn(cf)
        return loss_fn_1(cf_y, y_prime) + lambda_ * loss_fn_2(x, cf)

    @loop_tqdm(n_steps)
    def gen_cf_step(
        i, cf_opt_state: Tuple[Array, optax.OptState] #x: Array, cf: Array, opt_state: optax.OptState
    ) -> Tuple[jnp.DeviceArray, optax.OptState]:
        cf, opt_state = cf_opt_state
        cf_grads = jax.grad(loss_fn)(cf, x, pred_fn)
        cf, opt_state = grad_update(cf_grads, cf, opt_state, opt)
        cf = apply_fn(x, cf, hard=False)
        return cf, opt_state

    cf = jnp.array(x, copy=True)
    opt = optax.rmsprop(lr)
    opt_state = opt.init(cf)
    # for _ in tqdm(range(n_steps)):
    #     cf, opt_state = gen_cf_step(x, cf, opt_state)
    cf, opt_state = lax.fori_loop(0, n_steps, gen_cf_step, (cf, opt_state))

    cf = apply_fn(x, cf, hard=True)
    return cf


# %% ../../nbs/methods/01_vanilla.ipynb 5
class VanillaCFConfig(BaseParser):
    n_steps: int = 1000
    lr: float = 0.001
    lambda_: float = 0.01  # loss = validity_loss + lambda_ * cost


# %% ../../nbs/methods/01_vanilla.ipynb 6
class VanillaCF(BaseCFModule):
    name = "VanillaCF"

    def __init__(
        self,
        configs: dict | VanillaCFConfig = None
    ):
        if configs is None:
            configs = VanillaCFConfig()
        self.configs = validate_configs(configs, VanillaCFConfig)

    def generate_cf(
        self,
        x: jnp.ndarray,  # `x` shape: (k,), where `k` is the number of features
        pred_fn: Callable[[jnp.DeviceArray], jnp.DeviceArray],
    ) -> jnp.DeviceArray:
        return _vanilla_cf(
            x=x,  # `x` shape: (k,), where `k` is the number of features
            pred_fn=pred_fn,  # y = pred_fn(x)
            n_steps=self.configs.n_steps,
            lr=self.configs.lr,  # learning rate for each `cf` optimization step
            lambda_=self.configs.lambda_,  #  loss = validity_loss + lambda_params * cost
            apply_fn=self.data_module.apply_constraints
        )

    def generate_cfs(
        self,
        X: jnp.DeviceArray,  # `x` shape: (b, k), where `b` is batch size, `k` is the number of features
        pred_fn: Callable[[jnp.DeviceArray], jnp.DeviceArray],
        is_parallel: bool = False,
    ) -> jnp.DeviceArray:
        def _generate_cf(x: jnp.DeviceArray) -> jnp.ndarray:
            return self.generate_cf(x, pred_fn)

        return (
            jax.vmap(_generate_cf)(X) if not is_parallel else jax.pmap(_generate_cf)(X)
        )

