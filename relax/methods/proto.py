# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/05c_methods.prototype.ipynb.

# %% ../../nbs/05c_methods.prototype.ipynb 3
from __future__ import annotations
from ..import_essentials import *
from .base import BaseCFModule, BaseParametricCFModule
from ..data import TabularDataModule
from ..module import BaseTrainingModule, MLP
from ..trainer import train_model, TrainingConfigs
from ..utils import validate_configs, binary_cross_entropy, make_model, init_net_opt, grad_update
from functools import partial

# %% auto 0
__all__ = ['ProtoCFConfig', 'ProtoCF']

# %% ../../nbs/05c_methods.prototype.ipynb 4
class AEConfigs(BaseParser):
    enc_sizes: List[int]
    dec_sizes: List[int]
    dropout_rate: float = 0.3
    lr: float = 0.001

class AE(hk.Module):
    def __init__(
        self,
        m_config: Dict[str, Any],
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.configs = validate_configs(m_config, AEConfigs) #PredictiveModelConfigs(**m_config)

    def __call__(
        self,
        x: jnp.ndarray,
        is_training: bool = True
    ) -> jnp.ndarray:
        input_shape = x.shape[-1]
        z = MLP(sizes=self.configs.enc_sizes, dropout_rate=self.configs.dropout_rate, name='Encoder')(x, is_training)
        x = MLP(sizes=self.configs.enc_sizes, dropout_rate=self.configs.dropout_rate, name='Decoder')(z, is_training)
        x = hk.Linear(input_shape, name='Decoder')(x)
        return x, z

# %% ../../nbs/05c_methods.prototype.ipynb 5
class AETrainingModule(BaseTrainingModule):
    def __init__(
        self,
        m_configs: Dict[str, Any]
    ):
        self.save_hyperparameters(m_configs)
        self.net = make_model(m_configs, AE)
        self.configs = validate_configs(m_configs, AEConfigs)
        # self.configs = PredictiveTrainingModuleConfigs(**m_configs)
        self.opt = optax.adam(learning_rate=self.configs.lr)

    def init_net_opt(self, data_module, key):
        X, _ = data_module.train_dataset[:100]
        params, opt_state = init_net_opt(
            self.net, self.opt, X=X, key=key
        )
        return params, opt_state

    @partial(jax.jit, static_argnames=['self', 'is_training'])
    def forward(self, params, rng_key, x, is_training: bool = True):
        return self.net.apply(params, rng_key, x, is_training = is_training)

    def encode(self, params, rng_key, x):
        _, z = self.forward(params, rng_key, x, is_training=False)
        return z

    def loss_fn(self, params, rng_key, batch, is_training=True):
        x, y = batch
        x_hat, z = self.forward(params, rng_key, x, is_training)
        return jnp.mean(vmap(optax.l2_loss)(x, x_hat))

    @partial(jax.jit, static_argnames=['self'])
    def _training_step(self, params, opt_state, rng_key, batch):
        grads = jax.grad(self.loss_fn)(params, rng_key, batch)
        upt_params, opt_state = grad_update(grads, params, opt_state, self.opt)
        return upt_params, opt_state

    def training_step(
        self,
        params: hk.Params,
        opt_state: optax.OptState,
        rng_key: random.PRNGKey,
        batch: Tuple[jnp.array, jnp.array]
    ) -> Tuple[hk.Params, optax.OptState]:
        upt_params, opt_state = self._training_step(params, opt_state, rng_key, batch)

        loss = self.loss_fn(params, rng_key, batch)
        self.log_dict({
            'train/train_loss_1': loss.item()
        })
        return params, opt_state

    def validation_step(self, params, rng_key, batch):
        x, y = batch
        loss = self.loss_fn(params, rng_key, batch, is_training=False)
        logs = {
            'val/val_loss': loss.item(),
        }
        self.log_dict(logs)

# %% ../../nbs/05c_methods.prototype.ipynb 6
def _proto_cf(
    x: jnp.DeviceArray, # `x` shape: (k,), where `k` is the number of features
    pred_fn: Callable[[jnp.DeviceArray], jnp.DeviceArray], # y = pred_fn(x)
    n_steps: int,
    lr: float, # learning rate for each `cf` optimization step
    lambda_: float, #  loss = validity_loss + lambda_params * cost
    ae: AETrainingModule,
    ae_params: hk.Params,
    sampled_data_pos: jnp.DeviceArray,
    sampled_data_neg: jnp.DeviceArray,
    sampled_label: jnp.DeviceArray,
    apply_constraints_fn: Callable
) -> jnp.DeviceArray: # return `cf` shape: (k,)
    def proto(data):
        return ae.encode(ae_params, jax.random.PRNGKey(0), data)

    def loss_fn_1(cf_y: jnp.DeviceArray, y_prime: jnp.DeviceArray):
        return jnp.mean(binary_cross_entropy(y_pred=cf_y, y=y_prime))

    def loss_fn_2(x: jnp.DeviceArray, cf: jnp.DeviceArray):
        return jnp.mean(optax.l2_loss(cf, x)) + 0.1 * jnp.mean(jnp.mean(jnp.abs(x - cf)))

    def loss_fn_3(cf, data):
        error = proto(cf) - proto(data)
        return jnp.mean(0.5 * (error) ** 2)

    def loss_fn(
        cf: jnp.DeviceArray, # `cf` shape: (k, 1)
        x: jnp.DeviceArray,  # `x` shape: (k, 1)
        pred_fn: Callable[[jnp.DeviceArray], jnp.DeviceArray]
    ):
        y_pred = pred_fn(x)
        y_prime = 1. - y_pred
        cf_y = pred_fn(cf)

        y_prime_round = jnp.mean(jnp.round(y_prime))

        # print(sampled_label.shape)
        # print(y_prime.shape)
        return loss_fn_1(cf_y, y_prime) + loss_fn_2(x, cf) \
            + loss_fn_3(cf, sampled_data_pos) * y_prime_round + loss_fn_3(cf, sampled_data_neg) * (1 - y_prime_round)

    @jax.jit
    def gen_cf_step(
        x: jnp.DeviceArray, cf: jnp.DeviceArray, opt_state: optax.OptState
    ) -> Tuple[jnp.DeviceArray, optax.OptState]:
        cf_grads = jax.grad(loss_fn)(cf, x, pred_fn)
        cf, opt_state = grad_update(cf_grads, cf, opt_state, opt)
        cf = apply_constraints_fn(x, cf, hard=False)
        cf = jnp.clip(cf, 0., 1.)
        return cf, opt_state

    x_size = x.shape
    if len(x_size) > 1 and x_size[0] != 1:
        raise ValueError(f"""Invalid Input Shape: Require `x.shape` = (1, k) or (k, ),
but got `x.shape` = {x.shape}. This method expects a single input instance.""")
    if len(x_size) == 1:
        x = x.reshape(1, -1)
    cf = jnp.array(x, copy=True)
    opt = optax.rmsprop(lr)
    opt_state = opt.init(cf)
    for _ in tqdm(range(n_steps)):
        cf, opt_state = gen_cf_step(x, cf, opt_state)

    cf = apply_constraints_fn(x, cf, hard=True)
    return cf.reshape(x_size)

# %% ../../nbs/05c_methods.prototype.ipynb 7
class ProtoCFConfig(BaseParser):
    
    n_steps: int = 1000
    lr: float = 0.01
    lambda_: float = 0.01 # loss = validity_loss + lambda_params * cost
    ae_configs: Dict[str, Any] = {
        "enc_sizes": [50, 10],
        "dec_sizes": [10, 50],
        "dropout_rate": 0.3,
        'lr': 0.03,
    }


# %% ../../nbs/05c_methods.prototype.ipynb 8
class ProtoCF(BaseCFModule, BaseParametricCFModule):
    name = "ProtoCF"
    _ae_params: hk.Params = None
    _ae_module: AETrainingModule

    def __init__(
        self, 
        configs: Dict | ProtoCFConfig = None
    ):
        if configs is None:
            configs = ProtoCFConfig()
        self.configs = validate_configs(configs, ProtoCFConfig)

    def train(
        self, 
        data_module: TabularDataModule, # data module
        t_configs: TrainingConfigs | dict = None # training configs
    ):
        _default_t_configs = dict(n_epochs=10, batch_size=128)
        if t_configs is None: 
            t_configs = _default_t_configs
        t_configs = validate_configs(t_configs, TrainingConfigs)
        # train autoencoder
        self._ae_module = AETrainingModule(self.configs.ae_configs)
        self._ae_params, _ = train_model(self._ae_module, data_module, t_configs)

        sampled_data, sampled_label = next(iter(data_module.train_dataloader(t_configs.batch_size)))
        self.sampled_data, self.sampled_label = map(jnp.array, (sampled_data, sampled_label))

        self.sampled_pos = self.sampled_data[(self.sampled_label == 1.).reshape(-1), :]
        self.sampled_neg = self.sampled_data[(self.sampled_label == 0.).reshape(-1), :]

    def _is_module_trained(self) -> bool: 
        return not (self._ae_params is None)
    
    @deprecated(removed_in='0.1.0', deprecated_in='0.0.11')
    def update_cat_info(self, data_module: TabularDataModule):
        sampled_data, sampled_label = next(iter(data_module.train_dataloader()))
        self.sampled_data, self.sampled_label = map(jnp.array, (sampled_data, sampled_label))

        self.sampled_pos = self.sampled_data[(self.sampled_label == 1.).reshape(-1), :]
        self.sampled_neg = self.sampled_data[(self.sampled_label == 0.).reshape(-1), :]
        self.ae = AETrainingModule(self.configs.ae_configs)
        self.ae_params, _ = train_model(self.ae, data_module, self.configs.ae_t_configs)
        return super().update_cat_info(data_module)

    def generate_cf(
        self,
        x: jnp.ndarray, # `x` shape: (k,), where `k` is the number of features
        pred_fn: Callable[[jnp.DeviceArray], jnp.DeviceArray]
    ) -> jnp.DeviceArray:
        return _proto_cf(
            x= x, # `x` shape: (k,), where `k` is the number of features
            pred_fn=pred_fn, # y = pred_fn(x)
            n_steps=self.configs.n_steps,
            lr=self.configs.lr, # learning rate for each `cf` optimization step
            lambda_=self.configs.lambda_, #  loss = validity_loss + lambda_params * cost
            ae=self._ae_module,
            ae_params=self._ae_params,
            sampled_data_pos=self.sampled_pos,
            sampled_data_neg=self.sampled_neg,
            sampled_label=self.sampled_label,
            apply_constraints_fn=self.data_module.apply_constraints            
        )

    def generate_cfs(
        self,
        X: jnp.DeviceArray, # `x` shape: (b, k), where `b` is batch size, `k` is the number of features
        pred_fn: Callable[[jnp.DeviceArray], jnp.DeviceArray],
        is_parallel: bool = False
    ) -> jnp.DeviceArray:
        def _generate_cf(x: jnp.DeviceArray) -> jnp.ndarray:
            return self.generate_cf(x, pred_fn)
        return jax.vmap(_generate_cf)(X) if not is_parallel else jax.pmap(_generate_cf)(X)