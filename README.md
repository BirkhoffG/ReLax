# CFNet: An Algorithmic Recourse Library in Jax
> A fast and scalable library for counterfactual explanations in Jax.


## Key Features

- **fast**: code runs significantly faster than existing CF explanation libraries.
- **scalable**: code can be accelerated over *CPU*, *GPU*, and *TPU*
- **flexible**: we provide flexible API for researchers to allow full customization.


TODO: 
- implement various methods of CF explanations


## Install

`cfnet` is built on top of [Jax](https://jax.readthedocs.io/en/latest/index.html). It also uses [Pytorch](https://pytorch.org/) to load data.

### Running on CPU

If you only need to run `cfnet` on CPU, you can simply install via `pip` or clone the `GitHub` project.

Installation via PyPI:
```bash
pip install cfnet
```

Editable Install:
```bash
git clone https://github.com/BirkhoffG/cfnet.git
pip install -e cfnet
```

### Running on GPU or TPU

If you wish to run `cfnet` on GPU or TPU, please first install this library via `pip install cfnet`.

Then, you should install the right GPU or TPU version of Jax by following steps in the [install guidelines](https://github.com/google/jax#installation).



## A Minimum Example

```
from cfnet.utils import load_json
from cfnet.datasets import TabularDataModule
from cfnet.training_module import PredictiveTrainingModule
from cfnet.train import train_model
from cfnet.methods import VanillaCF
from cfnet.evaluate import generate_cf_results_local_exp, benchmark_cfs
from cfnet.import_essentials import *

data_configs = {
    "data_dir": "assets/data/s_adult.csv",
    "data_name": "adult",
    "batch_size": 256,
    "continous_cols": ["age","hours_per_week"],
    "discret_cols": ["workclass","education","marital_status","occupation","race","gender"],
    "imutable_cols": ["race","gender"]
}
m_configs = {
    'lr': 0.003,
    "sizes": [50, 10, 50],
    "dropout_rate": 0.3
}
t_configs = {
    'n_epochs': 10,
    'monitor_metrics': 'val/val_loss',
    'logger_name': 'pred'
}
cf_configs = {
    'n_steps': 1000,
    'lr': 0.001
}

# load data
dm = TabularDataModule(data_configs)

# specify the ML model 
training_module = PredictiveTrainingModule(m_configs)

# train ML model
params, opt_state = train_model(
    training_module, dm, t_configs
)

# define CF Explanation Module
pred_fn = lambda x: training_module.forward(
    params, random.PRNGKey(0), x, is_training=False)
cf_exp = VanillaCF(cf_configs)

# generate cf explanations
cf_results = generate_cf_results_local_exp(cf_exp, dm, pred_fn)

# benchmark different cf explanation methods
benchmark_cfs([cf_results])
```
