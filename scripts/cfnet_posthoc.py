from cfnet.import_essentials import *
from cfnet.datasets import TabularDataModule, MinMaxScaler, OneHotEncoder, find_imutable_idx_list, NumpyDataset
from cfnet.train import train_model
from cfnet.training_module import CounterNetTrainingModule, PredictiveTrainingModule
from cfnet.evaluate import CFExplanationResults, benchmark_cfs, DEFAULT_METRICS
from cfnet.interfaces import BaseCFExplanationModule
from cfnet.utils import load_json
from copy import deepcopy
from .utils_configs import get_configs
jax.config.update('jax_platform_name', 'cpu')



class TabularDataModulePosthoc(TabularDataModule):
    def __init__(self, data_configs: Dict, pred_fn: Callable[[jnp.DeviceArray], jnp.DeviceArray]):
        self.pred_fn = pred_fn
        super().__init__(data_configs)

    def prepare_data(self):
        def split_x_and_y(data: pd.DataFrame):
            X = data[data.columns[:-1]]
            y = data[[data.columns[-1]]]
            return X, y

        X, y = split_x_and_y(self.data)

        # preprocessing
        if self.normalizer:
            X_cont = self.normalizer.transform(X[self.continous_cols])
        else:
            self.normalizer = MinMaxScaler()
            X_cont = self.normalizer.fit_transform(
                X[self.continous_cols]) if self.continous_cols else np.array([[] for _ in range(len(X))])

        if self.encoder:
            X_cat = self.encoder.transform(X[self.discret_cols])
        else:
            self.encoder = OneHotEncoder(sparse=False)
            X_cat = self.encoder.fit_transform(
                X[self.discret_cols]) if self.discret_cols else np.array([[] for _ in range(len(X))])
        X = np.concatenate((X_cont, X_cat), axis=1)
        # get categorical arrays
        self.cat_arrays = self.encoder.categories_ if self.discret_cols else []
        self.imutable_idx_list = find_imutable_idx_list(
            imutable_col_names=self.imutable_cols,
            discrete_col_names=self.discret_cols,
            continuous_col_names=self.continous_cols,
            cat_arrays=self.cat_arrays
        )
        y = self.pred_fn(X)
        y = np.array(y)
        # y = np.round(y)
        
        # prepare train & test
        train_test_tuple = train_test_split(X, y, shuffle=False)
        train_X, test_X, train_y, test_y = map(lambda x: x.astype(jnp.float32), train_test_tuple)
        if self.sample_frac:
            train_size = int(len(train_X) * self.sample_frac)
            train_X, train_y = train_X[:train_size], train_y[:train_size]

        self.train_dataset = NumpyDataset(train_X, train_y)
        self.val_dataset = NumpyDataset(test_X, test_y)
        self.test_dataset = self.val_dataset


def generate_cf_results_cfnet_posthoc(
    cf_module: BaseCFExplanationModule,
    dm: TabularDataModule, # pass normal dm, NOT dm_posthoc
    pred_fn: Optional[Callable[[jnp.DeviceArray], jnp.DeviceArray]] = None,
    params: Optional[hk.Params] = None, # params of cfnet
    rng_key: Optional[random.PRNGKey] = None
) -> CFExplanationResults:
    # validate arguments
    if (pred_fn is None) and (params is None) and (rng_key is None):
        raise ValueError("A valid `pred_fn: Callable[jnp.DeviceArray], jnp.DeviceArray]` or `params: hk.Params` needs to be passed.")
    # prepare
    X, y = dm.test_dataset[:]
    cf_module.update_cat_info(dm)
    # generate cfs
    current_time = time.time()
    cfs = cf_module.generate_cfs(X, params, rng_key)
    total_time = time.time() - current_time

    return CFExplanationResults(
        cf_name=cf_module.name, data_module=dm,
        cfs=cfs, total_time=total_time,
        pred_fn=pred_fn
    )


mlp_t_configs = {
    'n_epochs': 10,
    'monitor_metrics': 'val/val_loss',
    'logger_name': 'pred'
}
cfnet_t_configs = {
    'n_epochs': 100,
    'monitor_metrics': 'val/val_loss',
    'logger_name': 'pred'
}


def main():
    configs_list = get_configs('student')
    cf_results_list = []

    for configs in configs_list:
        dm = TabularDataModule(configs['data_configs'])
        mlp = PredictiveTrainingModule(configs['mlp_configs'])
        cfnet = CounterNetTrainingModule(configs['cfnet_configs'])
        
        params, _ = train_model(
            mlp, dm, mlp_t_configs
        )

        _params = deepcopy(params)
        pred_fn = lambda x: mlp.forward(_params, random.PRNGKey(0), x, is_training=False)

        dm_posthoc = TabularDataModulePosthoc(
            configs['data_configs'], pred_fn
        )

        cfnet_params, _ = train_model(
            cfnet, dm_posthoc, cfnet_t_configs
        )

        cf_results = generate_cf_results_cfnet_posthoc(
            cfnet, dm, pred_fn, cfnet_params, jax.random.PRNGKey(0)
        )
        cf_results_list.append(cf_results)

    benchmark_df = benchmark_cfs(cf_results_list, DEFAULT_METRICS + ['manifold_dist', 'sparsity'])
    # benchmark_df.to_csv('result.csv')
    print(benchmark_df)


if __name__ == "__main__":
    main()


