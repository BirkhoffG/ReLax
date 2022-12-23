# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01_data.module.ipynb.

# %% ../../nbs/01_data.module.ipynb 3
from __future__ import annotations
from ..import_essentials import *
from ..utils import load_json, validate_configs, cat_normalize
from .loader import Dataset, DataLoader, _supported_backends
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.base import TransformerMixin
from urllib.request import urlretrieve

# %% auto 0
__all__ = ['BaseDataModule', 'find_imutable_idx_list', 'TabularDataModuleConfigs', 'TabularDataModule', 'sample', 'load_data']

# %% ../../nbs/01_data.module.ipynb 5
class BaseDataModule(ABC):
    """DataModule Interface"""

    @property
    @abstractmethod
    def data_name(self) -> str: 
        return
        
    @property
    @abstractmethod
    def data(self) -> Any:
        return
    
    @property
    @abstractmethod
    def train_dataset(self) -> Dataset:
        return
    
    @property
    @abstractmethod
    def val_dataset(self) -> Dataset:
        return

    @property
    @abstractmethod
    def test_dataset(self) -> Dataset:
        return

    def train_dataloader(self, batch_size):
        raise NotImplementedError

    def val_dataloader(self, batch_size):
        raise NotImplementedError

    def test_dataloader(self, batch_size):
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def transform(self, data) -> jnp.DeviceArray:
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, x: jnp.DeviceArray) -> Any:
        raise NotImplementedError

    def apply_constraints(
        self, 
        x: jnp.DeviceArray,
        cf: jnp.DeviceArray,
        hard: bool
    ) -> jnp.DeviceArray:
        return cf
    
    def apply_regularization(
        self, 
        x: jnp.DeviceArray,
        cf: jnp.DeviceArray,
        hard: bool
    ):
        raise NotImplementedError


# %% ../../nbs/01_data.module.ipynb 7
def find_imutable_idx_list(
    imutable_col_names: List[str],
    discrete_col_names: List[str],
    continuous_col_names: List[str],
    cat_arrays: List[List[str]],
) -> List[int]:
    imutable_idx_list = []
    for idx, col_name in enumerate(continuous_col_names):
        if col_name in imutable_col_names:
            imutable_idx_list.append(idx)

    cat_idx = len(continuous_col_names)

    for i, (col_name, cols) in enumerate(zip(discrete_col_names, cat_arrays)):
        cat_end_idx = cat_idx + len(cols)
        if col_name in imutable_col_names:
            imutable_idx_list += list(range(cat_idx, cat_end_idx))
        cat_idx = cat_end_idx
    return imutable_idx_list

# %% ../../nbs/01_data.module.ipynb 8
def _check_cols(data: pd.DataFrame, configs: TabularDataModuleConfigs) -> pd.DataFrame:
    data = data.astype({
        col: float for col in configs.continous_cols
    })
    
    cols = configs.continous_cols + configs.discret_cols
    # check target columns
    target_col = data.columns[-1]
    assert not target_col in cols, \
        f"continous_cols or discret_cols contains target_col={target_col}."
    
    # check imutable cols
    for col in configs.imutable_cols:
        assert col in cols, \
            f"imutable_cols=[{col}] is not specified in `continous_cols` or `discret_cols`."
    data = data[cols + [target_col]]
    return data


# %% ../../nbs/01_data.module.ipynb 9
def _process_data(
    df: pd.DataFrame | None, configs: TabularDataModuleConfigs
) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(configs.data_dir)
    elif isinstance(df, pd.DataFrame):
        df = df
    else:
        raise ValueError(f"{type(df).__name__} is not supported as an input type for `TabularDataModule`.")

    df = _check_cols(df, configs)
    return df

# %% ../../nbs/01_data.module.ipynb 10
def _transform_df(
    transformer: TransformerMixin,
    data: pd.DataFrame,
    cols: List[str] | None,
):
    return (
        transformer.transform(data[cols])
            if cols else np.array([[] for _ in range(len(data))])
    )

# %% ../../nbs/01_data.module.ipynb 12
def _inverse_transform_np(
    transformer: TransformerMixin,
    x: jnp.DeviceArray,
    cols: List[str] | None
):
    assert len(cols) <= x.shape[-1], \
        f"x.shape={x.shape} probably will not match len(cols)={len(cols)}"
    if cols:
        data = transformer.inverse_transform(x)
        return pd.DataFrame(data=data, columns=cols)
    else:
        return None


# %% ../../nbs/01_data.module.ipynb 15
def _init_scalar_encoder(
    data: pd.DataFrame,
    configs: TabularDataModuleConfigs
):  
    # fit scalar
    if configs.normalizer:
        scalar = configs.normalizer
    else:
        scalar = MinMaxScaler()
        if configs.continous_cols:
            scalar.fit(data[configs.continous_cols])
    
    # fit encoder
    if configs.encoder:
        encoder = configs.encoder
    else:
        encoder = OneHotEncoder(sparse=False)
        if configs.discret_cols:
            encoder.fit(data[configs.discret_cols])
    return dict(scalar=scalar, encoder=encoder)


# %% ../../nbs/01_data.module.ipynb 16
class TabularDataModuleConfigs(BaseParser):
    """Configurator of `TabularDataModule`."""

    data_dir: str = Field(description="The directory of dataset.")
    data_name: str = Field(description="The name of `TabularDataModule`.")
    continous_cols: List[str] = Field(
        [], description="Continuous features/columns in the data."
    )
    discret_cols: List[str] = Field(
        [], description="Categorical features/columns in the data."
    )
    imutable_cols: List[str] = Field(
        [], description="Immutable features/columns in the data."
    )
    normalizer: Optional[Any] = Field(
        None, description="Fitted scalar for continuous features."
    )
    encoder: Optional[Any] = Field(
        None, description="Fitted encoder for categorical features."
    )
    sample_frac: Optional[float] = Field(
        None, description="Sample fraction of the data. Default to use the entire data.", 
        ge=0., le=1.0
    )
    backend: str = Field(
        "jax", description=f"`Dataloader` backend. Currently supports: {_supported_backends()}"
    )


# %% ../../nbs/01_data.module.ipynb 20
class TabularDataModule(BaseDataModule):
    """DataModule for tabular data"""
    cont_scalar = None # scalar for normalizing continuous features
    cat_encoder = None # encoder for encoding categorical features

    def __init__(
        self, 
        data_config: dict | TabularDataModuleConfigs, # Configurator of `TabularDataModule`
        data: pd.DataFrame = None # Data in `pd.DataFrame`. If `data` is `None`, the DataModule will load data from `data_dir`.
    ):
        self._configs: TabularDataModuleConfigs = validate_configs(
            data_config, TabularDataModuleConfigs
        )
        self._data = _process_data(data, self._configs)
        # init idx lists
        self.cat_idx = len(self._configs.continous_cols)
        self._imutable_idx_list = []
        self.prepare_data()

    def prepare_data(self):
        scalar_encoder_dict = _init_scalar_encoder(
            data=self._data, configs=self._configs
        )
        self.cont_scalar = scalar_encoder_dict['scalar']
        self.cat_encoder = scalar_encoder_dict['encoder']
        X, y = self.transform(self.data)

        self._imutable_idx_list = find_imutable_idx_list(
            imutable_col_names=self._configs.imutable_cols,
            discrete_col_names=self._configs.discret_cols,
            continuous_col_names=self._configs.continous_cols,
            cat_arrays=self.cat_encoder.categories_,
        )
        
        # prepare train & test
        train_test_tuple = train_test_split(X, y, shuffle=False)
        train_X, test_X, train_y, test_y = map(
             lambda x: x.astype(float), train_test_tuple
         )
        if self._configs.sample_frac:
            train_size = int(len(train_X) * self._configs.sample_frac)
            train_X, train_y = train_X[:train_size], train_y[:train_size]
        
        self._train_dataset = Dataset(train_X, train_y)
        self._val_dataset = Dataset(test_X, test_y)
        self._test_dataset = self.val_dataset

    @property
    def data_name(self) -> str: 
        return self._configs.data_name
    
    @property
    def data(self) -> pd.DataFrame:
        """Loaded data in `pd.DataFrame`."""
        return self._data
    
    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset
    
    @property
    def val_dataset(self) -> Dataset:
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    def train_dataloader(self, batch_size):
        return DataLoader(self.train_dataset, self._configs.backend, 
            batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
        )

    def val_dataloader(self, batch_size):
        return DataLoader(self.val_dataset, self._configs.backend,
            batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
        )

    def test_dataloader(self, batch_size):
        return DataLoader(self.val_dataset, self._configs.backend,
            batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
        )

    def transform(
        self, 
        data: pd.DataFrame, # Data to be transformed to `numpy.ndarray`
    ) -> Tuple[np.ndarray, np.ndarray]: # Return `(X, y)`
        """Transform data into numerical representations."""
        # TODO: validate `data`
        X_cont = _transform_df(
            self.cont_scalar, data, self._configs.continous_cols
        )
        X_cat = _transform_df(
            self.cat_encoder, data, self._configs.discret_cols
        )
        X = np.concatenate((X_cont, X_cat), axis=1)
        y = data.iloc[:, -1:].to_numpy()
        
        return X, y

    def inverse_transform(
        self, 
        x: jnp.DeviceArray, # The transformed input to be scaled back
        y: jnp.DeviceArray = None # The transformed label to be scaled back. If `None`, the target columns will not be scaled back.
    ) -> pd.DataFrame: # Transformed `pd.DataFrame`. 
        """Scaled back into `pd.DataFrame`."""
        X_cont_df = _inverse_transform_np(
            self.cont_scalar, x[:, :self.cat_idx], self._configs.continous_cols
        )
        X_cat_df = _inverse_transform_np(
            self.cat_encoder, x[:, self.cat_idx:], self._configs.discret_cols
        )
        if y is not None:
            y_df = pd.DataFrame(data=y, columns=[self.data.columns[-1]])
        else:
            y_df = None
        
        return pd.concat(
            [X_cont_df, X_cat_df, y_df], axis=1
        )

    def apply_constraints(
        self, 
        x: jnp.DeviceArray, # input
        cf: jnp.DeviceArray, # Unnormalized counterfactuals
        hard: bool = False # Apply hard constraints or not
    ) -> jnp.DeviceArray:
        """Apply categorical normalization and immutability constraints"""
        cat_arrays = self.cat_encoder.categories_ \
            if self._configs.discret_cols else []
        cf = cat_normalize(
            cf, cat_arrays=cat_arrays, 
            cat_idx=len(self._configs.continous_cols),
            hard=hard
        )
        # apply immutable constraints
        if len(self._configs.imutable_cols) > 0:
            cf = cf.at[:, self._imutable_idx_list].set(x[:, self._imutable_idx_list])
        return cf

    def apply_regularization(
        self, 
        x: jnp.DeviceArray, # Input
        cf: jnp.DeviceArray, # Unnormalized counterfactuals
    ) -> float: # Return regularization loss
        """Apply categorical constraints by adding regularization terms"""
        reg_loss = 0.
        cat_arrays = self.cat_encoder.categories_
        cat_idx = len(self._configs.continous_cols)

        for col in cat_arrays:
            cat_idx_end = cat_idx + len(col)
            reg_loss += jnp.power(
                (jnp.sum(cf[cat_idx:cat_idx_end]) - 1.0), 2
            )
        return reg_loss


# %% ../../nbs/01_data.module.ipynb 41
def sample(datamodule: BaseDataModule, frac: float = 1.0): 
    X, y = datamodule.train_dataset[:]
    size = int(len(X) * frac)
    return X[:size], y[:size]

# %% ../../nbs/01_data.module.ipynb 46
DEFAULT_DATA_CONFIGS = {
    'adult': {
        'data' :'assets/data/s_adult.csv',
        'conf' :'assets/configs/data_configs/adult.json',
    },
    'heloc': {
        'data': 'assets/data/s_home.csv',
        'conf': 'assets/configs/data_configs/home.json'
    },
    'oulad': {
        'data': 'assets/data/s_student.csv',
        'conf': 'assets/configs/data_configs/student.json'
    }
}

# %% ../../nbs/01_data.module.ipynb 47
def _validate_dataname(data_name: str):
    if data_name not in DEFAULT_DATA_CONFIGS.keys():
        raise ValueError(f'`data_name` must be one of {DEFAULT_DATA_CONFIGS.keys()}, '
            f'but got data_name={data_name}.')

# %% ../../nbs/01_data.module.ipynb 48
def load_data(
    data_name: str, # The name of data
    return_config: bool = False, # Return `data_config `or not
    data_configs: dict = None # Data configs to override default configuration
) -> TabularDataModule | Tuple[TabularDataModule, TabularDataModuleConfigs]: 
    _validate_dataname(data_name)

    # get data/config urls
    _data_path = DEFAULT_DATA_CONFIGS[data_name]['data']
    _conf_path = DEFAULT_DATA_CONFIGS[data_name]['conf']
    
    data_url = f"https://github.com/BirkhoffG/cfnet/raw/master/{_data_path}"
    conf_url = f"https://github.com/BirkhoffG/cfnet/raw/master/{_conf_path}"

    # create new dir
    data_dir = Path(os.getcwd()) / "cf_data"
    if not data_dir.exists():
        os.makedirs(data_dir)
    data_path = data_dir / f'{data_name}.csv'
    conf_path = data_dir / f'{data_name}.json'

    # download data/configs
    if not data_path.is_file():
        urlretrieve(data_url, data_path)    
    if not conf_path.is_file():
        urlretrieve(conf_url, conf_path)

    # read config
    config = load_json(conf_path)['data_configs']
    config['data_dir'] = str(data_path)

    if not (data_configs is None):
        config.update(data_configs)

    config = TabularDataModuleConfigs(**config)
    data_module = TabularDataModule(config)

    if return_config:
        return data_module, config
    else:
        return data_module

