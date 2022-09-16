from copy import deepcopy
from cfnet.train import train_model, TensorboardLogger
from cfnet.training_module import PredictiveTrainingModule
from cfnet.datasets import TabularDataModule
from cfnet.methods import VanillaCF, DiverseCF, ProtoCF
from cfnet.evaluate import evaluate_cfs, generate_cf_results_local_exp, benchmark_cfs, metrics2fn
from cfnet.import_essentials import *
from .utils_configs import get_configs
import argparse


def load_configs(config_path):
    with open(config_path) as f:
        return json.load(f)


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
    'lr': 0.001,
    'lambda_': 0.005
}
cf_exp_cls = {
    'vanilla': VanillaCF,
    'diverse': DiverseCF,
    'proto': ProtoCF
}

cf_results_df_list = []


def store_benchmark():
    benchmark_df = pd.concat(cf_results_df_list)
    benchmark_df.to_csv('result_vanilla.csv')
    print(benchmark_df)


def get_cf_exp_cls(method_name: str):
    if method_name == 'all':
        return [v for k, v in cf_exp_cls.items()]
    
    if method_name not in cf_exp_cls.keys():
        raise ValueError(f"`data_name` should be one of `{cf_exp_cls.keys()}`, but got `{method_name}`")
    
    return [ cf_exp_cls[method_name] ]


def main(args):
    
    configs_list = get_configs(args.data_name)
    exp_cls_list = get_cf_exp_cls(args.method)

    for configs in configs_list:
        dm = TabularDataModule(configs["data_configs"]) 
        model = PredictiveTrainingModule(m_configs)
        
        params, opts_state = train_model(model, dm, t_configs)
        _params = deepcopy(params)
        pred_fn = lambda x: model.forward(_params, random.PRNGKey(0), x, is_training=False)
        
        for exp_cls in exp_cls_list:
            cf_exp = exp_cls(cf_configs)
            cf_results = generate_cf_results_local_exp(cf_exp, dm, pred_fn)
            cf_results_df = evaluate_cfs(
                cf_results, metrics=metrics2fn.keys(), 
                return_dict=False, return_df=True)
            cf_results_df_list.append(cf_results_df)

    store_benchmark()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', 
                        type=str, 
                        default='all')
    parser.add_argument('--method',
                        type=str,
                        default='all')
    args = parser.parse_args()
    main(args)
    
    # try:
    #     main(args)
    # except Exception as e:
    #     print("something is wrong: ", e)
    #     store_benchmark()


