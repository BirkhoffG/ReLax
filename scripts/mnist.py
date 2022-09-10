from cfnet.nets import CounterNetConv, PredictivConvNet
from cfnet.datasets import MNISTDataModule, MNISTDataConfigs
from cfnet.train import train_model
from cfnet.training_module import PredictiveTrainingModule, CounterNetTrainingModuleConv
from cfnet.utils import make_model

if __name__ == "__main__":
    d_configs = MNISTDataConfigs(batch_size=128)
    dm = MNISTDataModule(d_configs)
    model = PredictiveTrainingModule(
        net=make_model(None, PredictivConvNet),
        m_configs={"lr": 0.003}
    )

    t_configs = {
        'n_epochs': 10,
        'monitor_metrics': 'val/val_loss'
    }

    train_model(
        model, dm, t_configs
    )

    cfnet_configs = {
        "lr": 0.003,
        "lambda_1": 1.0,
        "lambda_3": 0.1,
        "lambda_2": 0.2,
    }
    cfnet_model = CounterNetTrainingModuleConv(
        m_configs=cfnet_configs
    )
    train_model(
        cfnet_model, dm, t_configs
    )
