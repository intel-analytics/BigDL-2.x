from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from zoo.automl.common import metrics
from zoo.zouwu.autots.model.auto_lstm import AutoLSTM
from zoo.orca.automl import hp
import argparse
from zoo.orca import init_orca_context, stop_orca_context


input_feature_dim = 10
output_feature_dim = 2
past_seq_len = 5
future_seq_len = 1


def get_x_y(size):
    x = np.random.randn(size, past_seq_len, input_feature_dim)
    y = np.random.randn(size, future_seq_len, output_feature_dim)
    return x, y


class RandomDataset(Dataset):

    def __init__(self, size=1000):
        x, y = get_x_y(size)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_dataloader_creator(config):
    return DataLoader(RandomDataset(size=1000),
                      batch_size=config["batch_size"],
                      shuffle=True)


def valid_dataloader_creator(config):
    return DataLoader(RandomDataset(size=400),
                      batch_size=config["batch_size"],
                      shuffle=True)


def test_fit_np(args):
    auto_lstm = AutoLSTM(input_feature_num=input_feature_dim,
                         output_target_num=output_feature_dim,
                         optimizer='Adam',
                         loss=torch.nn.MSELoss(),
                         metric="mse",
                         hidden_dim=hp.grid_search([32, 64]),
                         layer_num=hp.randint(1, 3),
                         lr=hp.choice([0.001, 0.003, 0.01]),
                         dropout=hp.uniform(0.1, 0.2),
                         logs_dir="/tmp/auto_lstm",
                         cpus_per_trial=args.cpus_per_trial,
                         name="auto_lstm")
    auto_lstm.fit(data=get_x_y(size=1000),
                  epochs=args.epoch,
                  batch_size=hp.choice([32, 64]),
                  validation_data=get_x_y(size=400),
                  n_sampling=args.n_sampling
                  )
    best_model = auto_lstm.get_best_model()
    best_model_mse = best_model.evaluate(
        x=get_x_y(400)[0], y=get_x_y(400)[1], metrics=['mse'])
    print(f"best model mse is {np.mean(best_model_mse[0][0]):.2f}")


def test_fit_data_creator(args):
    auto_lstm = AutoLSTM(input_feature_num=input_feature_dim,
                         output_target_num=output_feature_dim,
                         optimizer='Adam',
                         loss=torch.nn.MSELoss(),
                         metric="mse",
                         hidden_dim=hp.grid_search([32, 64]),
                         layer_num=hp.randint(1, 3),
                         lr=hp.choice([0.001, 0.003, 0.01]),
                         dropout=hp.uniform(0.1, 0.2),
                         logs_dir="/tmp/auto_lstm",
                         cpus_per_trial=args.cpus_per_trial,
                         name="auto_lstm")
    auto_lstm.fit(data=train_dataloader_creator,
                  epochs=args.epoch,
                  batch_size=hp.choice([32, 64]),
                  validation_data=valid_dataloader_creator,
                  n_sampling=args.n_sampling,
                  )
    best_model = auto_lstm.get_best_model()
    val_x, val_y = next(iter(valid_dataloader_creator(best_model.config)))
    best_model_mse = best_model.evaluate(x=val_x.numpy(), y=val_y.numpy(), metrics=['mse'])
    print(f'best modle mse is {np.mean(best_model_mse[0][0]):.2f}')
    print(f"best_model config is {best_model.config}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=1,
                        help="The number of nodes to be used in the cluster. "
                            "You can change it depending on your own cluster setting.")
    parser.add_argument('--cluster_mode', type=str,
                        default='local', help="The mode for the Spark cluster.")
    parser.add_argument('--cores', type=int, default=4,
                        help="The number of cpu cores you want to use on each node."
                            "You can change it depending on your own cluster setting.")
    parser.add_argument('--memory', type=str, default="10g", 
                        help="The memory you want to use on each node."
                            "You can change it depending on your own cluster setting.")

    parser.add_argument('--epoch', type=int, default=1, 
                        help="Max number of epochs to train in each trial.")
    parser.add_argument('--cpus_per_trial', type=int,
                        default=2, help="Int. Number of cpus for each trial")
    parser.add_argument('--n_sampling', type=int, default=1, 
                        help=" Number of times to sample from the search_space.")
    parser.add_argument("--workers_per_node",
                        type=int,
                        default=2,
                        help="The number of workers to run on each node")
    parser.add_argument('--k8s_master',
                        type=str,
                        default="",
                        help="The k8s master. "
                        "It should be k8s://https://<k8s-apiserver-host>: "
                        "<k8s-apiserver-port>.")
    parser.add_argument("--container_image",
                        type=str,
                        default="",
                        help="The runtime k8s image. "
                        "You can change it with your k8s image.")
    parser.add_argument('--k8s_driver_host',
                        type=str,
                        default="",
                        help="The k8s driver localhost. ")
    parser.add_argument('--k8s_driver_port',
                        type=str,
                        default="",
                        help="The k8s driver port.")
    args = parser.parse_args()
    if args.cluster_mode == "local":
        init_orca_context(cluster_mode="local",
                          cores=args.cores,
                          num_nodes=args.num_nodes,
                          memory=args.memory,
                          init_ray_on_spark=True)
    elif args.cluster_mode == "yarn":
        init_orca_context(cluster_mode="yarn-client",
                          cores=args.cores,
                          memory=args.memory,
                          init_ray_on_spark=True)
    elif args.cluster_mode == "k8s":
        if not args.k8s_master or not args.container_image \
                or not args.k8s_driver_host or not args.k8s_driver_port:
            parser.print_help()
            parser.error('k8s_master, container_image,'
                         'k8s_driver_host/port are required not to be empty')
        init_orca_context(cluster_mode="k8s",
                          master=args.k8s_master,
                          container_image=args.container_image,
                          cores=args.cores,
                          init_ray_on_spark=True,
                          conf={
                              "spark.driver.host": args.k8s_driver_host,
                              "spark.driver.port": args.k8s_driver_port
                          })
    test_fit_np(args)
    test_fit_data_creator(args)
    stop_orca_context()
