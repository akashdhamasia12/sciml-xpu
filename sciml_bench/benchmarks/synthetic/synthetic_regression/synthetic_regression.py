# For running the code, these packages are required:
# pip install torch
# pip install prettytable
#
# To run the code issue this command:
# python synthetic_v1.py
#
from warnings import warn
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch
except ImportError:
    warn("No IPEX or CCL bindings installed!")
try:
    from sciml_bench.core.lightning import xpu
except ImportError:
    warn("Unable to load sciml_bench.core.lightning.xpu module")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import yaml
import lightning.pytorch as pl
import torch.distributed as dist
from torch.utils.data import DataLoader
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.plugins.precision import NativeMixedPrecisionPlugin
from lightning.pytorch.strategies import SingleDeviceStrategy
from sciml_bench.core.lightning.strategies import IntelMPIEnvironment, CustomDDPStrategy
from sciml_bench.core.lightning.ipex import IPEXCallback

from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.core.utils import homogenize_mpi_env

class RegressionDataset(torch.utils.data.Dataset):

    def __init__(self, num_samples, input_size):
        super().__init__()
        X = torch.randn(num_samples, input_size)
        y = X ** 2 + 15 * np.sin(X) **3
        y_t = torch.sum(y, dim=1)
        self._x = X
        self._y = y_t

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index]
        y = self._y[index]
        return x, y

class SyntheticRegression(pl.LightningModule):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.net = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size),  # input -> hidden layer
            nn.Sigmoid(),                        
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size), # hidden -> output layer
            nn.Sigmoid()	
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x).flatten()
        loss = F.mse_loss(y_hat, y) 
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer


def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    default_args = {
        'input_size': 784,
        'batch_size': 128,
        'num_samples': 1024000,
        'hidden_size': 3000,
        'epochs': 1,
        'comm_backend': 'ccl',
        'accelerator': 'cpu',
        'precision': '32',
        'ipex': True,
        'dist_url': "127.0.0.1",
        'dist_port': "29500"
    }

    # Log top level process
    log = params_out.log.console
    log.begin(f'Running benchmark synthetic_regression in training mode')

    # Parse input arguments against default ones 
    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    if args["comm_backend"] in ["mpi", "ccl"]:
        rank = int(os.getenv("PMI_RANK", 0))
        local_rank = int(os.getenv("MPI_LOCALRANKID", 0))
        world_size = int(os.getenv("PMI_SIZE", 1))
    else:
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
    params_out.activate(rank=rank, local_rank=local_rank)
    num_samples = args['num_samples']
    batch_size = args['batch_size']

    with log.subproc("Creating dataset"):
        dataset = RegressionDataset(num_samples, args['input_size'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    log.message(f"Number of samples: {num_samples}")
    log.message(f"Total number of batches: {len(dataloader)}, {num_samples/batch_size}")

    model = SyntheticRegression(
        args['input_size'],
        args['hidden_size'],
        output_size=1
    )
    if args["accelerator"] in ["cpu", "xpu"] and args["ipex"]:
        cb = [IPEXCallback()]
    else:
        cb = []

    if args["accelerator"] == "cpu" and args["precision"] != "bf16":
        precision_plugin = None
    elif args["accelerator"] == "xpu" and args["precision"] == "bf16":
        precision_plugin = xpu.XPUBF16Plugin()
    else:
       precision_plugin = NativeMixedPrecisionPlugin(args["precision"], args["accelerator"])

    if args["comm_backend"] in ["mpi", "ccl"]:
        # this benchmark is intend for single node usage
        env = IntelMPIEnvironment(main_address=args["dist_url"], main_port=args["dist_port"])
    else:
        env = LightningEnvironment()

    if world_size > 1:
        strategy = CustomDDPStrategy(
            accelerator=args["accelerator"],
            cluster_environment=env,
            process_group_backend=args["comm_backend"],
            precision_plugin=precision_plugin,
            static_graph=True,
            bucket_cap_mb=200
            )
    else:
        strategy = SingleDeviceStrategy(
            device=torch.device(args["accelerator"], 0),
            precision_plugin=precision_plugin,
            accelerator=args["accelerator"]
        )

    with log.subproc('Training model'):
        trainer = pl.Trainer(
            devices=world_size, strategy=strategy, 
            max_epochs=args['epochs'], default_root_dir=params_in.output_dir,
            callbacks=cb
        )
        start_time = time.time()
        trainer.fit(model, dataloader)
        end_time = time.time()

    time_taken = end_time - start_time

    metrics = {}
    metrics['time'] = time_taken
    metrics_file = params_in.output_dir / 'metrics.yml'
    with log.subproc('Saving training metrics to a file'):
        with open(metrics_file, 'w') as handle:
            yaml.dump(metrics, handle)  