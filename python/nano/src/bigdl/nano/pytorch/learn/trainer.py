from logging import warn
import torch
import pytorch_lightning as pl
import intel_pytorch_extension as ipex
from bigdl.nano.pytorch.accelerator.ipex_accelerator import IPEXAccelerator
from bigdl.nano.pytorch.ddp_spawn import DDPSpawnPlugin
from pytorch_lightning.plugins.environments import LightningEnvironment
from typing import Any, List, Optional

distributed_backends = ["spawn", "ray"]


class Trainer(pl.Trainer):

    def __init__(self, num_processes: int = 1,
                 use_ipex: bool = True,
                 enable_bf16=False,
                 distributed_backend="spawn",
                 cpu_for_each_process: Optional[List[List[int]]] = None,
                 *args: Any, **kwargs: Any) -> None:
        """
        A pytorch lightning trainer that uses bigdl.nano lite optimization.

        :param num_processes: number of processes in distributed training. default: 4.
        :param use_ipex: whether we use ipex as accelerator for trainer. default: True.
        :param cpu_for_each_process: A list of length `num_processes`, each containing a list of
            indices of cpus each process will be using. default: None, and the cpu will be
            automatically and evenly distributed among processes.
        """

        # Check keyword arguments
        if "accelerator" in kwargs:
            warn(f"""Accelerator will be specified by bigdl.nano,
            accelerator entered {kwargs['accelerator']} will be ignored. """)

            kwargs.pop('accelerator')
        if "plugins" in kwargs:
            warn(f"""Plugins will be specified by bigdl.nano,
             plugines entered {kwargs['plugins']} will be ignored. """)

            kwargs.pop('plugins')
        if cpu_for_each_process is not None:
            if len(cpu_for_each_process) != num_processes:
                raise ValueError(f"The length of `cpu_for_each_process` ("
                                 f"{len(cpu_for_each_process)}) is not equal to the number of"
                                 f" processes {num_processes}.")

        # Initialize trainer

        if num_processes == 1:
            accelerator = IPEXAccelerator(enable_bf16=enable_bf16) if use_ipex else None
            print(accelerator)

            super().__init__(accelerator=accelerator, *args, **kwargs)
        else:
            plugin = None
            assert distributed_backend in distributed_backends, \
                f"Distributed backends supported now are spawn and ray," \
                " but get {distributed_backend}."
            if distributed_backend == "spawn":
                device = ipex.DEVICE if use_ipex else "cpu"
                plugin = DDPSpawnPlugin(parallel_devices=[
                    torch.device(device) for _ in range(num_processes)],
                    cpu_for_each_process=cpu_for_each_process,
                    cluster_environment=LightningEnvironment())
            elif distributed_backend == "ray":
                # Import RayPlugins may entangle with openmp even if it has not been used,
                # which leads to an unacceptably low performance.
                # So we import when we need.
                from bigdl.nano.pytorch.ray_distributed import RayPlugin
                plugin = RayPlugin(num_workers=num_processes, use_ipex=use_ipex)  # type: ignore

            accelerator = IPEXAccelerator(training_type_plugin=plugin,  # type: ignore
                                          enable_bf16=enable_bf16) if use_ipex else None

            super().__init__(accelerator=accelerator, plugins=[plugin], *args, **kwargs)
