
from typing import Any
from logging import getLogger

import torch
from torch import distributed as dist
from sciml_bench.core.lightning.utils import _has_pl

if _has_pl:
    from lightning import pytorch as pl
    from lightning.pytorch.strategies import (
        DDPStrategy,
        StrategyRegistry,
        SingleDeviceStrategy,
    )
    from lightning.pytorch.plugins.environments import LightningEnvironment

    import os
    from typing import Union, Optional

    log = getLogger("lightning.pytorch.plugins.environments")

    class IntelMPIEnvironment(LightningEnvironment):
        def __init__(
            self, main_address: Optional[str] = None, main_port: int = 8679
        ) -> None:
            super().__init__()
            self.main_address = main_address
            self.main_port = main_port

        @staticmethod
        def _get_env_var(key: str, default_value: Union[int, str]) -> str:
            expected_type = type(default_value)
            if key not in os.environ:
                log.warn(f"{key} not found in environment variables, using default value of {default_value}")
                value = default_value
                # set the environment variable for this run
                os.environ[key] = str(value)
            else:
                value = expected_type(os.getenv(key))
            return value

        def world_size(self) -> int:
            world_size = self._get_env_var("PMI_SIZE", 1)
            return world_size

        def local_rank(self) -> int:
            local_rank = self._get_env_var("MPI_LOCALRANKID", 0)
            return local_rank

        def global_rank(self) -> int:
            rank = self._get_env_var("PMI_RANK", 0)
            return rank

        @property
        def main_address(self) -> str:
            return self._main_address

        @main_address.setter
        def main_address(self, value: Union[None, str]) -> None:
            # first grab it from environment variables
            if value is None:
                value = self._get_env_var("HYDRA_BSTRAP_LOCALHOST", "127.0.0.1")
            self._main_address = value

        @property
        def main_port(self) -> int:
            return self._main_port

        @main_port.setter
        def main_port(self, value: int) -> None:
            self._main_port = value

        @property
        def creates_processes_externally(self) -> bool:
            """
            Override this because we rely on `mpiexec` or `mpirun` for
            the process spawning.
            """
            return True


    class SingleXPUStrategy(SingleDeviceStrategy):

        """
        This class implements the strategy for using a single PVC tile.
        """

        strategy_name = "pvc_single"

        def __init__(
            self,
            device: Optional[str] = "xpu",
            checkpoint_io=None,
            precision_plugin=None,
        ):
            super().__init__(
                device=device,
                accelerator=XPUAccelerator(),
                checkpoint_io=checkpoint_io,
                precision_plugin=precision_plugin,
            )

        @property
        def is_distributed(self) -> bool:
            return False

        def setup(self, trainer) -> None:
            self.model_to_device()
            super().setup(trainer)

        def setup_optimizers(self, trainer) -> None:
            super().setup_optimizers(trainer)

        def model_to_device(self) -> None:
            self.model.to(self.root_device)

        @classmethod
        def register_strategies(cls, strategy_registry) -> None:
            strategy_registry.register(
                cls.strategy_name, cls, description=f"{cls.__class__.__name__}"
            )
    
    class CustomDDPStrategy(DDPStrategy):
        def setup_environment(self) -> None:
            super().setup_environment()
            # call barrier as the first collective to synchronize workers
            self.barrier()
        
        def barrier(self, *args: Any, **kwargs: Any) -> None:
            """
            Implements a stand-in replacement for barrier, by calling
            `dist.all_reduce` on a dummy tensor that exists on the
            target device.
            """
            dummy_tensor = torch.zeros((1,), device=self.root_device)
            dist.all_reduce(dummy_tensor)

        def teardown(self) -> None:
            """
            Invokves barrier before running the base `teardown` method.
            The idea is to let workers synchronize first before ending
            execution prematurely, which sometimes will prevent metrics
            from being saved for SciML.
            """
            self.barrier()
            return super().teardown()
