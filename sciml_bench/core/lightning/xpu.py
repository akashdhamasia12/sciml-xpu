from typing import Union, List, Dict, Any, Optional
from datetime import timedelta

from sciml_bench.core.lightning.utils import _has_ipex, _has_pl

_has_xpu = False

if _has_ipex:
    import intel_extension_for_pytorch as ipex
    if hasattr(ipex, "xpu"):
        _has_xpu = True

if _has_xpu and _has_pl:
    from lightning.pytorch.accelerators import Accelerator, AcceleratorRegistry
    from lightning.pytorch.strategies import SingleDeviceStrategy, DDPStrategy
    from lightning.pytorch.plugins import NativeMixedPrecisionPlugin

    import torch
    from torch import distributed as dist

    from sciml_bench.core.lightning.strategies import IntelMPIEnvironment

    class XPUAccelerator(Accelerator):

        """
        Implements a class for handling Intel XPU offloading, particularly the Data Center
        GPU Max Series (previously codename Ponte Vecchio).
        """

        @staticmethod
        def parse_devices(devices: Union[int, List[int]]) -> List[int]:
            """
            Parse the `trainer` input for devices and homogenize them.
            Parameters
            ----------
            devices : Union[int, List[int]]
                Single or list of device numbers to use
            Returns
            -------
            List[int]
                List of device numbers to use
            """
            if isinstance(devices, int):
                # in the case that a single number of devices is given,
                # generate a list of indices
                devices = list(range(devices))
            return devices

        def setup_device(self, device: torch.device) -> None:
            """
            Configure the current process to use a specified device.
            Perhaps unreliably and misguiding, the IPEX implementation of this method
            tries to mirror the CUDA version but `ipex.xpu.set_device` actually refuses
            to accept anything other than an index. I've tried to work around this
            by grabbing the index from the device if possible, and just setting
            it to the first device if not using a distributed/multitile setup.
            """
            # first try and see if we can grab the index from the device
            index = getattr(device, "index", None)
            if index is None and not dist.is_initialized():
                index = 0
            torch.xpu.set_device(index)

        def teardown(self) -> None:
            # as it suggests, this is run on cleanup
            torch.xpu.empty_cache()

        def get_device_stats(self, device) -> Dict[str, Any]:
            return torch.xpu.memory_stats(device)

        @staticmethod
        def get_parallel_devices(devices: List[int]) -> List[torch.device]:
            """
            Return a list of torch devices corresponding to what is available.
            Essentially maps indices to `torch.device` objects.
            Parameters
            ----------
            devices : List[int]
                List of integers corresponding to device numbers
            Returns
            -------
            List[torch.device]
                List of `torch.device` objects for each device
            """
            return [torch.device("xpu", i) for i in devices]

        @staticmethod
        def auto_device_count() -> int:
            return torch.xpu.device_count()

        @staticmethod
        def is_available() -> bool:
            """
            Determines if an XPU is actually available.

            Returns
            -------
            bool
                True if devices are detected, otherwise False
            """
            try:
                return torch.xpu.device_count() != 0
            except (AttributeError, NameError):
                return False

        @classmethod
        def register_accelerators(cls, accelerator_registry) -> None:
            accelerator_registry.register(
                "xpu",
                cls,
                description="Intel XPU interface for PyTorch Lightning.",
            )

    # add PVC to the registry
    AcceleratorRegistry.register("xpu", XPUAccelerator)

    class SingleXPUStrategy(SingleDeviceStrategy):

        """
        This class implements the strategy for using a single PVC tile.
        """

        strategy_name = "xpu_1T"

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

    class XPUDDPStrategy(DDPStrategy):
        strategy_name = "pvc_multi"

        def __init__(
            self,
            parallel_devices: Optional[List[torch.device]] = None,
            checkpoint_io: Optional["CheckpointIO"] = None,
            precision_plugin: Optional["PrecisionPlugin"] = None,
            ddp_comm_state: Optional[object] = None,
            ddp_comm_hook: Optional[callable] = None,
            ddp_comm_wrapper: Optional[callable] = None,
            model_averaging_period: Optional[int] = None,
            process_group_backend: Optional[str] = "ccl",
            timeout: Optional["timedelta"] = timedelta(seconds=180),
            main_address: Optional[str] = None,
            main_port: Optional[int] = 8679,
            **kwargs: Union[Any, Dict[str, Any]],
        ) -> None:
            if process_group_backend not in ["mpi", "ccl"]:
                raise ValueError(
                    f"{process_group_backend} is not supported for PVC: only 'mpi' and 'ccl'."
                )

            super().__init__(
                XPUAccelerator(),
                parallel_devices,
                IntelMPIEnvironment(main_address, main_port),
                checkpoint_io,
                precision_plugin,
                ddp_comm_state,
                ddp_comm_hook,
                ddp_comm_wrapper,
                model_averaging_period,
                process_group_backend,
                timeout,
                **kwargs,
            )

        def determine_ddp_device_ids(self) -> None:
            return None

    class XPUBF16Plugin(NativeMixedPrecisionPlugin):
        def __init__(self):
            super().__init__("bf16", "xpu")

        def auto_cast_context_manager(self):
            """
            Overrides the default behavior, which relies on `torch.amp` where only
            CPU and CUDA backends are supported. This uses the `xpu.amp` interface
            explicitly, as done in the IPEX documentation.
            """
            return torch.xpu.amp.autocast(
                self.device, enabled=True, dtype=torch.bfloat16
            )
