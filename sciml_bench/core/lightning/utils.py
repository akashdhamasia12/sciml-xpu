
_has_ipex = False
_has_pl = False
from warnings import warn

try:
    import intel_extension_for_pytorch as ipex
    _has_ipex = True
except ImportError:
    warn("IPEX not installed - XPU Lightning classes disabled.")

try:
    from lightning import pytorch as pl
    _has_pl = True
except ImportError:
    warn("No PyTorch Lightning installation - not loading Lightning classes.")


