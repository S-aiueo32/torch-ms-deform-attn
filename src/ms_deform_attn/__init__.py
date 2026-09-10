"""CPU multi-scale deformable attention for PyTorch."""
from .functional import MSDeformAttnFunction, ms_deform_attn, ms_deform_attn_core_pytorch

__all__ = ["MSDeformAttnFunction", "ms_deform_attn", "ms_deform_attn_core_pytorch"]
__version__ = "0.1.0"
