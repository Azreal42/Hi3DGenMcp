
from dataclasses import dataclass


@dataclass
class Config:
    local_cache_dir: str = "./weights"

    # trellis   
    trellis_model_name: str = "trellis_normal-v0-1"
    trellis_dino_model_name: str = "dinov2_vitl14_reg"
    trellis_use_fp16: bool = True
    trellis_attention_backtend: str = "xformers"
    trellis_sparse_backend: str = "spconv"
    trellis_spconv_algo: str = "implicit_gemm"

    # stable_x
    stable_x_model_name: str = "yoso-normal-v1-8-1"
    



