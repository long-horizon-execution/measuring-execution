# config_definitions.py
from dataclasses import dataclass, field
from ._config import *


# --- Run Settings ---
@dataclass
class RunSettings:
    output_dir: str = "output"
    eval_seed: int = 42
    log_level: str = "INFO"
    model_name: str = ""


# --- Main Configuration Class ---
@dataclass
class MainConfig:
    run_settings: RunSettings = field(default_factory=RunSettings)
    exp: str = "dict_sum" # Default experiment key
    wandb_settings: WandBSettings = field(default_factory=WandBSettings)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    openrouter_config: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    experiments: Experiments = field(
        default_factory=Experiments
    )
