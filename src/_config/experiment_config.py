from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union


@dataclass
class BaseExperimentConfig:
    type: str  # e.g., "PrefixSum", "Nand"
    num_samples: int = 10
    working_capacity: int = 1  # How many inputs the model processes at once per output.
    llm_max_tokens: int = 1000
    # llm stop sequences can be empty list or a list of string
    llm_stop_sequences: Union[List[str], List[List[str]]] = field(
        default_factory=lambda: []
    )
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    local_dataset_path: Optional[str] = None
    
    start_at: int = 0  # Start index for the dataset
    pass_at_k: int = 0 # Pass if the correct answer is within the top K predictions


@dataclass
class DictSumExecutionExperimentConfig(BaseExperimentConfig):
    type: str = "DictSum"
    dict_size: int = 10
    horizon_length: int = 10
    min_input_value: int = -9
    max_input_value: int = 9

# --- Top-level Experiments Dictionary ---
@dataclass
class Experiments:
    dict_sum: DictSumExecutionExperimentConfig = field(
        default_factory=DictSumExecutionExperimentConfig
    )