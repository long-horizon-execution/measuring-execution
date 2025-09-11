from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    provider: str = "vllm"
    name: str = "google/gemma-3-1b-it"
    thinking_mode: bool = False
    tensor_parallel_size: Optional[int] = None
    sliding_window_size: Optional[int] = None # Number of turns to keep in memory for multi-turn models
    fill_history: bool = False # Whether to fill history 
    incorrect_probability: float = 0.0 # when filling history, the probability of earlier turns being incorrect
    target_step: Optional[int] = None 
    majority_vote: int = 1 # Number of votes to consider for majority voting in multi-turn models, 0 means no majority voting
    max_model_len: int = 32768
    multi_turn: bool = True
    cot: bool = False
    num_tasks_per_turn: int = 1
    num_workers: int = 10
    gpu_memory_utilization: float = 0.90
    
    early_stopping: bool = False
    early_stopping_threshold: float = 0.0 # Threshold for early stopping based on prefix correctness