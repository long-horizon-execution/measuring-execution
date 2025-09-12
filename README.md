# The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs

This project contains the code accompanying the paper [The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](https://arxiv.org/abs/2509.09677). 

Link to [Dataset Page](https://huggingface.co/datasets/arvindh75/Long-Horizon-Execution)

If you like our work, consider citing us!

```
@misc{
      sinha2025illusiondiminishingreturnsmeasuring,
      title={The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs}, 
      author={Akshit Sinha and Arvindh Arun and Shashwat Goel and Steffen Staab and Jonas Geiping},
      year={2025},
      eprint={2509.09677},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.09677}, 
}
```

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Experiments](#experiments)
- [Output and Logging](#output-and-logging)
- [Development](#development)

## Setup

### Prerequisites

- Python 3.10 or higher
- [uv package manager](https://github.com/astral-sh/uv)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd long-horizon-execution
```

2. Install dependencies (uv will handle this automatically):
```bash
uv sync
```

### Environment Variables

To use OpenRouter API, set your API key:
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

## Project Structure

```
├── main.py                    # Main entry point
├── experiment_runner.py       # Core experiment orchestration
├── llm_clients.py            # Unified LLM client implementations
├── generate_dataset_json.py   # Dataset generation utilities
├── utils.py                  # General utility functions
├── pyproject.toml            # Project dependencies and configuration
├── words_alpha.txt           # Word list for experiments
├── src/
│   ├── config.py             # Main configuration classes
│   ├── _config/              # Configuration modules
│   │   ├── experiment_config.py    # Experiment-specific configs
│   │   ├── model_config.py        # LLM model configurations
│   │   ├── openrouter_config.py   # OpenRouter API settings
│   │   └── wandb_config.py        # Weights & Biases integration
│   └── experiments/          # Experiment implementations
│       ├── base_experiment.py     # Base experiment class
│       └── dict_sum/              # Dictionary sum experiment
│           ├── exp.py             # Main experiment logic
│           └── dict_sum_util.py   # Utilities and evaluators
```

## Configuration

The project uses a hierarchical configuration system with the following main components:

### Model Configuration
- `provider`: LLM provider ("openrouter", "vllm", etc.)
- `name`: Model name/identifier
- `multi_turn`: Enable multi-turn conversation mode
- `thinking_mode`: Enable chain-of-thought reasoning
- `cot`: Chain-of-thought mode
- `max_model_len`: Maximum model context length

### Experiment Configuration
- `exp`: Experiment type (e.g., "dict_sum")
- `num_samples`: Number of test samples
- `dict_size`: Size of dictionaries (for dict_sum)
- `working_capacity`: Number of inputs processed per turn
- `horizon_length`: Length of the execution horizon (working_capacity * number of turns you want to run)
- `llm_temperature`: Sampling temperature
- `llm_top_p`: Top-p sampling parameter
- `llm_max_tokens`: Maximum tokens per response

### Weights & Biases Configuration
- `mode`: "online", "offline", or "disabled"
- `project`: W&B project name

## Running Experiments

### Basic Usage

```bash
uv run main.py --cfg.exp dict_sum
```

### Complete Example

Here's a comprehensive example with all configuration options:

```bash
uv run main.py --cfg.exp dict_sum \
    --cfg.model_config.provider "openrouter" \
    --cfg.model_config.name "$MODEL" \
    --cfg.model_config.thinking_mode $THINKING_MODE \
    --cfg.model_config.cot $COT \
    --cfg.model_config.max_model_len 40960 \
    --cfg.experiments.dict_sum.num_samples 1 \
    --cfg.experiments.dict_sum.dict_size 100 \
    --cfg.experiments.dict_sum.working_capacity ${WORKING_CAPACITY} \
    --cfg.experiments.dict_sum.horizon_length ${WORKING_CAPACITY} \
    --cfg.experiments.dict_sum.llm_temperature 0.6 \
    --cfg.experiments.dict_sum.llm_top_p 0.95 \
    --cfg.experiments.dict_sum.llm_max_tokens 100000 \
    --cfg.experiments.dict_sum.max_input_value 99 \
    --cfg.experiments.dict_sum.min_input_value -99 \
    --cfg.wandb_settings.mode "online" \
    --cfg.wandb_settings.project "frontier-final" \
    --cfg.experiments.dict_sum.local_dataset_path "dict_sum_100.json"
```

### Thinking Modes
- `thinking_mode=true`: Enables advanced reasoning mode for supported models (e.g., Claude 4)
- `cot=true`: Enables chain-of-thought reasoning for step-by-step problem solving, supported for models that are not explicitly in thinking mode (e.g., Deepseek V3)

**Only one of these can be enabled at a time.**

Disabling both will run the model in standard mode, where it attempts to execute each turn in one go without intermediate reasoning steps.

### Environment Variables

You can use environment variables for dynamic configuration:

```bash
export MULTI_TURN=true
export MODEL="anthropic/claude-3.5-sonnet"
export THINKING_MODE=true
export COT=true
export WORKING_CAPACITY=10

uv run main.py --cfg.exp dict_sum \
    --cfg.model_config.multi_turn $MULTI_TURN \
    --cfg.model_config.name "$MODEL"
    # ... other parameters
```

## Experiments

### Dictionary Sum Experiment (`dict_sum`)

The main experiment implemented evaluates a model's ability to perform arithmetic operations over dictionaries across multiple turns:

- **Task**: Given a dictionary with key-value pairs, perform cumulative sum operations
- **Evaluation**: Accuracy of final computed values
- **Parameters**:
  - `dict_size`: Number of key-value pairs in each dictionary
  - `horizon_length`: Number of operations to perform
  - `working_capacity`: Number of operations processed per turn
  - `min_input_value`/`max_input_value`: Range of values in dictionaries

### Experiment Modes

2. **Multi-turn mode** (`multi_turn=true`): Operations split across multiple conversation turns **REQUIRED**
3. **Chain-of-thought** (`cot=true`): Enables step-by-step reasoning
4. **Thinking mode** (`thinking_mode=true`): Advanced reasoning mode for supported models

## Output and Logging

### Output Directory Structure

Results are saved to timestamped directories:
```
output/
└── dict_sum_{model_name}_{timestamp}_{mode_flags}/
    ├── results.json          # Experiment results
    ├── config.yaml          # Full configuration used
    └── logs/                # Detailed logs
```

### Logging

The application provides detailed logging including:
- Configuration validation
- Experiment progress
- Model response times
- Error handling
- W&B integration status

### Weights & Biases Integration

Results are automatically logged to W&B when configured:
- Experiment metrics and results
- Configuration parameters
- Model performance statistics

## Development

### Adding New Experiments

1. Create a new experiment directory under `src/experiments/`
2. Implement the experiment class inheriting from `BaseExperiment`
3. Add configuration class in `src/_config/experiment_config.py`
4. Register the experiment in the appropriate modules

### Dataset Generation

Use `generate_dataset_json.py` to create custom datasets:

```bash
uv run generate_dataset_json.py
```

### Testing

Run experiments with minimal samples for testing:

```bash
uv run main.py --cfg.exp dict_sum \
    --cfg.experiments.dict_sum.num_samples 1 \
    --cfg.wandb_settings.mode "disabled"
```

## Dependencies

Key dependencies include:
- `jsonargparse`: Configuration management
- `openai`: OpenRouter and OpenAI API integration
- `vllm`: Local model inference
- `wandb`: Experiment tracking
- `torch`: Deep learning framework
- `transformers`: Hugging Face model support

For a complete list, see `pyproject.toml`.
