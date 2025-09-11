# experiments/base_experiment.py
from abc import ABC, abstractmethod
import numpy as np

from src.config import RunSettings  # Keep for type hinting if experiments use numpy


class BaseExperiment(ABC):
    """
    Abstract base class for defining an experiment.
    All specific experiment classes should inherit from this and implement its methods.
    """

    def __init__(self, common_config: RunSettings):
        """
        Initializes the experiment.

        Args:
            common_config (dict): General configuration applicable to all experiments
                                  (e.g., output_dir, eval_seed from 'run_settings').
            experiment_specific_config (dict): Configuration specific to this experiment
                                               (e.g., num_automata for CA).
        """
        self.common_config = common_config
        self.eval_seed = common_config.eval_seed
        self.output_dir_base = common_config.output_dir
        self.results = (
            {}
        )  # To store various results throughout the experiment lifecycle
        self.ground_truth_data = []
        self.prompts = []
        self.llm_raw_outputs = []
        self.processed_llm_outputs = []
        self.parsed_llm_outputs = []
        self.evaluation_metrics = {}
        self.model_name = common_config.model_name
        
        # Initialize naming attributes (will be set by ExperimentRunner)
        self.model_name_for_path = None
        self.run_timestamp = None
        self.is_multi_turn = False  # Default to single-turn unless specified otherwise
        self.is_cot = common_config.cot
        if self.eval_seed is not None:
            np.random.seed(self.eval_seed)
            # import random # If using Python's random
            # random.seed(self.eval_seed)
        self.START_TEMPLATE = (
            None  # Placeholder for the start template, to be set by subclasses
        )
        self.FOLLOW_UP_TEMPLATE = (
            None  # Placeholder for the follow-up template, to be set by subclasses
        )

    @abstractmethod
    def setup(self):
        """
        Handles all experiment-specific setup.
        This can include:
        - Loading or generating data.
        - Defining ground truth.
        - Initializing any experiment-specific objects.
        Should populate `self.ground_truth_data` and any other necessary attributes.
        """
        pass

    @abstractmethod
    def prepare_prompts(self, llm_client_info=None):
        """
        Generates the list of prompts to be sent to the LLM.
        Prompts should be stored in `self.prompts`.

        Args:
            llm_client_info (dict, optional): Information about the LLM client,
                                             if needed for prompt generation.
        Returns:
            list: A list of prompts (strings).
        """
        pass

    @abstractmethod
    def prepare_multi_turn_prompts(self, num_tasks_per_turn=1):
        """
        Converts the prompts in `self.prompts` to a multi-turn format if necessary.
        This is useful for LLMs that require a specific format for multi-turn conversations.
        The modified prompts should still be stored in `self.prompts`.

        Returns:
            multi_turn_prompts (list): A list of prompts formatted for multi-turn interaction. will only contain problem instances
        """
        pass

    @abstractmethod
    def process_llm_outputs(self, llm_outputs_raw, enable_thinking=False):
        """
        Parses the raw text output from the LLM into a structured format.
        The raw outputs are passed as `llm_outputs_raw`.
        Processed outputs should be stored in `self.processed_llm_outputs`.

        Args:
            llm_outputs_raw (list): A list of raw LLM output objects/strings,
                                   corresponding to the prompts.
                                   (e.g. for vLLM, list of RequestOutput)
            is_multi_turn (bool): Whether the outputs are from a multi-turn conversation. If True, the outputs may need special handling.

        Returns:
            list: A list of processed outputs.
        """
        pass

    @abstractmethod
    def evaluate_predictions(self):
        """
        Compares the `self.processed_llm_outputs` against `self.ground_truth_data`
        and computes relevant metrics.
        Evaluation metrics should be stored in `self.evaluation_metrics`.

        Returns:
            dict: A dictionary containing aggregated and per-instance evaluation results.
                  Example: {"aggregated_accuracy": 0.85, "per_instance_scores": [...]}
        """
        pass

    @abstractmethod
    def get_experiment_params_for_logging(self):
        """
        Returns a dictionary of experiment-specific parameters to be logged.
        This will be merged with common parameters for the final results file.

        Returns:
            dict: Experiment-specific parameters.
        """
        pass

    def get_results_to_save(self):
        """
        Returns a dictionary of all data that should be saved to the results file for this experiment.
        This method can be overridden if more specific control over saved data is needed.
        By default, it saves prompts, raw outputs, processed outputs, ground truth, and metrics.

        Returns:
            dict: Data to be saved.
        """
        return {
            "prompts": self.prompts,
            "llm_raw_outputs": [
                str(o) for o in self.llm_raw_outputs
            ],  # Ensure serializable
            "parsed_llm_outputs": self.parsed_llm_outputs,
            "ground_truth_data": [
                str(gt) for gt in self.ground_truth_data
            ],  # Ensure serializable
            "evaluation_metrics": self.evaluation_metrics,
        }

    def run_plotting(self, run_output_dir, base_filename):
        """
        Optional: Generate and save any plots specific to this experiment.
        This method can be overridden by concrete experiment classes.

        Args:
            run_output_dir (str): The directory where results for this specific run are saved.
            base_filename (str): A base filename prefix for plot files.
        """
        # logging.info(f"No specific plotting implemented for {self.__class__.__name__}.")
        pass

    def get_start_template(self):
        """
        Returns the start template for the experiment.
        This can be used to initialize the LLM or for logging purposes.

        Returns:
            str: The start template string.
        """
        return self.START_TEMPLATE
