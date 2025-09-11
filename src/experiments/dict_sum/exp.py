import logging
import random
import numpy as np
import os
import re
import json
from typing import Dict, List, Tuple, Union, Any
from src.config import RunSettings
from src.experiments.dict_sum.dict_sum_util import (
    DictSumUtil,
    DictSumEvaluator,
    DictSumOutputParser,
    NewDictSumEvaluator,
)
from src._config.experiment_config import DictSumExecutionExperimentConfig
from src.experiments.base_experiment import BaseExperiment


class DictContainer:
    """Simple container to hold dictionary data for template compatibility."""

    def __init__(self, dict_data):
        self.dict = dict_data


class DictSumExecutionExperiment(BaseExperiment):
    """
    Experiment class for DictSum execution.

    This class sets up, runs, and evaluates an experiment where a large language
    model must maintain a running sum based on values from a dictionary.
    It supports both single-turn (all keys at once) and multi-turn (keys
    provided in batches) execution modes.
    """

    def __init__(
        self,
        common_config: RunSettings,
        experiment_specific_config: "DictSumExecutionExperimentConfig",
    ):
        """
        Initializes the DictSumExecutionExperiment.

        Args:
            common_config: General configuration applicable to all experiments.
            experiment_specific_config: Configuration specific to this experiment.
        """
        super().__init__(common_config)
        self.config = experiment_specific_config
        self._assign_config_attributes()

        # Initialize state attributes
        self.ground_truth_data: List[Dict[str, Any]] = []
        self.prompts: List[str | List[str]] = []
        self.llm_raw_outputs: List[str] = []
        self.processed_llm_outputs: List[List[Any]] = []
        self.per_instance_eval_results: List[Dict[str, Any]] = []
        self.evaluation_metrics: Dict[str, Any] = {}
        self.parsed_llm_outputs = []
        self.is_multi_turn = False
        self.setup()
        self._initialize_templates()

    def _assign_config_attributes(self):
        """Assigns attributes from the specific experiment config."""
        self.horizon_length = self.config.horizon_length
        self.num_samples = self.config.num_samples
        self.dict_size = self.config.dict_size
        self.min_input_value = self.config.min_input_value
        self.max_input_value = self.config.max_input_value
        self.working_capacity = self.config.working_capacity
        self.dataset_json_path = self.config.local_dataset_path
        self.pass_at_k = self.config.pass_at_k

    def _initialize_templates(self):
        """Initializes prompt templates based on working capacity and CoT mode."""
        if self.is_cot:
            self._initialize_cot_templates()
        else:
            self._initialize_non_cot_templates()

    def _initialize_cot_templates(self):
        """Initializes chain-of-thought prompt templates."""
        if self.working_capacity == 1:
            self.START_TEMPLATE = (
                "You are an AI assistant. I will provide you with a dictionary and then give you keys one at a time. "
                "Your task is to keep a running total (starting from 0) by adding the value associated with each key I provide.\n\n"
                "For each key I provide, think step by step about the calculation, then respond with the current total enclosed between <answer></answer> tags.\n\n"
                "Examples:\n"
                "Dictionary to maintain: {'apple': 5, 'banana': 0, 'cherry': 7, 'grape': -4, 'kiwi': 2, 'mango': -1}\n\n"
                "**User**: apple\n"
                "**Assistant**: The value for 'apple' is 5. Starting from 0, the running sum is 0 + 5 = 5. <answer>5</answer>\n\n"
                "**User**: banana\n"
                "**Assistant**: The value for 'banana' is 0. The current running sum is 5, so 5 + 0 = 5. <answer>5</answer>\n\n"
                "**User**: cherry\n"
                "**Assistant**: The value for 'cherry' is 7. The current running sum is 5, so 5 + 7 = 12. <answer>12</answer>\n\n"
                "**User**: grape\n"
                "**Assistant**: The value for 'grape' is -4. The current running sum is 12, so 12 + (-4) = 8. <answer>8</answer>\n\n"
                "**User**: kiwi\n"
                "**Assistant**: The value for 'kiwi' is 2. The current running sum is 8, so 8 + 2 = 10. <answer>f10</answer>\n\n"
                "**User**: mango\n"
                "**Assistant**: The value for 'mango' is -1. The current running sum is 10, so 10 + (-1) = 9. <answer>9</answer>\n\n"
                "Now, here is the actual task:"
                f"The dictionary to maintain:\n"
                f"{self.generator_util.dict}\n"
                "Ready to start!\n"
                "IMPORTANT: Always think step by step about each calculation, showing your work. Only put the final running sum in <answer> tags.\n\n"
            )
        else:
            self.START_TEMPLATE = (
                f"You are an AI assistant. I will provide you with a dictionary and then give you keys in groups of {self.working_capacity}. "
                f"Your task is to keep a running total (starting from 0) by adding the values associated with the keys I provide.\n\n"
                f"In each turn, I'll provide {self.working_capacity} keys (comma-separated). "
                f"Think step by step about the calculation, then respond with the current running sum, enclosed in <answer> tags.\n\n"
                f"Examples:\n"
                f"Dictionary to maintain: {{'apple': 5, 'banana': 0, 'cherry': 7, 'grape': -4, 'kiwi': 2, 'mango': -1}}\n\n"
                f"Example 1: keys in groups of 2\n"
                f"**User**: apple, banana\n"
                f"**Assistant**: Let me look up the values: 'apple' = 5, 'banana' = 0. Starting from 0, I calculate: 0 + 5 + 0 = 5. <answer>5</answer>\n\n"
                f"**User**: cherry, grape\n"
                f"**Assistant**: Let me look up the values: 'cherry' = 7, 'grape' = -4. Current running sum is 5, so: 5 + 7 + (-4) = 5 + 7 - 4 = 8. <answer>8</answer>\n\n"
                f"**User**: kiwi, mango\n"
                f"**Assistant**: Let me look up the values: 'kiwi' = 2, 'mango' = -1. Current running sum is 8, so: 8 + 2 + (-1) = 8 + 2 - 1 = 9. <answer>9</answer>\n\n"
                f"Example 2: keys in groups of 3\n"
                f"**User**: apple, banana, cherry\n"
                f"**Assistant**: Let me look up the values: 'apple' = 5, 'banana' = 0, 'cherry' = 7. Starting from 0, I calculate: 0 + 5 + 0 + 7 = 12. <answer>12</answer>\n\n"
                f"**User**: grape, kiwi, mango\n"
                f"**Assistant**: Let me look up the values: 'grape' = -4, 'kiwi' = 2, 'mango' = -1. Current running sum is 12, so: 12 + (-4) + 2 + (-1) = 12 - 4 + 2 - 1 = 9. <answer>10</answer>\n\n"
                f"Example 3: keys in groups of 6\n"
                f"**User**: apple, banana, cherry, grape, kiwi, mango\n"
                f"**Assistant**: Let me look up the values: 'apple' = 5, 'banana' = 0, 'cherry' = 7, 'grape' = -4, 'kiwi' = 2, 'mango' = -1. Starting from 0, I calculate: 0 + 5 + 0 + 7 + (-4) + 2 + (-1) = 5 + 7 - 4 + 2 - 1 = 9. <answer>9</answer>\n\n"
                f"Now, here is the actual task:"
                f"Dictionary to maintain:\n"
                f"{self.generator_util.dict}\n\n"
                f"Ready to start!\n"
                f"IMPORTANT: Always think step by step, showing how you look up each value and calculate the running sum. "
                f"Only put the final running sum in <answer> tags.\n\n"
            )

        examples = (
            f"Example traces with a different dictionary:\n"
            f"Dictionary: {{'grape': 3, 'honeydew': -2, 'kiwi': 8, 'lemon': 0, 'mango': -5, 'nectarine': 1}}\n\n"
            f"Example 1: keys in groups of 2\n"
            f"**User**: grape, honeydew\n"
            f"**Assistant**: Let me look up the values: 'grape' = 3, 'honeydew' = -2. Starting from 0, I calculate: 0 + 3 + (-2) = 3 - 2 = 1. <answer>1</answer>\n\n"
            f"**User**: kiwi, lemon\n"
            f"**Assistant**: Let me look up the values: 'kiwi' = 8, 'lemon' = 0. Current running sum is 1, so: 1 + 8 + 0 = 9. <answer>9</answer>\n\n"
            f"**User**: mango, nectarine\n"
            f"**Assistant**: Let me look up the values: 'mango' = -5, 'nectarine' = 1. Current running sum is 9, so: 9 + (-5) + 1 = 9 - 5 + 1 = 5. <answer>5</answer>\n\n"
            f"Example 2: keys in groups of 6\n"
            f"**User**: grape, honeydew, kiwi, lemon, mango, nectarine\n"
            f"**Assistant**: Let me look up the values: 'grape' = 3, 'honeydew' = -2, 'kiwi' = 8, 'lemon' = 0, 'mango' = -5, 'nectarine' = 1. Starting from 0, I calculate: 0 + 3 + (-2) + 8 + 0 + (-5) + 1 = 3 - 2 + 8 - 5 + 1 = 5. <answer>5</answer>\n\n"
        )

        self.FOLLOW_UP_TEMPLATE = (
            f"IMPORTANT: Always think step by step, showing how you look up each value and calculate the running sum.\n"
            f"Only put the final running sum in <answer> tags.\n\n"
        )
        # self.FOLLOW_UP_TEMPLATE += examples
        self.FOLLOW_UP_TEMPLATE += "Here are the next keys to process:\n"

    def _initialize_non_cot_templates(self):
        """Initializes non-chain-of-thought prompt templates."""
        if self.working_capacity == 1:
            self.START_TEMPLATE = (
                "You are an AI assistant. I will provide you with a dictionary and then give you keys one at a time. "
                "Your task is to keep a running total (starting from 0) by adding the value associated with each key I provide.\n\n"
                "For each key I provide, respond with the current total enclosed between <answer></answer> tags.\n\n"
                "Examples:\n"
                "Dictionary to maintain: {'apple': 5, 'banana': 0, 'cherry': 7, 'grape': -4, 'kiwi': 2, 'mango': -1}\n\n"
                "**User**: apple\n"
                "**Assistant**: <answer>5</answer>\n\n"
                "**User**: banana\n"
                "**Assistant**: <answer>5</answer>\n\n"
                "**User**: cherry\n"
                "**Assistant**: <answer>12</answer>\n\n"
                "**User**: grape\n"
                "**Assistant**: <answer>8</answer>\n\n"
                "**User**: kiwi\n"
                "**Assistant**: <answer>10</answer>\n\n"
                "**User**: mango\n"
                "**Assistant**: <answer>9</answer>\n\n"
                "Now, here is the actual task:"
                f"The dictionary to maintain:\n"
                f"{self.generator_util.dict}\n"
                "Ready to start!\n"
                "IMPORTANT: DO NOT OUTPUT ANY OTHER TEXT OUTSIDE ANSWER TAGS. Do not perform any calculations. Only provide the final running sum OF ALL TURNS in <answer> tags.\n\n"
            )
        else:
            self.START_TEMPLATE = (
                f"You are an AI assistant. I will provide you with a dictionary and then give you keys in groups of {self.working_capacity}. "
                f"Your task is to keep a running total (starting from 0) by adding the values associated with the keys I provide.\n\n"
                f"In each turn, I'll provide {self.working_capacity} keys (comma-separated). "
                f"Respond with the current running sum, enclosed in <answer> tags.\n\n"
                f"Examples:\n"
                f"Dictionary to maintain: {{'apple': 5, 'banana': 0, 'cherry': 7, 'grape': -4, 'kiwi': 2, 'mango': -1}}\n\n"
                f"Example 1: keys in groups of 2\n"
                f"**User**: apple, banana\n"
                f"**Assistant**: <answer>5</answer>\n\n"
                f"**User**: cherry, grape\n"
                f"**Assistant**: <answer>8</answer>\n\n"
                f"**User**: kiwi, mango\n"
                f"**Assistant**: <answer>9</answer>\n\n"
                f"Example 2: keys in groups of 3\n"
                f"**User**: apple, banana, cherry\n"
                f"**Assistant**: <answer>12</answer>\n\n"
                f"**User**: grape, kiwi, mango\n"
                f"**Assistant**: <answer>9</answer>\n\n"
                f"Example 3: keys in groups of 6\n"
                f"**User**: apple, banana, cherry, grape, kiwi, mango\n"
                f"**Assistant**: <answer>9</answer>\n\n"
                f"Now, here is the actual task:"
                f"Dictionary to maintain:\n"
                f"{self.generator_util.dict}\n\n"
                f"Ready to start!\n"
                f"IMPORTANT: DO NOT OUTPUT ANY OTHER TEXT OUTSIDE ANSWER TAGS. Only provide the final running sum OF ALL TURNS in <answer> tags.\n\n"
            )
        # self.FOLLOW_UP_TEMPLATE = (
        #     f"IMPORTANT: Only put the current running sum in <answer> tags.\n\n"
        # )
        # # self.FOLLOW_UP_TEMPLATE += examples
        # self.FOLLOW_UP_TEMPLATE += "Here are the next keys to process:\n"

    def _load_from_local_json(self):
        """Load ground-truth data from a local JSON file (generate_dataset_json.py format)."""
        if not os.path.isfile(self.dataset_json_path):  # type: ignore
            raise FileNotFoundError(
                f"Local dataset file not found: {self.dataset_json_path}"
            )

        with open(self.dataset_json_path, "r") as f:  # type: ignore
            data_json = json.load(f)

        # Handle both single-task and merged dataset formats
        if (
            "experiment_type" in data_json
            and data_json["experiment_type"] == "dict_sum"
        ):
            task_blob = data_json  # Single-task file
        elif "dict_sum" in data_json:
            task_blob = data_json["dict_sum"]  # Merged dataset
        else:
            raise ValueError("No valid dict_sum data found in JSON")

        # Extract required data
        self.ground_truth_data = task_blob.get("ground_truth_data", [])
        if not self.ground_truth_data:
            raise ValueError("No ground_truth_data found in the JSON for dict_sum")

        # Load dictionary
        if "dictionary" not in task_blob:
            raise ValueError("dictionary not found in provided dataset JSON")
        self.generator_util = DictContainer(task_blob["dictionary"])

        # Apply config overrides from JSON (take minimum to respect user limits)
        json_config = task_blob.get("config", {})
        for param in ["horizon_length", "num_samples", "dict_size"]:
            if param in json_config:
                current_val = getattr(self, param)
                setattr(
                    self,
                    param,
                    (
                        min(current_val, json_config[param])
                        if current_val
                        else json_config[param]
                    ),
                )

        # Apply data limits
        if self.num_samples and self.num_samples > 0:
            start = self.config.start_at
            end = start + self.num_samples
            self.ground_truth_data = self.ground_truth_data[start:end]

        # Truncate entries if horizon_length is specified
        if self.horizon_length and self.horizon_length > 0:
            for entry in self.ground_truth_data:
                for key in ["input", "values", "output"]:
                    if key in entry and len(entry[key]) > self.horizon_length:
                        entry[key] = entry[key][: self.horizon_length]

        # Update horizon_length to match actual data
        if self.ground_truth_data:
            self.horizon_length = len(self.ground_truth_data[0]["input"])

        logging.info(
            f"Loaded {len(self.ground_truth_data)} dict_sum samples from JSON ({self.dataset_json_path})"
        )

    def setup(self):
        """Sets up the experiment by either generating or loading data."""
        if self.dataset_json_path:
            self._load_from_local_json()
        else:
            # Generate data dynamically (original behavior)
            self.generator_util = DictSumUtil(
                num_pairs=self.dict_size,
                min_value=self.min_input_value,
                max_value=self.max_input_value,
                horizon_length=self.horizon_length,
                num_instances=self.num_samples,
                seed=self.eval_seed,
            )
            self.ground_truth_data = self.generator_util.entries
        self.evaluator = NewDictSumEvaluator(
            entries=self.ground_truth_data, working_capacity=self.working_capacity
        )

        logging.info(
            f"DictSumExecutionExperiment setup complete with {len(self.ground_truth_data)} samples, "
            f"{'dynamically generated' if not self.dataset_json_path else 'loaded from local JSON'}."
        )
        
        if self.pass_at_k > 0:
            logging.info(f"Pass@{self.pass_at_k} evaluation will be used.")
            logging.info(f"original number of samples: {len(self.ground_truth_data)}")
            # duplicate the ground truth data pass_at_k times
            duplicated_gt_data = []
            for entry in self.ground_truth_data:
                for _i in range(self.pass_at_k):
                    duplicated_gt_data.append(entry)
            logging.info(f"new number of samples: {len(duplicated_gt_data)}")
            self.ground_truth_data = duplicated_gt_data

    # --- Prompt Preparation ---

    def _get_few_shot_example_data(self) -> Tuple[Dict[str, int], List[Dict]]:
        """Generates static data for few-shot examples."""
        example_dictionary = {
            "apple": 5,
            "banana": 0,
            "cherry": 7,
            "grape": -4,
            "kiwi": 2,
            "mango": -1,
        }
        example_key_sequences = [
            ["apple", "banana", "cherry", "grape"],
            ["kiwi", "mango", "apple", "banana", "cherry"],
            ["grape", "apple", "kiwi", "kiwi", "apple", "grape", "banana", "kiwi"],
        ]

        examples_raw = []
        for keys in example_key_sequences:
            values = [example_dictionary[key] for key in keys]
            cumulative_sums = list(np.cumsum(values))
            examples_raw.append(
                {"input_keys": keys, "values": values, "results_full": cumulative_sums}
            )
        return example_dictionary, examples_raw

    def _format_few_shot_example(self, example_raw: Dict, example_idx: int) -> str:
        """Formats a single few-shot example into a text string."""
        input_display = str(example_raw["input_keys"])
        full_results = example_raw["results_full"]

        # In single-turn prompts, demonstrate varied working capacities for robustness.
        # W=1 always shows per-item sums. W>1 can show varied block sizes.
        if self.working_capacity == 1:
            w_demo = 1
            intro_text = f"Running sum after each key:"
            results_to_display = full_results
        else:
            w_demo = [1, 2, 3][example_idx] if example_idx < 3 else example_idx + 2
            if w_demo == 1:
                intro_text = f"Illustrating with sum after each key:"
                results_to_display = full_results
            else:
                intro_text = f"Illustrating with sum after each block of {w_demo} keys:"
                results_to_display = full_results[w_demo - 1 :: w_demo]

        results_str = f"<answer>{','.join(map(str, results_to_display))}</answer>"
        return f"Example {example_idx+1}:\nInput Keys: {input_display}\n{intro_text}\nResult: {results_str}"

    def prepare_prompts(self, llm_client_info=None):
        """
        Generates prompts for the single-turn Dict Sum task.
        """
        if not self.ground_truth_data or not self.generator_util:
            logging.warning(
                "Ground truth data or generator util not available. No prompts generated."
            )
            return self.prompts

        example_dict, examples_raw = self._get_few_shot_example_data()
        example_texts = [
            self._format_few_shot_example(ex, i) for i, ex in enumerate(examples_raw)
        ]
        all_examples_text = "\n\n".join(example_texts)

        if self.working_capacity == 1:
            intro = (
                "Your task is to calculate running sums for a list of keys from a dictionary.\n"
                "For each key, output the sum of all values from the start up to that key.\n"
                "You are NOT allowed to use any external tools or libraries, or write any code."
                "If needed, perform any reasoning inside <think> tags, but do not include any calculations outside of these tags.\n"
                "Your output must be a single line with comma-separated values inside one set of <answer></answer> tags."
            )
            final_instruction = (
                "Your Answer (single line, comma-separated values in one <answer> tag):"
            )
        else:
            intro = (
                f"Your task is to calculate running sums for a list of keys from a dictionary.\n"
                f"Process keys in groups of {self.working_capacity}. After each group, output the total sum from the beginning.\n"
                f"Your output must be a single line with comma-separated values inside one set of <answer></answer> tags."
            )
            final_instruction = f"Your Answer (single line, comma-separated sums in one <answer> tag):"

        base_prompt = (
            f"{intro}\n\n"
            f"Here are some examples using this example dictionary: {example_dict}\n\n"
            f"{all_examples_text}\n\n"
            f"Now you will be given the actual dictionary to use for the task:\n{self.generator_util.dict}\n\n"
        )

        self.prompts = [
            f"{base_prompt}Input Keys: {str(entry['input'])}\n{final_instruction}"
            for entry in self.ground_truth_data
        ]
        return self.prompts

    def prepare_multi_turn_prompts(self, num_tasks_per_turn=None):
        """

        Converts ground truth data into multi-turn prompts based on working_capacity.
        """
        self.is_multi_turn = True
        if not self.ground_truth_data:
            logging.warning(
                "Ground truth data not available. No multi-turn prompts generated."
            )
            return self.prompts

        self.prompts = []
        for entry in self.ground_truth_data:
            keys = entry["input"]
            # Chunk keys into sub-lists of size `working_capacity` and join them.
            # This creates one turn of dialogue for each chunk.
            turns = [
                ", ".join(keys[j : j + self.working_capacity])
                for j in range(0, len(keys), self.working_capacity)
            ]
            self.prompts.append(
                [((self.FOLLOW_UP_TEMPLATE or "") + turn) for turn in turns]
            )  # Filter out empty turns

        logging.debug(
            f"Prepared {len(self.prompts)} multi-turn prompt sequences, each with "
            f"up to {len(self.prompts[0]) if self.prompts else 0} turns."
        )
        return self.prompts

    # --- Processing and Evaluation ---

    def process_llm_outputs(self, llm_outputs_raw: List[str], enable_thinking=False):
        """Processes raw LLM outputs using the dedicated DictSumOutputParser."""
        self.llm_raw_outputs = llm_outputs_raw
        
        self.processed_llm_outputs = []
        parser = DictSumOutputParser()
        if self.is_multi_turn:        
            for raw_output in llm_outputs_raw:
                parsed_strings = parser.parse(raw_output, enable_thinking=enable_thinking)
                self.processed_llm_outputs.append(parsed_strings)

            return self.processed_llm_outputs
        else:
            for raw_output in llm_outputs_raw:
                parsed_strings = parser.parse_single_turn(raw_output, enable_thinking=enable_thinking)
                self.processed_llm_outputs.append(parsed_strings)

        logging.debug(
            f"Example processed LLM output: {self.processed_llm_outputs[0] if self.processed_llm_outputs else 'N/A'}"
        )

    @property
    def _num_expected_outputs(self) -> int:
        """Calculates the number of expected outputs per instance."""
        if self.horizon_length == 0:
            return 0
        # Use ceiling division to find how many blocks of size W are needed
        return (
            self.horizon_length + self.working_capacity - 1
        ) // self.working_capacity

    def _evaluate_single_instance(
        self, llm_output_sequence: List[Any], ground_truth_entry: Dict
    ) -> Dict:
        """Evaluates a single instance against its ground truth."""
        evaluator = DictSumEvaluator(
            entry=ground_truth_entry,
            working_capacity=self.working_capacity,
            is_multi_turn=self.is_multi_turn,
        )
        eval_result = evaluator.evaluate(llm_output_sequence)

        self.parsed_llm_outputs.append(eval_result["parsed_llm_output"])

        return {
            "input_keys": ground_truth_entry["input"],
            "input_values": ground_truth_entry["values"],
            "cumsum": ground_truth_entry["output"],
            "parsed_llm_output": eval_result["parsed_llm_output"],
            "effective_ground_truth": eval_result["effective_ground_truth"],
            "llm_raw_output": llm_output_sequence,
            "full_correctness": eval_result["full_correctness"],
            "prefix_correctness_list": eval_result["prefix_correctness_list"],
            "index_correctness_list": eval_result["index_correctness_list"],
            "step_correctness_list": eval_result["step_correctness_list"],
            "num_correct_steps": sum(eval_result.get("step_correctness_list", [])),
            "num_correct_prefixes": sum(eval_result.get("prefix_correctness_list", [])),
            "total_steps": len(eval_result.get("step_correctness_list", [])),
        }

    def evaluate_predictions(self):
        """Evaluates all processed LLM outputs and computes aggregate metrics."""
        if not self.processed_llm_outputs or len(self.processed_llm_outputs) != len(
            self.ground_truth_data
        ):
            logging.error(
                "LLM outputs are missing or do not match the number of ground truth instances."
            )
            self.evaluation_metrics = {
                "aggregated_metrics": {},
                "per_instance_results": [],
            }
            return self.evaluation_metrics

        num_outputs = self._num_expected_outputs
        sum_prefix_acc = np.zeros(num_outputs)
        sum_step_acc = np.zeros(num_outputs)
        sum_index_acc = np.zeros(num_outputs)

        for i, llm_output in enumerate(self.processed_llm_outputs):
            gt_entry = self.ground_truth_data[i]
            instance_result = self._evaluate_single_instance(llm_output, gt_entry)
            instance_result["prompt_index"] = i
            self.per_instance_eval_results.append(instance_result)

            # Aggregate positional accuracies
            sum_prefix_acc += np.array(
                instance_result["prefix_correctness_list"]
                + [0] * (num_outputs - len(instance_result["prefix_correctness_list"]))
            )
            sum_step_acc += np.array(
                instance_result["step_correctness_list"]
                + [0] * (num_outputs - len(instance_result["step_correctness_list"]))
            )
            sum_index_acc += np.array(
                instance_result["index_correctness_list"]
                + [0] * (num_outputs - len(instance_result["index_correctness_list"]))
            )
        
        num_instances = len(self.per_instance_eval_results)
        
        # Calculate bootstrap confidence intervals
        bootstrap_results = self.calculate_bootstrap_confidence_intervals()
        
        self.evaluation_metrics = {
            "aggregated_metrics": {
                "mean_full_correctness": (
                    np.mean(
                        [
                            res["full_correctness"]
                            for res in self.per_instance_eval_results
                        ]
                    )
                    if num_instances > 0
                    else 0.0
                ),
                "avg_prefix_accuracy_per_position": (
                    (sum_prefix_acc / num_instances).tolist()
                    if num_instances > 0
                    else []
                ),
                "avg_step_accuracy_per_position": (
                    (sum_step_acc / num_instances).tolist() if num_instances > 0 else []
                ),
                "avg_index_accuracy_per_position": (
                    (sum_index_acc / num_instances).tolist()
                    if num_instances > 0
                    else []
                ),
            },
            "bootstrap_confidence_intervals": bootstrap_results,
            "per_instance_results": self.per_instance_eval_results,
        }
        return self.evaluation_metrics

    def calculate_bootstrap_confidence_intervals(self, n_bootstrap=1000, confidence_level=0.95):
        """
        Calculate bootstrap confidence intervals for all metrics at every step.
        
        Args:
            n_bootstrap: Number of bootstrap samples to generate
            confidence_level: Confidence level for the intervals (default 0.95 for 95% CI)
            
        Returns:
            Dict containing bootstrap confidence intervals for all metrics
        """
        if not self.per_instance_eval_results:
            logging.warning("No evaluation results available for bootstrap sampling")
            return {}
            
        num_instances = len(self.per_instance_eval_results)
        num_outputs = self._num_expected_outputs
        
        # Calculate alpha for confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Initialize arrays to store bootstrap samples
        bootstrap_prefix_acc = np.zeros((n_bootstrap, num_outputs))
        bootstrap_step_acc = np.zeros((n_bootstrap, num_outputs))
        bootstrap_index_acc = np.zeros((n_bootstrap, num_outputs))
        bootstrap_full_correctness = np.zeros(n_bootstrap)
        
        # Perform bootstrap sampling
        for bootstrap_idx in range(n_bootstrap):
            # Sample with replacement
            sampled_indices = np.random.choice(num_instances, size=num_instances, replace=True)
            sampled_results = [self.per_instance_eval_results[i] for i in sampled_indices]
            
            # Calculate metrics for this bootstrap sample
            sum_prefix_acc = np.zeros(num_outputs)
            sum_step_acc = np.zeros(num_outputs)
            sum_index_acc = np.zeros(num_outputs)
            sum_full_correctness = 0
            
            for result in sampled_results:
                # Pad arrays to match num_outputs length
                prefix_padded = result["prefix_correctness_list"] + [0] * (num_outputs - len(result["prefix_correctness_list"]))
                step_padded = result["step_correctness_list"] + [0] * (num_outputs - len(result["step_correctness_list"]))
                index_padded = result["index_correctness_list"] + [0] * (num_outputs - len(result["index_correctness_list"]))
                
                sum_prefix_acc += np.array(prefix_padded)
                sum_step_acc += np.array(step_padded)
                sum_index_acc += np.array(index_padded)
                sum_full_correctness += result["full_correctness"]
            
            # Store bootstrap sample results
            bootstrap_prefix_acc[bootstrap_idx] = sum_prefix_acc / num_instances
            bootstrap_step_acc[bootstrap_idx] = sum_step_acc / num_instances
            bootstrap_index_acc[bootstrap_idx] = sum_index_acc / num_instances
            bootstrap_full_correctness[bootstrap_idx] = sum_full_correctness / num_instances
        
        # Calculate confidence intervals
        prefix_acc_ci = {
            "lower": np.percentile(bootstrap_prefix_acc, lower_percentile, axis=0).tolist(),
            "upper": np.percentile(bootstrap_prefix_acc, upper_percentile, axis=0).tolist(),
            "mean": np.mean(bootstrap_prefix_acc, axis=0).tolist()
        }
        
        step_acc_ci = {
            "lower": np.percentile(bootstrap_step_acc, lower_percentile, axis=0).tolist(),
            "upper": np.percentile(bootstrap_step_acc, upper_percentile, axis=0).tolist(),
            "mean": np.mean(bootstrap_step_acc, axis=0).tolist()
        }
        
        index_acc_ci = {
            "lower": np.percentile(bootstrap_index_acc, lower_percentile, axis=0).tolist(),
            "upper": np.percentile(bootstrap_index_acc, upper_percentile, axis=0).tolist(),
            "mean": np.mean(bootstrap_index_acc, axis=0).tolist()
        }
        
        full_correctness_ci = {
            "lower": float(np.percentile(bootstrap_full_correctness, lower_percentile)),
            "upper": float(np.percentile(bootstrap_full_correctness, upper_percentile)),
            "mean": float(np.mean(bootstrap_full_correctness))
        }
        
        bootstrap_results = {
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap,
            "num_instances": num_instances,
            "prefix_accuracy_ci": prefix_acc_ci,
            "step_accuracy_ci": step_acc_ci,
            "index_accuracy_ci": index_acc_ci,
            "full_correctness_ci": full_correctness_ci
        }
        
        logging.info(f"Calculated bootstrap confidence intervals with {n_bootstrap} samples")
        logging.info(f"Full correctness: {full_correctness_ci['mean']:.3f} "
                    f"[{full_correctness_ci['lower']:.3f}, {full_correctness_ci['upper']:.3f}]")
        
        return bootstrap_results

    # --- Reporting & Plotting ---

    def get_experiment_params_for_logging(self) -> Dict[str, Any]:
        """Returns experiment-specific parameters for logging."""
        return {
            "experiment_type": "dict_sum_execution",
            "horizon_length": self.horizon_length,
            "num_samples": self.num_samples,
            "dict_size": self.dict_size,
            "min_input_value": self.min_input_value,
            "max_input_value": self.max_input_value,
            "working_capacity": self.working_capacity,
            "is_multi_turn": self.is_multi_turn,
        }

    def run_plotting(self, run_output_dir: str, base_filename: str):

        self.evaluator.save_json_log(output_dir=run_output_dir)
        # self.evaluator.calculate_drop_step_with_bootstrap_ci()

        """Plots average prefix and step accuracy versus position."""
        if self.horizon_length == 0:
            logging.info("Horizon length is 0. Skipping accuracy plotting.")
            return

        aggregated_metrics = self.evaluation_metrics.get("aggregated_metrics", {})
        bootstrap_ci = self.evaluation_metrics.get("bootstrap_confidence_intervals", {})
        
        avg_prefix_acc = aggregated_metrics.get("avg_prefix_accuracy_per_position")
        avg_step_acc = aggregated_metrics.get("avg_step_accuracy_per_position")
        avg_index_acc = aggregated_metrics.get("avg_index_accuracy_per_position")

        if not avg_prefix_acc and not avg_step_acc and not avg_index_acc:
            logging.info("No accuracy data available for plotting.")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logging.warning("Matplotlib not installed. Skipping plotting.")
            return

        num_plot_points = self._num_expected_outputs
        positions = np.arange(1, num_plot_points + 1)

        plt.figure(figsize=(12, 7))

        # Plot with confidence intervals if available
        if avg_prefix_acc:
            if bootstrap_ci and "prefix_accuracy_ci" in bootstrap_ci:
                prefix_ci = bootstrap_ci["prefix_accuracy_ci"]
                lower_err = np.array(avg_prefix_acc) - np.array(prefix_ci["lower"])
                upper_err = np.array(prefix_ci["upper"]) - np.array(avg_prefix_acc)
                plt.errorbar(positions, avg_prefix_acc, yerr=[lower_err, upper_err], 
                            fmt="o-", label="Avg. Prefix Accuracy", capsize=3)
            else:
                plt.plot(positions, avg_prefix_acc, "o-", label="Avg. Prefix Accuracy")
                
        if avg_step_acc:
            if bootstrap_ci and "step_accuracy_ci" in bootstrap_ci:
                step_ci = bootstrap_ci["step_accuracy_ci"]
                lower_err = np.array(avg_step_acc) - np.array(step_ci["lower"])
                upper_err = np.array(step_ci["upper"]) - np.array(avg_step_acc)
                plt.errorbar(positions, avg_step_acc, yerr=[lower_err, upper_err], 
                            fmt="x--", label="Avg. Step Accuracy", capsize=3)
            else:
                plt.plot(positions, avg_step_acc, "x--", label="Avg. Step Accuracy")
                
        if avg_index_acc:
            if bootstrap_ci and "index_accuracy_ci" in bootstrap_ci:
                index_ci = bootstrap_ci["index_accuracy_ci"]
                lower_err = np.array(avg_index_acc) - np.array(index_ci["lower"])
                upper_err = np.array(index_ci["upper"]) - np.array(avg_index_acc)
                plt.errorbar(positions, avg_index_acc, yerr=[lower_err, upper_err], 
                            fmt="s-.", label="Avg. Index Accuracy", capsize=3)
            else:
                plt.plot(positions, avg_index_acc, "s-.", label="Avg. Index Accuracy")

        title_mode = "Multi-turn" if self.is_multi_turn else "Single-turn"
        ci_text = " (with 95% CI)" if bootstrap_ci else ""
        plt.title(
            f"Dict Sum Accuracy (H={self.horizon_length}, W={self.working_capacity}) - {title_mode}{ci_text}\n"
            f"{self.run_timestamp}_{self.model_name_for_path} (N={self.num_samples})"
        )
        xlabel = (
            f"Position Index (Output Step, W={self.working_capacity})"
            if self.working_capacity > 1
            else "Position in List (W=1)"
        )
        plt.xlabel(xlabel)
        plt.ylabel("Average Accuracy")

        plt.ylim(-0.05, 1.05)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        if plt.gca().has_data():
            plt.legend()

        os.makedirs(run_output_dir, exist_ok=True)
        plot_filename = os.path.join(
            run_output_dir, f"{base_filename}_accuracy_plot.png"
        )
        plt.savefig(plot_filename)
        plt.close()
        logging.info(f"Saved accuracy plot to {plot_filename}")

    def get_results_to_save(self) -> Dict[str, Any]:
        """Appends the start template to the results to be saved."""
        res = super().get_results_to_save()
        res["start_template"] = self.START_TEMPLATE
        return res
