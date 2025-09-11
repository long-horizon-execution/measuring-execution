import math
import random
import re
import logging
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Union, Optional, Tuple

import numpy as np


def have_a_word(l, k):
    with open("words_alpha.txt") as f:
        all_words = [i for i in f.read().splitlines() if len(i) == l]
        return random.sample(all_words, k)


class DictCreator:
    """
    A class for creating seeded random dictionaries with gibberish word keys
    and numeric values within a specified range.
    """

    def __init__(
        self,
        num_pairs: int,
        min_value: int = 0,
        max_value: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize the DictCreator with parameters for dictionary creation.
        Args:
            num_pairs (int): Number of key-value pairs to create
            min_value (int): Minimum value for the numeric values (inclusive)
            max_value (int): Maximum value for the numeric values (inclusive)
            seed (Optional[int]): Seed for random number generation for reproducibility
        """

        self.num_pairs = num_pairs
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed

    def create_dict(self, n: int, a: int, b: int) -> Dict[str, int]:
        """
        Create a dictionary with N key-value pairs where keys are random gibberish words
        and values are random numbers between a and b (inclusive).

        Args:
            n (int): Number of key-value pairs to create
            a (Union[int, float]): Lower bound for values (inclusive)
            b (Union[int, float]): Upper bound for values (inclusive)
            use_integers (bool): If True, generate integer values; if False, generate floats

        Returns:
            Dict[str, Union[int, float]]: Dictionary with random gibberish word keys and numeric values

        Raises:
            ValueError: If a > b
        """
        if a > b:
            raise ValueError(f"Lower bound {a} cannot be greater than upper bound {b}")

        # Reset seed if specified to ensure reproducible results
        if self.seed is not None:
            random.seed(self.seed)

        # Generate unique random words as keys
        result_dict = {}

        keys = have_a_word(5, n)  # Generate n unique words of length 5

        # fill the dictionary with unique keys and random values
        for word in keys:
            if word not in result_dict:
                # Generate random value
                value = random.randint(int(a), int(b))
                result_dict[word] = value

        return result_dict


class DictSumUtil:
    def __init__(
        self,
        num_pairs: int,
        min_value: int,
        max_value: int,
        horizon_length: int,
        num_instances: int,
        seed: Optional[int] = None,
    ):
        """
        Initialize the DictSumUtil with a dictionary and horizon length.

        Args:
            dict (Dict[str, int]): Dictionary with string keys and integer values
            horizon_length (int): Length of the horizon for processing
            num_instances (int): Number of rollout instances to generate
        """
        self.dict_creator = DictCreator(num_pairs, min_value, max_value, seed)
        self.dict = self.dict_creator.create_dict(num_pairs, min_value, max_value)
        self.horizon_length = horizon_length
        self.num_instances = num_instances
        self.entries = self.generate_rollout_instances()

    def generate_rollout_instance(self):
        """
        Generate a single rollout instance by selecting a random subset of the dictionary.

        Returns:
            Dict[str, int]: A dictionary containing a random subset of the original dictionary
        """
        keys = random.choices(list(self.dict.keys()), k=self.horizon_length)

        # we respect the order, however since this is addition, the order does not matter, still we will maintain it
        cumsum = 0
        prefix_sum = []
        values = []
        for key in keys:
            values.append(self.dict[key])
            prefix_sum.append(cumsum + self.dict[key])
            cumsum += self.dict[key]

        return {
            "input": keys,
            "output": prefix_sum,
            "values": values,
        }  # input  # output

    def generate_rollout_instances(self):
        """
        Generate multiple rollout instances based on the specified number of instances.
        Returns:
            List[Dict[str, Union[List[str], List[int]]]]: A list of dictionaries, each containing a rollout instance
            and its corresponding prefix sum.
        """
        instances = []
        for _ in range(self.num_instances):
            instance = self.generate_rollout_instance()
            instances.append(instance)

        return instances


class DictSumOutputParser:
    """
    Parses and validates Dict Sum LLM outputs on a per-answer basis.
    Similar to PrefixSumOutputParser but for dict sum tasks.
    """

    # The token to use for any chunk that is not a well-formed integer answer.
    NOOP_TOKEN = "<NoOp>"
    INVALID_FORMAT_TOKEN = "<IncorrectFormat>"
    # Pattern to find answer tags anywhere in the text (allowing chain of thought before)
    _ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    # Safe step delimiter used when multiple instance outputs are concatenated per step
    # Using ASCII Unit Separator (0x1F), highly unlikely to appear in model outputs
    STEP_DELIMITER = "\x1f"

    raw_llm_output: Any = []

    def parse(self, raw_output: Any, enable_thinking=False) -> List[Union[int, str]]:
        """
        Parses raw LLM output, validating each comma-separated chunk.
        """
        if enable_thinking:
            # remove thinking steps by replacing content within <think> tags
            output_text = re.sub(
                r"<think>.*?</think>", "", str(raw_output), flags=re.DOTALL
            ).strip()
        else:
            output_text = str(raw_output).strip()
            if not output_text:
                return []

        # Step 2: Split the entire output into chunks based on the per-step delimiter.
        # Prefer the safe delimiter if present; fall back to legacy '|'.
        delimiter = self.STEP_DELIMITER if self.STEP_DELIMITER in output_text else "|"
        chunks = [chunk.strip() for chunk in output_text.split(delimiter)]

        self.raw_llm_output = chunks

        validated_results: List[Union[int, str]] = []

        # Step 3: Validate each chunk individually.
        for chunk in chunks:
            # handle chain of thought reasoning by removing everything apart from the answer tags
            answer_match = self._ANSWER_PATTERN.search(str(chunk))
            if answer_match:
                content = answer_match.group(1).strip()
                try:
                    # Try to convert to integer
                    integer_value = int(content)
                    validated_results.append(integer_value)
                except ValueError:
                    validated_results.append(self.INVALID_FORMAT_TOKEN)
            else:
                # If no <answer> tag is found, mark as error
                logging.warning(
                    f"No <answer> tag found in chunk: {chunk}. Marking as NOOP."
                )
                validated_results.append(self.NOOP_TOKEN)

        return validated_results

    def parse_single_turn(self, raw_output, enable_thinking=False):
        """
        Parses raw LLM output for single-turn DictSum tasks, validating each comma-separated chunk.
        """
        if enable_thinking:
            # remove thinking steps by replacing content within <think> tags
            output_text = re.sub(
                r"<think>.*?</think>", "", str(raw_output), flags=re.DOTALL
            ).strip()
        else:
            output_text = str(raw_output).strip()
            if not output_text:
                return []

        self.raw_llm_output = raw_output
        logging.debug(f"Raw LLM output: {self.raw_llm_output}")
        # get the content within <answer> tags
        answer_match = self._ANSWER_PATTERN.search(str(output_text))
        if answer_match:
            content = answer_match.group(1).strip()
            # split by commas
            chunks = [chunk.strip() for chunk in content.split(",")]

            # check for each chunk if it is a valid integer else put INVALID_FORMAT_TOKEN
            validated_results: List[Union[int, str]] = []
            for chunk in chunks:
                try:
                    # Try to convert to integer
                    integer_value = int(chunk)
                    validated_results.append(integer_value)
                except ValueError:
                    validated_results.append(self.INVALID_FORMAT_TOKEN)
            return validated_results
        else:
            # put noop token
            return [self.NOOP_TOKEN]


class Metrics:
    """
    A class to hold and manage evaluation metrics for DictSum tasks.
    """

    def __init__(self, expected_steps):
        self.full_correctness = 0.0
        self.prefix_correctness_array = np.zeros(expected_steps, dtype=float)
        self.index_correctness_array = np.zeros(expected_steps, dtype=float)
        self.step_correctness_array = np.zeros(expected_steps, dtype=float)
        self.nan_count_array = np.zeros(expected_steps, dtype=float)
        self.invalid_count_array = np.zeros(expected_steps, dtype=float)

        # Bootstrap confidence interval arrays
        self.prefix_ci_min_array = np.zeros(expected_steps, dtype=float)
        self.prefix_ci_max_array = np.zeros(expected_steps, dtype=float)
        self.index_ci_min_array = np.zeros(expected_steps, dtype=float)
        self.index_ci_max_array = np.zeros(expected_steps, dtype=float)
        self.step_ci_min_array = np.zeros(expected_steps, dtype=float)
        self.step_ci_max_array = np.zeros(expected_steps, dtype=float)

    def reset(self):
        """
        Reset all metrics to their initial state.
        """
        self.full_correctness = 0.0
        self.prefix_correctness_array.fill(0)
        self.index_correctness_array.fill(0)
        self.step_correctness_array.fill(0)
        self.nan_count_array.fill(0)
        self.invalid_count_array.fill(0)
        self.prefix_ci_min_array.fill(0)
        self.prefix_ci_max_array.fill(0)
        self.index_ci_min_array.fill(0)
        self.index_ci_max_array.fill(0)
        self.step_ci_min_array.fill(0)
        self.step_ci_max_array.fill(0)


class NewDictSumEvaluator:
    """
    Evaluates LLM output for the DictSum task incrementally for all instances at once.
    """

    def __init__(
        self,
        entries: List[dict],
        working_capacity: int = 1,
        wandb_logger: Optional[Any] = None,
    ):
        """
        Args:
            entries (List[dict]): A list of ground truth entries from DictSumGenerator,
                                  each containing 'input' and 'output'.
            working_capacity (int): The number of input numbers processed by the LLM at once.
            wandb_logger: Optional Weights & Biases logger.
        """
        self.parser = DictSumOutputParser()
        self.wandb_logger = wandb_logger

        # Initialize JSON logging
        self.step_logs = []

        # convert entries to a more suitable format, which is a per step dict for all instances
        self.entries = self._convert_data_format(entries)
        self.k = len(self.entries["input"])  # number of steps

        # reshape entries from SxN to S'xNxW format where we group consecutive steps together
        # working_capacity determines how many consecutive steps are grouped together
        self.num_samples = len(self.entries["input"][0])
        num_step_groups = math.ceil(self.k / working_capacity)

        new_entries = {"input": [], "values": [], "output": []}

        self.THRESHOLD = 0.5  # Threshold for prefix accuracy to start calculating bootstrap confidence intervals
        self.bootstrap_results = (
            None  # Store bootstrap results for drop step calculation
        )

        for group_idx in range(num_step_groups):
            start_step = group_idx * working_capacity
            end_step = min(start_step + working_capacity, self.k)

            group_input_samples = []
            group_values_samples = []
            group_output_samples = []

            for sample_idx in range(self.num_samples):
                # For each sample, collect values from consecutive steps
                sample_input = [
                    self.entries["input"][step][sample_idx]
                    for step in range(start_step, end_step)
                ]
                sample_values = [
                    self.entries["values"][step][sample_idx]
                    for step in range(start_step, end_step)
                ]
                # Only keep the output from the last step in the group
                sample_output = self.entries["output"][end_step - 1][sample_idx]

                group_input_samples.append(sample_input)
                group_values_samples.append(sample_values)
                group_output_samples.append(sample_output)

            new_entries["input"].append(group_input_samples)
            new_entries["values"].append(group_values_samples)
            new_entries["output"].append(group_output_samples)

        self.entries = new_entries

        self.expected_output_len = len(self.entries["output"])
        logging.info(f"K: {self.k}, Expected Output Length: {self.expected_output_len}")

        # initialise metrics
        self.metrics = Metrics(self.expected_output_len)

        # Store per-sample correctness across all steps (2D: [step][sample])
        # This tracks whether each sample was correct up to each step
        self.prefix_correctness_per_step = []
        # Store per-sample correctness for index and step accuracy (for bootstrapping)
        self.index_correctness_per_step = []
        self.step_correctness_per_step = []
        for step in range(self.expected_output_len):
            self.prefix_correctness_per_step.append(
                np.ones(self.num_samples, dtype=int)
            )
            self.index_correctness_per_step.append(
                np.zeros(self.num_samples, dtype=int)
            )
            self.step_correctness_per_step.append(np.zeros(self.num_samples, dtype=int))

        # initialise llm's own output stream
        self.llm_expected_prefix_sum = [[] for _ in range(self.expected_output_len)]

        self.steps_evaluated = 0  # Track the number of steps evaluated

    def set_wandb_logger(self, wandb_logger):
        """
        Set the Weights & Biases logger for logging metrics.
        """
        self.wandb_logger = wandb_logger

    def save_json_log(self, output_dir):
        """
        Save the accumulated step logs to a JSON file.
        """
        try:
            # Ensure the directory exists if the path contains directories
            log_dir = os.path.dirname(output_dir)
            if log_dir:  # Only create directory if there's a directory part
                os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(output_dir, "eval_step_results.json")

            log_data = {"step_logs": self.step_logs}

            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2)

            logging.info(f"JSON log saved to: {log_path}")
        except Exception as e:
            logging.error(f"Failed to save JSON log: {e}")

    def add_noise_to_ground_truth(self, noise_level=10, err_probability=0.1):
        """
        independently add noise_level + output to each ground truth output with a probability of err_probability.
        """
        for step in range(self.expected_output_len):
            for i in range(self.num_samples):
                if random.random() < err_probability:
                    noise = random.randint(-1 * noise_level, noise_level)
                    self.entries["output"][step][i] += noise
                    logging.debug(
                        f"Added noise {noise} to ground truth output at step {step}, sample {i}"
                    )

    def _convert_data_format(self, data):
        """
        Convert data from the original format to a step-wise format.

        Args:
            data: List of dictionaries, each containing 'input', 'output', and 'values' keys

        Returns:
            dict with keys:
            - 'input': List where input[i] contains all inputs at step i across all sequences
            - 'values': List where values[i] contains all values at step i across all sequences
            - 'output': List where output[i] contains all outputs at step i across all sequences
        """
        if not data:
            return {"input": [], "values": [], "output": []}

        # Find the maximum sequence length
        max_length = max(len(seq["input"]) for seq in data)

        # Initialize the result structure
        result = {
            "input": [[] for _ in range(max_length)],
            "values": [[] for _ in range(max_length)],
            "output": [[] for _ in range(max_length)],
        }

        # Process each sequence
        for seq in data:
            seq_length = len(seq["input"])

            # Add data for each step in this sequence
            for step in range(seq_length):
                result["input"][step].append(seq["input"][step])
                result["values"][step].append(seq["values"][step])
                result["output"][step].append(seq["output"][step])

        return result

    def _handle_nans_and_invalids(self, llm_output, step):
        nan_count = sum(
            1
            for i in range(len(llm_output))
            if llm_output[i] == DictSumOutputParser.NOOP_TOKEN
        )
        invalid_count = sum(
            1
            for i in range(len(llm_output))
            if llm_output[i] == DictSumOutputParser.INVALID_FORMAT_TOKEN
        )

        self.metrics.nan_count_array[step] = nan_count / self.num_samples
        self.metrics.invalid_count_array[step] = invalid_count / self.num_samples

    def fill_in_expected_sum(self, step):
        """
        fill in the expected prefix sum till the current step with ground truth outputs.
        """
        if step is None:
            return

        for i in range(step):
            self.llm_expected_prefix_sum[i] = self.entries["output"][i]

    def evaluate_step(
        self,
        llm_output,
        step,
        num_tokens_generated=0,
        enable_thinking=False,
        filled_history=False,
    ):
        """
        This function evaluates the LLM output for a single step across all instances.

        Args:
            llm_output: The LLM output to be parsed (can be raw or pre-parsed)
            step: Current step number
            num_tokens_generated: Number of tokens generated by the LLM
            enable_thinking: Whether to enable thinking mode parsing
            filled_history: Whether history is filled with ground truth
            raw_llm_output: Raw LLM output before parsing (for logging)
        """
        # Store raw output for logging

        # parse the LLM output
        parsed_llm_output = self.parser.parse(
            llm_output, enable_thinking=enable_thinking
        )
        raw_llm_output = self.parser.raw_llm_output
        if len(parsed_llm_output) < (self.num_samples):
            # pad with NOOP_TOKEN if the output is shorter than num_samples
            logging.warning(
                f"LLM output for step {step} is shorter than expected ({len(parsed_llm_output)} < {self.num_samples}). Padding with NOOP_TOKEN."
            )
            parsed_llm_output = parsed_llm_output[: self.num_samples] + [
                DictSumOutputParser.NOOP_TOKEN
            ] * (self.num_samples - len(parsed_llm_output))
        elif len(parsed_llm_output) > self.num_samples:
            # truncate if the output is longer than num_samples
            logging.warning(
                f"LLM output for step {step} is longer than expected ({len(parsed_llm_output)} > {self.num_samples}). Truncating."
            )
            parsed_llm_output = parsed_llm_output[: self.num_samples]

        # Extract the expected outputs for the current step
        expected_outputs = self.entries["output"][step]

        self._handle_nans_and_invalids(parsed_llm_output, step)

        # index accuracy
        self.metrics.index_correctness_array[step] = self._calculate_index_accuracy(
            parsed_llm_output, expected_outputs, step
        )

        # prefix accuracy
        self.metrics.prefix_correctness_array[step] = self._calculate_prefix_accuracy(
            parsed_llm_output, expected_outputs, step
        )

        # step accuracy
        self.metrics.step_correctness_array[step] = self._calculate_step_accuracy(
            parsed_llm_output, step, filled_history=filled_history
        )

        # Calculate bootstrap confidence intervals
        bootstrap_cis = self.calculate_bootstrap_ci(step)

        # Store confidence intervals in metrics
        if "index" in bootstrap_cis:
            (
                self.metrics.index_ci_min_array[step],
                self.metrics.index_ci_max_array[step],
            ) = bootstrap_cis["index"]
        if "step" in bootstrap_cis:
            (
                self.metrics.step_ci_min_array[step],
                self.metrics.step_ci_max_array[step],
            ) = bootstrap_cis["step"]
        if "prefix" in bootstrap_cis:
            (
                self.metrics.prefix_ci_min_array[step],
                self.metrics.prefix_ci_max_array[step],
            ) = bootstrap_cis["prefix"]

        # Log detailed step information for JSON
        self._log_step_details(
            step=step,
            raw_llm_output=raw_llm_output,
            parsed_llm_output=parsed_llm_output,
            expected_outputs=expected_outputs,
            num_tokens_generated=num_tokens_generated,
            filled_history=filled_history,
        )

        self.steps_evaluated += 1

        # log to wandb
        self.log_to_wandb(step, num_tokens_generated)

    def _log_step_details(
        self,
        step: int,
        raw_llm_output: Any,
        parsed_llm_output: List[Union[int, str]],
        expected_outputs: List[int],
        num_tokens_generated: int,
        filled_history: bool,
    ):
        """
        Log detailed information for each step and instance to the JSON log.
        """
        step_log = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "num_tokens_generated": num_tokens_generated,
            "filled_history": filled_history,
            "aggregated_metrics": {
                "index_accuracy": float(self.metrics.index_correctness_array[step]),
                "prefix_accuracy": float(self.metrics.prefix_correctness_array[step]),
                "step_accuracy": float(self.metrics.step_correctness_array[step]),
                "nan_fraction": float(self.metrics.nan_count_array[step]),
                "invalid_fraction": float(self.metrics.invalid_count_array[step]),
                "confidence_intervals": {
                    "index_accuracy_ci_min": float(
                        self.metrics.index_ci_min_array[step]
                    ),
                    "index_accuracy_ci_max": float(
                        self.metrics.index_ci_max_array[step]
                    ),
                    "prefix_accuracy_ci_min": float(
                        self.metrics.prefix_ci_min_array[step]
                    ),
                    "prefix_accuracy_ci_max": float(
                        self.metrics.prefix_ci_max_array[step]
                    ),
                    "step_accuracy_ci_min": float(self.metrics.step_ci_min_array[step]),
                    "step_accuracy_ci_max": float(self.metrics.step_ci_max_array[step]),
                },
            },
            "instances": [],
        }

        # Log per-instance details
        for instance_idx in range(self.num_samples):
            # Get input data for this instance and step
            instance_input = (
                self.entries["input"][step][instance_idx]
                if step < len(self.entries["input"])
                else None
            )
            instance_values = (
                self.entries["values"][step][instance_idx]
                if step < len(self.entries["values"])
                else None
            )

            # Calculate individual accuracies for this instance
            instance_parsed_output = (
                parsed_llm_output[instance_idx]
                if instance_idx < len(parsed_llm_output)
                else DictSumOutputParser.NOOP_TOKEN
            )
            instance_expected_output = (
                expected_outputs[instance_idx]
                if instance_idx < len(expected_outputs)
                else None
            )

            # Index accuracy for this instance
            index_correct = (
                isinstance(instance_parsed_output, int)
                and instance_parsed_output == instance_expected_output
            )

            # Prefix accuracy for this instance (up to current step)
            prefix_correct = bool(self.prefix_correctness_per_step[step][instance_idx])

            # Step accuracy for this instance (if we have LLM's expected sum)
            step_correct = False
            if step < len(self.llm_expected_prefix_sum) and instance_idx < len(
                self.llm_expected_prefix_sum[step]
            ):
                if step == 0:
                    step_correct = index_correct
                else:
                    if instance_values is not None:
                        if filled_history:
                            expected_step_sum = sum(instance_values) + (
                                self.entries["output"][step - 1][instance_idx]
                                if step > 0
                                else 0
                            )
                        else:
                            expected_step_sum = sum(instance_values) + (
                                self.llm_expected_prefix_sum[step - 1][instance_idx]
                                if step > 0
                                else 0
                            )
                        step_correct = (
                            isinstance(
                                self.llm_expected_prefix_sum[step][instance_idx], int
                            )
                            and self.llm_expected_prefix_sum[step][instance_idx]
                            == expected_step_sum
                        )

            instance_log = {
                "instance_id": instance_idx,
                "input_given": instance_input,
                "input_values": instance_values,
                "raw_llm_output": (
                    str(raw_llm_output[instance_idx])
                    if instance_idx < len(raw_llm_output)
                    else "N/A"
                ),
                "parsed_answer": instance_parsed_output,
                "expected_answer": instance_expected_output,
                "accuracies": {
                    "index_correct": float(index_correct),
                    "prefix_correct": float(prefix_correct),
                    "step_correct": float(step_correct),
                },
                "is_noop": instance_parsed_output == DictSumOutputParser.NOOP_TOKEN,
                "is_invalid_format": instance_parsed_output
                == DictSumOutputParser.INVALID_FORMAT_TOKEN,
            }

            step_log["instances"].append(instance_log)

        self.step_logs.append(step_log)

    def log_to_wandb(self, step, num_tokens_generated=0):
        """
        Log metrics for this step to Weights & Biases if wandb_logger is provided.
        """
        if not self.wandb_logger:
            return

        metrics_to_log = {
            "prefix_accuracy": self.metrics.prefix_correctness_array[step],
            "index_accuracy": self.metrics.index_correctness_array[step],
            "step_accuracy": self.metrics.step_correctness_array[step],
            "nan_fraction": self.metrics.nan_count_array[step],
            "invalid_fraction": self.metrics.invalid_count_array[step],
            "tokens_generated_per_step": num_tokens_generated,
            "index_accuracy_ci_min": self.metrics.index_ci_min_array[step],
            "index_accuracy_ci_max": self.metrics.index_ci_max_array[step],
            "prefix_accuracy_ci_min": self.metrics.prefix_ci_min_array[step],
            "prefix_accuracy_ci_max": self.metrics.prefix_ci_max_array[step],
            "step_accuracy_ci_min": self.metrics.step_ci_min_array[step],
            "step_accuracy_ci_max": self.metrics.step_ci_max_array[step],
        }

        # Log the metrics at the desired step
        self.wandb_logger.log(metrics_to_log, step=step)
        logging.info(
            f"Step {step} - "
            f"Prefix Accuracy: {self.metrics.prefix_correctness_array[step]}, "
            f"Index Accuracy: {self.metrics.index_correctness_array[step]}, "
            f"Step Accuracy: {self.metrics.step_correctness_array[step]}, "
            f"NaN Fraction: {self.metrics.nan_count_array[step]}, "
            f"Invalid Fraction: {self.metrics.invalid_count_array[step]}, "
            f"Tokens Generated: {num_tokens_generated}"
            f", Index Acc CI: ({self.metrics.index_ci_min_array[step]}, {self.metrics.index_ci_max_array[step]})"
            f", Prefix Acc CI: ({self.metrics.prefix_ci_min_array[step]}, {self.metrics.prefix_ci_max_array[step]})"
            f", Step Acc CI: ({self.metrics.step_ci_min_array[step]}, {self.metrics.step_ci_max_array[step]})"
        )

    def _calculate_step_accuracy(self, llm_output, step, filled_history=False):
        """
        check if llm performs the step addition correctly given its own previous output.
        Also stores per-sample correctness for bootstrapping.
        """

        # replace all NOOP_TOKENs with 0
        if step == 0:
            # For the first step, we assume the model starts with a sum of 0
            llm_output = [0 if not isinstance(x, int) else x for x in llm_output]
        else:
            if not filled_history:
                llm_output = [
                    (
                        x
                        if isinstance(x, int)
                        else self.llm_expected_prefix_sum[step - 1][i]
                    )
                    for i, x in enumerate(llm_output)
                ]
            else:
                llm_output = [
                    x if isinstance(x, int) else -99999
                    for i, x in enumerate(llm_output)
                ]

        # fill the llm_expected_prefix_sum for this step. to be used in the next step
        self.llm_expected_prefix_sum[step] = llm_output
        if step == 0:
            # For the first step, copy index correctness to step correctness
            for i in range(self.num_samples):
                self.step_correctness_per_step[step][i] = (
                    self.index_correctness_per_step[step][i]
                )
            return self.metrics.index_correctness_array[step]

        correct_count = 0
        for i in range(self.num_samples):

            if filled_history:
                expected_prefix_sum = (
                    sum(self.entries["values"][step][i])
                    + self.entries["output"][step - 1][
                        i
                    ]  # use the previous step's output as we are preempting the LLM to use our ground truth
                )
            else:
                # add current values to the previous expected prefix sum
                expected_prefix_sum = (
                    sum(self.entries["values"][step][i])
                    + self.llm_expected_prefix_sum[step - 1][i]
                )
            logging.info(
                f"Step {step}, Sample {i}: LLM Output = {llm_output[i]}, Expected Prefix Sum = {expected_prefix_sum}"
            )
            if self.llm_expected_prefix_sum[step][i] == expected_prefix_sum:
                correct_count += 1
                self.step_correctness_per_step[step][i] = 1
            else:
                self.step_correctness_per_step[step][i] = 0

        return correct_count / self.num_samples if self.num_samples > 0 else 0.0

    def _calculate_index_accuracy(self, llm_output, expected_outputs, step):
        """
        Calculate index accuracy for the given LLM output and expected outputs.
        Also stores per-sample correctness for bootstrapping.
        """
        # Calculate per-sample correctness
        for i in range(len(llm_output)):
            if i < len(expected_outputs) and llm_output[i] == expected_outputs[i]:
                self.index_correctness_per_step[step][i] = 1
            else:
                self.index_correctness_per_step[step][i] = 0

        # Calculate overall accuracy
        correct_count = sum(
            1 for i in range(len(llm_output)) if llm_output[i] == expected_outputs[i]
        )
        return correct_count / len(expected_outputs) if expected_outputs else 0.0

    def _calculate_prefix_accuracy(self, llm_output, expected_outputs, step):
        """
        Calculate prefix accuracy for the given LLM output and expected outputs.
        """
        for i in range(self.num_samples):
            # check if prefix acc is already zero from a previous step
            if step > 0 and self.prefix_correctness_per_step[step - 1][i] == 0:
                # Copy previous step's state (already failed)
                self.prefix_correctness_per_step[step][i] = 0
                continue

            # Check if the LLM output matches the expected output for this step
            if llm_output[i] != expected_outputs[i]:
                self.prefix_correctness_per_step[step][i] = 0
            # else: it remains 1 from initialization

        # Calculate the prefix correctness for this step
        return np.mean(self.prefix_correctness_per_step[step])

    def calculate_bootstrap_ci(
        self,
        step: int,
        metrics_to_bootstrap: List[str] = ["index", "step", "prefix"],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for the specified metrics at a given step.

        Args:
            step: The step for which to calculate confidence intervals
            metrics_to_bootstrap: List of metrics to bootstrap ("index", "step", "prefix")
            n_bootstrap: Number of bootstrap samples to generate
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Dictionary with CI bounds for each requested metric
        """
        if step >= self.expected_output_len or step < 0:
            raise ValueError(
                f"Step {step} is out of range [0, {self.expected_output_len-1}]"
            )

        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        results = {}

        # Set random seed for reproducible bootstrap sampling
        np.random.seed(42)

        for metric in metrics_to_bootstrap:
            if metric == "index":
                # Bootstrap sample from index correctness
                data = self.index_correctness_per_step[step]
            elif metric == "step":
                # Bootstrap sample from step correctness
                data = self.step_correctness_per_step[step]
            elif metric == "prefix":
                # Bootstrap sample from prefix correctness
                data = self.prefix_correctness_per_step[step]
            else:
                logging.warning(
                    f"Unknown metric '{metric}' for bootstrapping. Skipping."
                )
                continue

            # Generate bootstrap samples
            bootstrap_means = []
            for _ in range(n_bootstrap):
                # Sample with replacement
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))

            # Calculate confidence intervals using percentile method
            ci_min = np.percentile(bootstrap_means, lower_percentile)
            ci_max = np.percentile(bootstrap_means, upper_percentile)

            results[metric] = (ci_min, ci_max)

        return results


class DictSumEvaluator:
    """
    Evaluates LLM output for the prefix sum calculation task.
    Calculates full correctness, index correctness list, prefix correctness list, and step correctness list.
    """

    def __init__(
        self, entry: dict, working_capacity: int = 1, is_multi_turn: bool = False
    ):  # Add working_capacity and is_multi_turn
        """
        Args:
            entry (dict): A ground truth entry from PrefixSumGenerator,
                          containing 'input_list' and 'prefix_sums_list'.
            working_capacity (int): The number of input numbers processed by the LLM at once.
            is_multi_turn (bool): Flag indicating if the evaluation is for a multi-turn scenario.
        """
        if not isinstance(entry, dict) or "output" not in entry:
            raise ValueError(
                "Entry must be a dictionary with a 'prefix_sums_list' key."
            )
        if not isinstance(working_capacity, int) or working_capacity <= 0:
            raise ValueError("working_capacity must be a positive integer.")

        self.input_list: list[int] = entry.get("values", [])
        self.true_prefix_sums_list: list[int] = entry["output"]

        self.k = len(
            self.true_prefix_sums_list
        )  # Length of the true list of prefix sums
        self.working_capacity = working_capacity
        self.is_multi_turn = is_multi_turn

        if self.k == 0:
            self.num_expected_outputs = 0
            self.ground_truth_for_W = []
        else:
            self.num_expected_outputs = math.ceil(self.k / self.working_capacity)
            self.ground_truth_for_W = [
                self.true_prefix_sums_list[
                    min(
                        i * self.working_capacity + self.working_capacity - 1,
                        self.k - 1,
                    )
                ]
                for i in range(self.num_expected_outputs)
            ]
            # Ensure ground_truth_for_W is not empty if k > 0 but num_expected_outputs became 0 (e.g. k=1, W=2, ceil(1/2)=1, but if k=0.5, W=1, ceil(0.5/1)=1)
            # This case should ideally not happen with integer k and W.
            # However, if self.k > 0 and self.num_expected_outputs is 0 due to some edge case,
            # self.ground_truth_for_W might be empty.
            # A more robust way for ground_truth_for_W:
            # If k=5, W=2, expected indices are 1, 3, 4 (0-indexed from true_prefix_sums_list)
            # corresponding to sums after [n0,n1], [n0,n1,n2,n3], [n0,n1,n2,n3,n4]

            # Recalculate ground_truth_for_W with clearer logic
            gt_for_w = []
            if self.k > 0:
                for i in range(self.num_expected_outputs):
                    # The index in the original true_prefix_sums_list
                    # corresponds to the end of the current block of W inputs.
                    idx = (i + 1) * self.working_capacity - 1
                    # Ensure the index does not exceed the bounds of true_prefix_sums_list
                    idx = min(idx, self.k - 1)
                    gt_for_w.append(self.true_prefix_sums_list[idx])
            self.ground_truth_for_W = gt_for_w

            self.init_metrics()

    def init_metrics(self):
        """
        Initializes metrics for evaluation.
        """
        self.num_evaluated = 0
        self.prefix_correctness_list = [0.0] * self.num_expected_outputs
        self.index_correctness_list = [0.0] * self.num_expected_outputs
        self.step_correctness_list = [0.0] * self.num_expected_outputs

    def update_metrics(self, prefix_instance, index_instance, step_instance):
        self.num_evaluated += 1
        for i in range(self.num_expected_outputs):
            self.prefix_correctness_list[i] += prefix_instance[i]
            self.index_correctness_list[i] += index_instance[i]
            self.step_correctness_list[i] += step_instance[i]

    def finalize_metrics(self):
        if self.num_evaluated > 0:
            self.prefix_correctness_list = [
                x / self.num_evaluated for x in self.prefix_correctness_list
            ]
            self.index_correctness_list = [
                x / self.num_evaluated for x in self.index_correctness_list
            ]
            self.step_correctness_list = [
                x / self.num_evaluated for x in self.step_correctness_list
            ]
        else:
            self.prefix_correctness_list = [0.0] * self.num_expected_outputs
            self.index_correctness_list = [0.0] * self.num_expected_outputs
            self.step_correctness_list = [0.0] * self.num_expected_outputs

    def _parse_llm_output(
        self, llm_output: List[Union[int, str]]
    ) -> list[Union[int, str]]:  # Changed type hint
        """
        Adjusts the parsed LLM output list (already parsed by PrefixSumOutputParser)
        to match the expected number of outputs based on working_capacity.
        If too long, it's truncated. If too short, it's padded with NOOP_TOKEN.
        """
        # llm_output is already a list of ints or NOOP_TOKENs from PrefixSumOutputParser

        # Pad with NOOP_TOKEN if too short, or truncate if too long
        if len(llm_output) > self.num_expected_outputs:
            # Truncate to num_expected_outputs elements if too long
            llm_output = llm_output[: self.num_expected_outputs]
        elif len(llm_output) < self.num_expected_outputs:
            # Pad with NOOP_TOKEN if too short
            # Using PrefixSumOutputParser.NOOP_TOKEN for padding
            llm_output = llm_output + ["<PAD"] * (
                self.num_expected_outputs - len(llm_output)
            )
        return llm_output

    def _calculate_expected_prefix_sum_at_position(
        self, model_previous_sum: int, current_input: int
    ) -> int:
        """
        Calculate what the prefix sum should be at a given position based on the model's previous sum.

        Args:
            model_previous_sum (int): The model's output for the previous prefix sum
            current_input (int): The current input element to add

        Returns:
            int: The expected prefix sum (model_previous_sum + current_input)
        """
        return model_previous_sum + current_input

    def evaluate(
        self, llm_output_sequence: List[Union[int, str]]
    ) -> dict:  # Changed argument name and type
        """
        Evaluates the LLM's output sequence against the true list of prefix sums,
        considering the working_capacity.

        Args:
            llm_output_sequence (List[Union[int, str]]): The parsed output sequence from the LLM
                                                        (list of ints or NOOP_TOKENs).

        Returns:
            dict: Contains 'full_correctness' (1.0 or 0.0),
                  'prefix_correctness_list' (list of floats, 1.0 or 0.0, length num_expected_outputs),
                  'index_correctness_list' (list of floats, 1.0 or 0.0, length num_expected_outputs),
                  'step_correctness_list' (list of floats, 1.0 or 0.0, length num_expected_outputs),
                  and 'parsed_llm_output' (the adjusted list of ints/NOOP_TOKENs).
        """
        # The input llm_output_sequence is already parsed by PrefixSumOutputParser in the Experiment class
        # Now, adjust its length according to num_expected_outputs
        adjusted_llm_output = self._parse_llm_output(llm_output_sequence)

        if not self.input_list and not self.true_prefix_sums_list:  # k=0 case
            # If k=0, ground_truth_for_W is [], num_expected_outputs is 0.
            # adjusted_llm_output should also be [].
            full_correctness = 1.0 if not adjusted_llm_output else 0.0
            return {
                "full_correctness": full_correctness,
                "prefix_correctness_list": [],
                "index_correctness_list": [],
                "step_correctness_list": [],
                "parsed_llm_output": adjusted_llm_output,
                "notes": "Evaluated for k=0.",
            }

        # Ensure self.num_expected_outputs is used for initializing lists
        # This handles k=0 correctly as num_expected_outputs would be 0.
        if self.num_expected_outputs == 0 and self.k > 0:
            # This can happen if W > k. For example k=3, W=5. num_expected_outputs = ceil(3/5) = 1.
            # ground_truth_for_W will have 1 element: true_prefix_sums_list[k-1].
            # This block is more of a safeguard for unexpected scenarios where num_expected_outputs might be zero
            # despite k > 0. The current logic for num_expected_outputs = math.ceil(self.k / self.working_capacity)
            # should make it >= 1 if self.k >= 1.
            # If k=0, num_expected_outputs is 0, ground_truth_for_W is []. This is handled above.
            pass  # Let the normal logic proceed, lists will be sized by num_expected_outputs.

        # Full Correctness: Compare adjusted_llm_output with self.ground_truth_for_W
        full_correctness = 0.0
        # Check if all elements are integers before direct comparison
        # and if the lists are identical.
        # NOOP_TOKENs will prevent full correctness unless ground_truth_for_W also expects them (which it doesn't).
        # So, if adjusted_llm_output contains any NOOP_TOKEN, it cannot be fully correct.
        if (
            all(isinstance(x, int) for x in adjusted_llm_output)
            and adjusted_llm_output == self.ground_truth_for_W
        ):
            full_correctness = 1.0

        prefix_correctness_list = [0.0] * self.num_expected_outputs
        index_correctness_list = [0.0] * self.num_expected_outputs
        step_correctness_list = [0.0] * self.num_expected_outputs

        # Model's previous cumulative sum (from its own output sequence)
        # For the first step (i=0), this is 0.
        # For subsequent steps, it's the model's output for the previous block of W.
        model_previous_block_cumulative_sum = 0

        for i in range(self.num_expected_outputs):
            # Index Correctness for the i-th output (corresponds to a block of W inputs)
            # Compares the i-th element of adjusted_llm_output with the i-th element of ground_truth_for_W.
            if (
                i < len(adjusted_llm_output)
                and i < len(self.ground_truth_for_W)
                and adjusted_llm_output[i] == self.ground_truth_for_W[i]
                and isinstance(
                    adjusted_llm_output[i], int
                )  # Ensure it's not NOOP_TOKEN
            ):
                index_correctness_list[i] = 1.0

            # Prefix Correctness for the prefix ending at the i-th output
            # Checks if the first (i+1) elements of adjusted_llm_output match ground_truth_for_W.
            current_prefix_len = i + 1
            # Ensure all elements in the sub-list are integers for a valid prefix match.
            if (
                all(
                    isinstance(x, int) for x in adjusted_llm_output[:current_prefix_len]
                )
                and adjusted_llm_output[:current_prefix_len]
                == self.ground_truth_for_W[:current_prefix_len]
            ):
                prefix_correctness_list[i] = 1.0

            # Step Correctness for the i-th output block
            # This is the sum of inputs in the current block of W, added to the model's *previous* cumulative sum.
            # Or, if multi-turn, it's added to the *true* cumulative sum of the previous block if the model is stateless per turn.
            # The current implementation in exp.py passes the full sequence for single-turn, and turn-by-turn for multi-turn.
            # For multi-turn, the `model_previous_block_cumulative_sum` needs to be the *true* sum before the current turn's inputs.
            # For single-turn, it's the model's *own* previous output.

            # Determine the sum of the current block of W input numbers.
            start_input_idx = i * self.working_capacity
            end_input_idx = min((i + 1) * self.working_capacity, self.k)
            current_block_input_sum = sum(
                self.input_list[start_input_idx:end_input_idx]
            )

            # Determine the base sum to which current_block_input_sum should be added.
            base_sum_for_step = 0
            if self.is_multi_turn:
                # In multi-turn, the model is given its own previous output as context.
                # Step correctness checks if the model correctly added the current block
                # to its own previous output, regardless of whether that previous output was correct.
                if i == 0:
                    # First step: check if model output equals sum of first block
                    expected_step_sum = current_block_input_sum
                    if (
                        i < len(adjusted_llm_output)
                        and isinstance(adjusted_llm_output[i], int)
                        and adjusted_llm_output[i] == expected_step_sum
                    ):
                        step_correctness_list[i] = 1.0
                else:
                    # Later steps: check if model correctly added current block to its previous output
                    if (
                        i < len(adjusted_llm_output)
                        and isinstance(adjusted_llm_output[i], int)
                        and isinstance(
                            model_previous_block_cumulative_sum, int
                        )  # Previous output was valid
                        and adjusted_llm_output[i]
                        == model_previous_block_cumulative_sum + current_block_input_sum
                    ):
                        step_correctness_list[i] = 1.0
            else:  # Single-turn
                # In single-turn, step correctness checks if the model correctly added the current block
                # to its own previous output, regardless of whether that previous output was correct.
                if i == 0:
                    # First step: check if model output equals sum of first block
                    expected_step_sum = current_block_input_sum
                    if (
                        i < len(adjusted_llm_output)
                        and isinstance(adjusted_llm_output[i], int)
                        and adjusted_llm_output[i] == expected_step_sum
                    ):
                        step_correctness_list[i] = 1.0
                else:
                    # Later steps: check if model correctly added current block to its previous output
                    if (
                        i < len(adjusted_llm_output)
                        and isinstance(adjusted_llm_output[i], int)
                        and isinstance(
                            model_previous_block_cumulative_sum, int
                        )  # Previous output was valid
                        and adjusted_llm_output[i]
                        == model_previous_block_cumulative_sum + current_block_input_sum
                    ):
                        step_correctness_list[i] = 1.0

            # Update model_previous_block_cumulative_sum for the next iteration (for single-turn context)
            # If the model's output was not an int (e.g. NOOP_TOKEN), it means the chain of correct sums is broken.
            # For step correctness, we often want to see if the *next* step is correct *given the model made a mistake*.
            # However, the definition of `model_previous_block_cumulative_sum` here is the model's *actual* last output.
            if i < len(adjusted_llm_output) and isinstance(adjusted_llm_output[i], int):
                model_previous_block_cumulative_sum = adjusted_llm_output[i]
            else:
                # If the model's output for this block is not an integer (e.g., NOOP), then for the next step in single-turn,
                # its chain is broken. What value should it take?
                # If we use ground_truth_for_W[i], then we are not penalizing error propagation for step correctness.
                # If we use a value that ensures failure (like float('nan')), then all subsequent steps are 0.
                # Let's use the model's output if it's an int, otherwise, for the purpose of the *next* step's base,
                # we can assume the chain is broken and further step calculations are not meaningful in the same way.
                # However, the current `step_correctness_list[i]` is for the *current* step.
                # The `model_previous_block_cumulative_sum` is for the *next* step.

                if i < len(
                    adjusted_llm_output
                ):  # Re-check boundary, though loop ensures it for current i
                    if isinstance(adjusted_llm_output[i], int):
                        model_previous_block_cumulative_sum = adjusted_llm_output[i]
                    else:
                        # If model output is not an int (e.g. NOOP), then for the next step in single-turn,
                        # the base sum is effectively unknown or incorrect from the model's perspective.
                        # We can set it to a value that will likely cause future step checks to fail if they rely on it,
                        # or use the true value if we want to evaluate future steps independently of this error.
                        # For now, let's use a value that signifies an error state, e.g., float('inf') or a very distinct number.
                        # This ensures that if `base_sum_for_step` uses this, `expected_step_sum` will be off.
                        model_previous_block_cumulative_sum = float(
                            "inf"
                        )  # Mark as broken chain for single-turn
                else:  # Should not happen due to loop range and padding
                    model_previous_block_cumulative_sum = float("inf")

        return {
            "full_correctness": full_correctness,
            "prefix_correctness_list": prefix_correctness_list,
            "index_correctness_list": index_correctness_list,
            "step_correctness_list": step_correctness_list,
            "parsed_llm_output": adjusted_llm_output,
            "effective_ground_truth": self.ground_truth_for_W,
        }
        
    def log_to_wandb(self, wandb_logger, step_arr, index_arr, prefix_arr):
        if not wandb_logger:
            return
        
        for step in range(self.num_expected_outputs):
            metrics_to_log = {
                f"prefix_accuracy": prefix_arr[step],
                f"index_accuracy": index_arr[step],
                f"step_accuracy": step_arr[step],
            }
            wandb_logger.log(metrics_to_log, step=step)
            logging.info(
                f"Step {step} - "
                f"Prefix Accuracy: {prefix_arr[step]}, "
                f"Index Accuracy: {index_arr[step]}, "
                f"Step Accuracy: {step_arr[step]}"
            )

def test_evaluator_comparison():
    """
    Test function to compare the NewDictSumEvaluator and DictSumEvaluator
    using dummy data to ensure they produce equivalent results.
    Also demonstrates the new JSON logging functionality.
    """
    print("Testing evaluator comparison...")

    # Create dummy data for testing - all sequences have length 6 for working_capacity=2 testing
    dummy_entries = [
        {
            "input": ["word1", "word2", "word3", "word4", "word5", "word6"],
            "values": [10, 20, 30, 40, 50, 60],
            "output": [10, 30, 60, 100, 150, 210],  # prefix sums
        },
        {
            "input": ["word7", "word8", "word9", "word10", "word11", "word12"],
            "values": [15, 25, 35, 45, 55, 65],
            "output": [15, 40, 75, 120, 175, 240],  # prefix sums
        },
        {
            "input": ["word13", "word14", "word15", "word16", "word17", "word18"],
            "values": [5, 10, 15, 20, 25, 30],
            "output": [5, 15, 30, 50, 75, 105],  # prefix sums
        },
    ]

    # Test both interpretations of LLM outputs with working_capacity=2
    test_cases = [
        {
            "name": "Cumulative Outputs (Traditional) - WC=2",
            "description": "LLM outputs represent cumulative prefix sums at every 2nd step",
            "outputs": [
                [
                    30,
                    40,
                    15,
                ],  # Step 0: prefix sum after 2 elements: [10+20, 15+25, 5+10]
                [
                    100,
                    120,
                    50,
                ],  # Step 1: prefix sum after 4 elements: [30+30+40, 40+35+45, 15+15+20]
                [
                    210,
                    240,
                    105,
                ],  # Step 2: prefix sum after 6 elements: [100+50+60, 120+55+65, 50+25+30]
            ],
        },
        {
            "name": "Incremental Outputs (For NewDictSumEvaluator Step Correctness) - WC=2",
            "description": "LLM outputs represent incremental values to add (sum of 2 elements)",
            "outputs": [
                [30, 40, 15],  # Step 0: sum of first 2 elements: [10+20, 15+25, 5+10]
                [70, 80, 35],  # Step 1: sum of next 2 elements: [30+40, 35+45, 15+20]
                [110, 120, 55],  # Step 2: sum of last 2 elements: [50+60, 55+65, 25+30]
            ],
        },
    ]

    working_capacity = 2

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST CASE: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"{'='*60}")

        dummy_llm_outputs = test_case["outputs"]

        # Initialize NewDictSumEvaluator (evaluates all instances step by step)
        new_evaluator = NewDictSumEvaluator(
            dummy_entries,
            working_capacity=working_capacity,
        )

        # Evaluate each step with the new evaluator
        print("\n--- NewDictSumEvaluator Results ---")
        for step in range(len(dummy_llm_outputs)):
            if step < new_evaluator.expected_output_len:
                llm_output = dummy_llm_outputs[step].copy()
                # Pad with NoOp tokens if needed to match num_samples
                while len(llm_output) < new_evaluator.num_samples:
                    llm_output.append(DictSumOutputParser.NOOP_TOKEN)

                # Convert to proper format with answer tags for testing
                formatted_output = [f"<answer>{val}</answer>" for val in llm_output]
                raw_output = f"Step {step} raw: {' | '.join(formatted_output)}"

                new_evaluator.evaluate_step(
                    formatted_output,
                    step,
                    num_tokens_generated=50 + step * 10,  # Mock token count
                )
                print(
                    f"Step {step}: Index Acc = {new_evaluator.metrics.index_correctness_array[step]:.2f}, "
                    f"Prefix Acc = {new_evaluator.metrics.prefix_correctness_array[step]:.2f}, "
                    f"Step Acc = {new_evaluator.metrics.step_correctness_array[step]:.2f}"
                )

                # Show what the evaluator expects vs gets for this step
                print(f"  Expected outputs: {new_evaluator.entries['output'][step]}")
                print(f"  LLM outputs: {llm_output}")
                if step < len(new_evaluator.entries["values"]):
                    print(
                        f"  Values for this step: {new_evaluator.entries['values'][step]}"
                    )
                print()

        # Test DictSumEvaluator (evaluates one instance at a time for all steps)
        print("\n--- DictSumEvaluator Results (per instance) ---")
        old_evaluator_results = []

        for i, entry in enumerate(dummy_entries):
            old_evaluator = DictSumEvaluator(entry, working_capacity=working_capacity)

            # Create LLM output sequence for this instance based on working_capacity
            instance_llm_output = []
            num_expected_outputs = old_evaluator.num_expected_outputs

            for step in range(num_expected_outputs):
                if step < len(dummy_llm_outputs) and i < len(dummy_llm_outputs[step]):
                    instance_llm_output.append(dummy_llm_outputs[step][i])
                else:
                    instance_llm_output.append(DictSumOutputParser.NOOP_TOKEN)

            result = old_evaluator.evaluate(instance_llm_output)
            old_evaluator_results.append(result)

        # Show aggregated comparison
        print(f"\n--- Aggregated Results Comparison ---")
        print("NewDictSumEvaluator:")
        print(f"  Index Correctness: {new_evaluator.metrics.index_correctness_array}")
        print(f"  Prefix Correctness: {new_evaluator.metrics.prefix_correctness_array}")
        print(f"  Step Correctness: {new_evaluator.metrics.step_correctness_array}")

        # Aggregate old evaluator results
        if old_evaluator_results:
            max_steps = max(
                len(result["index_correctness_list"])
                for result in old_evaluator_results
            )
            aggregated_index = []
            aggregated_step = []

            for step in range(max_steps):
                index_values = [
                    (
                        result["index_correctness_list"][step]
                        if step < len(result["index_correctness_list"])
                        else 0.0
                    )
                    for result in old_evaluator_results
                ]
                step_values = [
                    (
                        result["step_correctness_list"][step]
                        if step < len(result["step_correctness_list"])
                        else 0.0
                    )
                    for result in old_evaluator_results
                ]

                aggregated_index.append(sum(index_values) / len(index_values))
                aggregated_step.append(sum(step_values) / len(step_values))

            print("\nDictSumEvaluator (aggregated):")
            print(f"  Index Correctness: {aggregated_index}")
            print(f"  Step Correctness: {aggregated_step}")

    print(f"\n{'='*60}")
    print("SUMMARY:")
    print("- NewDictSumEvaluator evaluates ALL instances for each step")
    print("- DictSumEvaluator evaluates ONE instance for all steps at a time")
    print("- Step correctness has different meanings in each evaluator")
    print("- Index correctness should be equivalent when aggregated")
    print("- JSON logs now capture detailed per-step, per-instance information")
    print(f"{'='*60}")


def example_usage_with_json_logging():
    """
    Example demonstrating how to use the updated NewDictSumEvaluator with JSON logging.
    """
    print("Example: Using NewDictSumEvaluator with JSON logging")

    # Create sample data
    dict_sum_util = DictSumUtil(
        num_pairs=10,
        min_value=1,
        max_value=20,
        horizon_length=5,
        num_instances=3,
        seed=42,
    )

    # Initialize evaluator
    evaluator = NewDictSumEvaluator(
        entries=dict_sum_util.entries,
        working_capacity=2,
    )

    # Simulate LLM outputs for each step
    simulated_outputs = [
        [30, 45, 15],  # Step 0 outputs for 3 instances
        [75, 90, 40],  # Step 1 outputs for 3 instances
        [120, 135, 70],  # Step 2 outputs for 3 instances
    ]

    print(f"Evaluating {len(simulated_outputs)} steps...")

    for step, outputs in enumerate(simulated_outputs):
        if step < evaluator.expected_output_len:
            # Format outputs with answer tags for proper parsing
            formatted_outputs = [f"<answer>{val}</answer>" for val in outputs]
            raw_output = f"LLM raw response for step {step}: {outputs}"

            evaluator.evaluate_step(
                llm_output=formatted_outputs,
                step=step,
                num_tokens_generated=25 + step * 5,
            )
            print(f"Completed step {step}")

    # Evaluation complete
    print(
        f"Evaluation complete! {evaluator.steps_evaluated} steps evaluated with {evaluator.num_samples} samples."
    )


if __name__ == "__main__":
    test_evaluator_comparison()
    print("\n" + "=" * 80 + "\n")
    example_usage_with_json_logging()
