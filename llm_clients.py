# llm_clients.py
import concurrent.futures
import json
import random
import requests
import time
import torch
import openai  # For OpenRouter
from tqdm import tqdm
from vllm import (
    LLM as VLLM_Engine,
    SamplingParams as VLLM_SamplingParams,
)  # Rename to avoid class name clash
import logging  # Or use standard logging
from typing import Callable, List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import logging as std_logging
import functools

from src._config.model_config import ModelConfig
from src._config.openrouter_config import OpenRouterConfig
import re

std_logging.getLogger("openai").setLevel(std_logging.WARNING)
std_logging.getLogger("httpx").setLevel(std_logging.WARNING)


def timing_decorator(func):
    """
    Decorator to measure and log the execution time of a function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Try to get a meaningful name for logging
        if hasattr(func, "__self__"):
            class_name = func.__self__.__class__.__name__
            func_name = f"{class_name}.{func.__name__}"
        else:
            func_name = func.__name__

        logging.info(f"⏱️  {func_name} completed in {execution_time:.4f} seconds")
        return result

    return wrapper


# Helper function to load the actual client (internal to this module)
def _load_actual_client(
    provider: str,
    model_name: str,
    model_config: Dict[str, Any],
    openrouter_specific_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Union[VLLM_Engine, openai.OpenAI]], Optional[int]]:
    """
    Loads the appropriate LLM client instance based on the provider.
    This is a helper function for the UnifiedLLM class.
    """
    if provider == "vllm":
        logging.info(f"Attempting to load vLLM model: {model_name}...")
        tensor_parallel_size = model_config.get("tensor_parallel_size")
        max_model_len = model_config.get("max_model_len")
        gpu_memory_utilization = model_config.get("gpu_memory_utilization", 0.90)
        dtype = model_config.get("dtype", "auto")
        enforce_eager = model_config.get(
            "enforce_eager", False
        )  # Example of another vLLM param

        actual_tp_size = 1
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if available_gpus == 0:
                logging.warning(
                    "No NVIDIA GPUs detected. vLLM may run on CPU if supported, or fail."
                )
                # Forcing tp_size to 1, but CPU execution for vLLM is generally not recommended/performant.
            elif tensor_parallel_size is None or tensor_parallel_size <= 0:
                actual_tp_size = available_gpus
                logging.info(
                    f"Using all available GPUs ({actual_tp_size}) for tensor parallelism."
                )
            elif tensor_parallel_size > available_gpus:
                logging.warning(
                    f"Requested tensor_parallel_size ({tensor_parallel_size}) "
                    f"exceeds available GPUs ({available_gpus}). Using {available_gpus} GPUs."
                )
                actual_tp_size = available_gpus
            else:
                actual_tp_size = tensor_parallel_size
                logging.info(
                    f"Using specified tensor_parallel_size ({actual_tp_size})."
                )
        else:
            logging.warning(
                "No CUDA available. vLLM will likely fail or run on CPU (if supported by version/model)."
            )
            actual_tp_size = 1  # Default, but vLLM's behavior on CPU can be limited

        try:
            client = VLLM_Engine(
                model=model_name,
                tensor_parallel_size=actual_tp_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                enforce_eager=enforce_eager,
                # Add other vLLM specific params from model_config if needed
            )
            logging.info(f"vLLM model loaded successfully: {model_name}")
            return client, actual_tp_size
        except Exception as e:
            logging.error(
                f"Error loading vLLM model ({model_name}): {e}", exc_info=True
            )
            logging.error(
                "Please ensure vLLM is installed correctly, the model is accessible, "
                "and GPU resources are available and compatible."
            )
            return None, None

    elif provider == "openrouter":
        if not openrouter_specific_config:
            logging.error("OpenRouter configuration is missing.")
            return None, None

        api_key = openrouter_specific_config.get("api_key")
        api_base = openrouter_specific_config.get("api_base")
        timeout = openrouter_specific_config.get("timeout", 60.0)  # Default timeout 60s

        logging.info(
            f"Initializing OpenAI client for OpenRouter with model: {model_name} and API base: {api_base}"
        )
        if not api_key:
            logging.error(
                "API key is required for OpenRouter provider. "
                "Check 'openrouter_config.api_key' in your YAML or the OPENROUTER_API_KEY env var."
            )
            return None, None
        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=timeout,
            )
            # Optional: Test connection by trying to list models.
            # try:
            #     client.models.list()
            #     logging.info(f"Successfully connected to OpenRouter API base: {api_base}")
            # except openai.AuthenticationError:
            #     logging.error(f"OpenRouter authentication failed for API base {api_base}. Check your API key.")
            #     return None, None
            # except Exception as e:
            #     logging.warning(f"Could not list models from OpenRouter (API base: {api_base}), but proceeding: {e}")

            logging.info("OpenAI client initialized successfully for OpenRouter.")
            return client, None  # tensor_parallel_size is not applicable
        except Exception as e:
            logging.error(
                f"Error initializing OpenAI client for OpenRouter: {e}", exc_info=True
            )
            return None, None
    else:
        logging.error(
            f"Unknown model provider: {provider}. Must be 'vllm' or 'openrouter'."
        )
        return None, None


class UnifiedLLM:
    """
    A unified wrapper class for interacting with different LLM providers (vLLM, OpenRouter).
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        model_config: ModelConfig,
        openrouter_config: OpenRouterConfig,
    ):
        self.provider = provider
        self.model_name = model_name
        self.model_config = model_config

        self.sliding_window_size = model_config.sliding_window_size
        self.fill_history = model_config.fill_history
        self.incorrect_probability = model_config.incorrect_probability
        self.early_stopping = model_config.early_stopping
        self.early_stopping_threshold = model_config.early_stopping_threshold
        self.target_step = model_config.target_step

        self.openrouter_config = openrouter_config
        self.client, self.actual_tp_size = _load_actual_client(
            provider, model_name, model_config, openrouter_config
        )

        if self.client is None:
            raise ValueError(
                f"Failed to load LLM client for provider {provider} and model {model_name}."
            )

        logging.info(
            f"UnifiedLLM initialized for provider: {self.provider}, model: {self.model_name}"
        )
        self.tokens_generated_per_step: List[int] = []

    def _remove_thinking_traces(self, message: str) -> str:
        """
        Remove thinking traces and extract clean content from LLM responses.

        First tries to remove <think>...</think> tags while preserving other content.
        If no thinking tags found, looks for <answer>...</answer> tags.
        If neither found, returns empty string.
        """
        thinking_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

        # Try to remove thinking traces first
        if thinking_pattern.search(message):
            cleaned_message = thinking_pattern.sub("", message).strip()
            logging.debug("Removed thinking traces from the message.")
        else:
            cleaned_message = message.strip()
            # Look for answer tags if no thinking traces found
            answer_match = answer_pattern.search(message)
            if answer_match:
                content = answer_match.group(1).strip()
                if self.model_config.cot:
                    cleaned_message = message  # give cot trace to history
                else:
                    cleaned_message = "<answer>" + content + "</answer>"
                logging.debug("Found CoT answer in the message")
            else:
                cleaned_message = "No Answer"
                logging.warning(
                    f"No thinking traces or CoT answer found in the message: {message}..."
                )

        logging.debug(f"Cleaning during history processing: {cleaned_message}")
        return cleaned_message

    def _execute_single_chat_api_turn(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        sleep_time: Optional[float],
        request_idx: Optional[
            Union[int, str]
        ] = None,  # Union for flexibility (e.g. "instance_idx-turn_idx")
    ) -> str:
        """
        Executes a single turn of chat with the LLM API using direct requests.
        """
        log_prefix = f"[Request {request_idx}] " if request_idx is not None else ""
        response_content = ""

        # Prepare the request payload
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }

        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens
            if max_tokens > 32000:
                # then we are reasoning
                payload["reasoning"] = {
                    "max_tokens": 32000,
                    "exclude": True,  # Exclude reasoning tokens from completion
                }
        if stop is not None:
            payload["stop"] = stop

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for bad status codes
            
            logging.info(f"{log_prefix}OpenRouter chat API call successful.")
            # logging.info(response)

            response_json = response.json()
            response_content = response_json["choices"][0]["message"]["content"].strip()

            # Track completion tokens from OpenRouter usage data
            if (
                "usage" in response_json
                and "completion_tokens" in response_json["usage"]
            ):
                completion_tokens = response_json["usage"]["completion_tokens"]
                self.tokens_generated_per_step.append(completion_tokens)
                logging.debug(f"{log_prefix}Completion tokens: {completion_tokens}")

            max_retries = 3
            retry_count = 0
            while not response_content and retry_count < max_retries:
                logging.warning(
                    f"{log_prefix}Received empty response content from OpenRouter chat API. Retrying ({retry_count + 1}/{max_retries})"
                )
                retry_count += 1
                time.sleep(0.5 * retry_count)  # Small progressive backoff

                try:
                    response = requests.post(
                        url, headers=headers, data=json.dumps(payload)
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    response_content = response_json["choices"][0]["message"][
                        "content"
                    ].strip()

                    # Track completion tokens from OpenRouter usage data in retry
                    if (
                        "usage" in response_json
                        and "completion_tokens" in response_json["usage"]
                    ):
                        completion_tokens = response_json["usage"]["completion_tokens"]
                        self.tokens_generated_per_step.append(completion_tokens)
                        logging.debug(
                            f"{log_prefix}Retry completion tokens: {completion_tokens}"
                        )

                except Exception as e:
                    logging.error(
                        f"{log_prefix}Error during retry {retry_count} of OpenRouter chat API call: {e}",
                        exc_info=True,
                    )
                    response_content = ""

            if not response_content:
                logging.warning(
                    f"{log_prefix}Received empty response content from OpenRouter chat API after retries."
                )
                return ""

            if sleep_time and sleep_time > 0.0:
                time.sleep(sleep_time)
            return response_content

        except Exception as e:
            logging.error(
                f"{log_prefix}Error during OpenRouter chat API call: {e}", exc_info=True
            )
            max_retries = 3
            retry_count = 0
            while not response_content and retry_count < max_retries:
                logging.warning(
                    f"{log_prefix}Rate limit retry ({retry_count + 1}/{max_retries})"
                )
                retry_count += 1
                time.sleep(3 * retry_count)  # Progressive backoff for rate limits
                try:
                    response = requests.post(
                        url, headers=headers, data=json.dumps(payload)
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    response_content = response_json["choices"][0]["message"][
                        "content"
                    ].strip()

                    # Track completion tokens from OpenRouter usage data in rate limit retry
                    if (
                        "usage" in response_json
                        and "completion_tokens" in response_json["usage"]
                    ):
                        completion_tokens = response_json["usage"][
                            "completion_tokens"
                        ]
                        self.tokens_generated_per_step.append(completion_tokens)
                        logging.debug(
                            f"{log_prefix}Rate limit retry completion tokens: {completion_tokens}"
                        )
                except Exception as retry_e:
                    logging.error(
                        f"{log_prefix}Error during rate limit retry {retry_count}: {retry_e}",
                        exc_info=True,
                    )
                    response_content = ""

            if not response_content:
                logging.warning(
                    f"{log_prefix}Received empty response content from OpenRouter chat API after rate limit retries."
                )
                return ""

            if sleep_time and sleep_time > 0.0:
                time.sleep(sleep_time)
            return response_content
        # else:
        #     logging.error(
        #         f"{log_prefix}OpenRouter API HTTP error in chat turn: Status {e.response.status_code}, Response: {e.response.text}"
        #     )
        # except requests.exceptions.ConnectionError as e:
        #     logging.error(
        #         f"{log_prefix}OpenRouter API connection error in chat turn: {e}"
        #     )
        # except requests.exceptions.RequestException as e:
        #     logging.error(f"{log_prefix}OpenRouter API request error in chat turn: {e}")
        # except Exception as e:
        #     logging.error(
        #         f"{log_prefix}Error during OpenRouter chat API call: {e}", exc_info=True
        #     )
        return ""

    def _execute_single_chat_api_turn_with_tokens(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        sleep_time: Optional[float],
        request_idx: Optional[
            Union[int, str]
        ] = None,  # Union for flexibility (e.g. "instance_idx-turn_idx")
    ) -> tuple[str, int]:
        """
        Executes a single turn of chat with the LLM API using direct requests.
        Returns both the response content and the number of completion tokens.
        """
        log_prefix = f"[Request {request_idx}] " if request_idx is not None else ""
        response_content = ""
        completion_tokens = 0

        # Prepare the request payload
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }

        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens
            if max_tokens > 32000:
                # then we are reasoning
                payload["reasoning"] = {"effort": "high"}
        if stop is not None:
            payload["stop"] = stop

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for bad status codes

            response_json = response.json()
            response_content = response_json["choices"][0]["message"]["content"].strip()

            # Track completion tokens from OpenRouter usage data
            if (
                "usage" in response_json
                and "completion_tokens" in response_json["usage"]
            ):
                completion_tokens = response_json["usage"]["completion_tokens"]
                logging.debug(f"{log_prefix}Completion tokens: {completion_tokens}")

            reasoning_content = response_json["choices"][0]["message"].get(
                "reasoning", None
            )
            logging.debug(f"{log_prefix}Reasoning content: {reasoning_content}")

            max_retries = 3
            retry_count = 0
            while not response_content and retry_count < max_retries:
                logging.warning(
                    f"{log_prefix}Received empty response content from OpenRouter chat API. Retrying ({retry_count + 1}/{max_retries})"
                )
                retry_count += 1
                time.sleep(0.5 * retry_count)  # Small progressive backoff

                try:
                    response = requests.post(
                        url, headers=headers, data=json.dumps(payload)
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    response_content = response_json["choices"][0]["message"][
                        "content"
                    ].strip()

                    # Track completion tokens from retry
                    if (
                        "usage" in response_json
                        and "completion_tokens" in response_json["usage"]
                    ):
                        completion_tokens = response_json["usage"]["completion_tokens"]
                        logging.debug(
                            f"{log_prefix}Retry completion tokens: {completion_tokens}"
                        )

                except Exception as e:
                    logging.warning(f"{log_prefix}Retry {retry_count} failed: {e}")

            if not response_content:
                logging.warning(
                    f"{log_prefix}Received empty response content from OpenRouter chat API after retries."
                )
                return "", 0

            if sleep_time and sleep_time > 0.0:
                time.sleep(sleep_time)
            return response_content, completion_tokens

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit error
                logging.warning(
                    f"{log_prefix}OpenRouter rate limit exceeded in chat turn: {e}"
                )

                max_retries = 3
                retry_count = 0
                while not response_content and retry_count < max_retries:
                    wait_time = (
                        2**retry_count
                    ) * 1.0  # Progressive backoff for rate limits
                    time.sleep(wait_time)
                    try:
                        response = requests.post(
                            url, headers=headers, data=json.dumps(payload)
                        )
                        response.raise_for_status()
                        response_json = response.json()
                        response_content = response_json["choices"][0]["message"][
                            "content"
                        ].strip()

                        # Track completion tokens from rate limit retry
                        if (
                            "usage" in response_json
                            and "completion_tokens" in response_json["usage"]
                        ):
                            completion_tokens = response_json["usage"][
                                "completion_tokens"
                            ]
                            logging.debug(
                                f"{log_prefix}Rate limit retry completion tokens: {completion_tokens}"
                            )
                    except Exception as retry_e:
                        logging.warning(
                            f"{log_prefix}Rate limit retry {retry_count + 1} failed: {retry_e}"
                        )
                    retry_count += 1

                if not response_content:
                    logging.warning(
                        f"{log_prefix}Received empty response content from OpenRouter chat API after rate limit retries."
                    )
                    return "", 0

                if sleep_time and sleep_time > 0.0:
                    time.sleep(sleep_time)
                return response_content, completion_tokens
            else:
                logging.error(
                    f"{log_prefix}OpenRouter API HTTP error in chat turn: Status {e.response.status_code}, Response: {e.response.text}"
                )
        except requests.exceptions.ConnectionError as e:
            logging.error(
                f"{log_prefix}OpenRouter API connection error in chat turn: {e}"
            )
        except requests.exceptions.RequestException as e:
            logging.error(f"{log_prefix}OpenRouter API request error in chat turn: {e}")
        except Exception as e:
            logging.error(
                f"{log_prefix}Error during OpenRouter chat API call: {e}", exc_info=True
            )
        return "", 0

    def _process_single_instance_vllm_chat(
        self,
        instance_idx: int,
        instance_prompts: List[str],
        start_template: str,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        enable_thinking: bool = False,
    ) -> str:
        """
        Processes a single instance using vLLM's built-in chat method.
        """
        current_history: List[Dict[str, str]] = []
        current_outputs: List[str] = []

        if start_template:  # Only add if start_template is non-empty
            current_history.append({"role": "system", "content": start_template})
            logging.debug(
                f"[Instance {instance_idx}] Starting chat history with start template: {start_template[:50]}"
            )
        else:
            logging.debug(
                f"[Instance {instance_idx}] Starting chat history without a start template."
            )

        effective_max_tokens_per_turn = None
        if max_tokens is not None:
            if instance_prompts:  # Ensure there are prompts to divide by
                if enable_thinking:
                    effective_max_tokens_per_turn = max_tokens
                else:
                    effective_max_tokens_per_turn = max_tokens // len(instance_prompts)
            else:
                pass

            if (
                effective_max_tokens_per_turn is not None
                and effective_max_tokens_per_turn <= 0
            ):
                logging.warning(
                    f"[Instance {instance_idx}] Calculated effective_max_tokens per turn is non-positive ({effective_max_tokens_per_turn}) from total max_tokens ({max_tokens}) and {len(instance_prompts)} turns. Setting to default 128 for subsequent turns if any."
                )
                effective_max_tokens_per_turn = 128
        else:
            logging.debug(
                f"[Instance {instance_idx}] max_tokens is None for the entire chat. vLLM will use its default for each turn."
            )

        logging.debug(
            f"[Instance {instance_idx}] Using effective max_tokens per turn: {effective_max_tokens_per_turn}"
        )

        # Process each turn
        for turn_idx, user_prompt in enumerate(instance_prompts):
            user_prompt_processed = user_prompt.strip()

            current_history.append({"role": "user", "content": user_prompt_processed})
            logging.debug(
                f"[Instance {instance_idx}, Turn {turn_idx}] Processing user prompt: '{user_prompt_processed[:50]}'"
            )

            # Create sampling params for this turn
            vllm_sampling_params = VLLM_SamplingParams(
                temperature=temperature,
                top_p=top_p,
                min_tokens=2,
                max_tokens=effective_max_tokens_per_turn,
                stop=stop if stop else [],
            )

            try:
                # Use vLLM's chat method for single conversation
                request_outputs = self.client.chat(
                    messages=[
                        current_history
                    ],  # vLLM chat expects list of conversations
                    sampling_params=vllm_sampling_params,
                    use_tqdm=True,  # Disable tqdm as requested
                    chat_template_kwargs={"enable_thinking": enable_thinking},
                )

                if request_outputs and request_outputs[0].outputs:
                    assistant_response = request_outputs[0].outputs[0].text.strip()
                else:
                    assistant_response = ""

                logging.debug(
                    f"[Instance {instance_idx}, Turn {turn_idx}] Received assistant response: '{assistant_response[:50]}'"
                )

                if not assistant_response:
                    logging.warning(
                        f"[Instance {instance_idx}, Turn {turn_idx}] vLLM response failed or was empty for prompt: '{user_prompt_processed[:100]}'"
                    )
                    current_history.append(
                        {"role": "assistant", "content": "Please Continue..."}
                    )  # last resort to avoid empty history
                    current_outputs.append("NaN")
                else:
                    history_content = self._remove_thinking_traces(assistant_response)
                    current_history.append(
                        {"role": "assistant", "content": history_content}
                    )
                    if history_content == "No Answer":
                        current_outputs.append("NaN")
                    else:
                        current_outputs.append(assistant_response)

            except Exception as e:
                logging.error(
                    f"[Instance {instance_idx}, Turn {turn_idx}] Error during vLLM chat generation: {e}",
                    exc_info=True,
                )
                current_outputs.append("NaN")

        if not current_outputs:
            # This can happen if instance_prompts was empty
            if not instance_prompts and start_template:
                logging.warning(
                    f"[Instance {instance_idx}] No user_prompts provided for this instance, so no chat turns executed beyond potential implicit use of start_template (if any). Returning empty string."
                )
            elif not instance_prompts and not start_template:
                logging.warning(
                    f"[Instance {instance_idx}] No start_template and no user_prompts provided. Returning empty string."
                )
            else:  # instance_prompts was not empty, but all turns failed
                logging.warning(
                    f"[Instance {instance_idx}] No outputs generated for this instance (all turns might have failed)."
                )
            return ""

        return "|".join(current_outputs)

    def _execute_openrouter_tasks_parallel(
        self,
        tasks_kwargs_list: List[Dict[str, Any]],
        target_callable: Callable[
            ..., str
        ],  # Made it more specific that callable returns str
        max_workers_input: Optional[int],
        progress_description: str,
    ) -> List[str]:
        """
        Private helper to execute multiple tasks in parallel using ThreadPoolExecutor,
        specifically for OpenRouter calls.
        """
        num_tasks = len(tasks_kwargs_list)
        if not num_tasks:
            return []

        outputs = [""] * num_tasks  # Pre-allocate for ordered results

        effective_max_workers = 1
        if max_workers_input is not None:
            if max_workers_input > 0:
                effective_max_workers = min(max_workers_input, num_tasks)
            else:
                logging.warning(
                    f"max_workers was {max_workers_input}, which is <= 0. Using 1 worker for {progress_description}."
                )
        else:  # max_workers_input is None (use default)
            effective_max_workers = min(
                10, num_tasks
            )  # Default: 10, or fewer if fewer tasks

        # Ensure effective_max_workers is at least 1 if there are tasks
        if num_tasks > 0 and effective_max_workers <= 0:  # Check for <=0 to be safe
            effective_max_workers = 1

        logging.info(
            f"Using {effective_max_workers} worker thread(s) for: {progress_description}"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=effective_max_workers
        ) as executor:
            future_to_index = {
                executor.submit(target_callable, **task_kwargs): index
                for index, task_kwargs in enumerate(tasks_kwargs_list)
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_index),
                total=num_tasks,
                desc=progress_description,
            ):
                index = future_to_index[future]
                try:
                    result_str = future.result()
                    outputs[index] = result_str
                except Exception as e:
                    # Identify the task for logging. Assumes 'request_idx' or 'instance_idx' might be in kwargs.
                    task_identifier_val = tasks_kwargs_list[index].get(
                        "request_idx",
                        tasks_kwargs_list[index].get(
                            "instance_idx", f"task_index_{index}"
                        ),
                    )
                    logging.error(
                        f"Parallel task (ID: {task_identifier_val}) for '{progress_description}' failed with an unexpected error: {e}",
                        exc_info=True,
                    )
                    outputs[index] = (
                        ""  # Ensure empty string on unexpected future error
                    )
        return outputs

    def _execute_openrouter_tasks_parallel_with_tokens(
        self,
        tasks_kwargs_list: List[Dict[str, Any]],
        max_workers_input: Optional[int],
        progress_description: str,
    ) -> tuple[List[str], int]:
        """
        Private helper to execute multiple tasks in parallel using ThreadPoolExecutor,
        specifically for OpenRouter calls that need token tracking.
        Returns both outputs and total tokens generated.
        """
        num_tasks = len(tasks_kwargs_list)
        if not num_tasks:
            return [], 0

        outputs = [""] * num_tasks  # Pre-allocate for ordered results
        total_tokens = 0

        effective_max_workers = 1
        if max_workers_input is not None:
            if max_workers_input > 0:
                effective_max_workers = min(max_workers_input, num_tasks)
            else:
                logging.warning(
                    f"max_workers was {max_workers_input}, which is <= 0. Using 1 worker for {progress_description}."
                )
        else:  # max_workers_input is None (use default)
            effective_max_workers = min(
                10, num_tasks
            )  # Default: 10, or fewer if fewer tasks

        # Ensure effective_max_workers is at least 1 if there are tasks
        if num_tasks > 0 and effective_max_workers <= 0:  # Check for <=0 to be safe
            effective_max_workers = 1

        logging.info(
            f"Using {effective_max_workers} worker thread(s) for: {progress_description}"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=effective_max_workers
        ) as executor:
            future_to_index = {
                executor.submit(
                    self._execute_single_chat_api_turn_with_tokens, **task_kwargs
                ): index
                for index, task_kwargs in enumerate(tasks_kwargs_list)
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_index),
                total=num_tasks,
                desc=progress_description,
            ):
                index = future_to_index[future]
                try:
                    result_str, tokens = future.result()
                    outputs[index] = result_str
                    total_tokens += tokens
                except Exception as e:
                    # Identify the task for logging. Assumes 'request_idx' or 'instance_idx' might be in kwargs.
                    task_identifier_val = tasks_kwargs_list[index].get(
                        "request_idx",
                        tasks_kwargs_list[index].get(
                            "instance_idx", f"task_index_{index}"
                        ),
                    )
                    logging.error(
                        f"Parallel task (ID: {task_identifier_val}) for '{progress_description}' failed with an unexpected error: {e}",
                        exc_info=True,
                    )
                    outputs[index] = (
                        ""  # Ensure empty string on unexpected future error
                    )
        return outputs, total_tokens

    def generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = 512,
        temperature: float = 0.1,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        sleep_time: Optional[float] = None,
        max_workers: Optional[int] = None,
        enable_thinking: bool = False,
        majority_vote: bool = False,
    ) -> List[str]:
        if not prompts:
            return []

        num_prompts = len(prompts)
        logging.info(
            f"Generating {num_prompts} completions with {self.provider} model {self.model_name}..."
        )
        logging.debug(
            f"Sampling params: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}, stop={stop}"
        )

        if self.provider == "vllm":
            outputs_text: List[str] = [""] * num_prompts
            vllm_sampling_params = VLLM_SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop if stop else [],
            )
            # Prepare messages for the chat format
            vllm_messages = [[{"role": "user", "content": p}] for p in prompts]

            try:
                # Use the chat method for single-turn generation
                request_outputs = self.client.chat(
                    messages=vllm_messages,
                    sampling_params=vllm_sampling_params,
                    use_tqdm=True,  # Matches behavior in multi-turn
                    chat_template_kwargs={"enable_thinking": enable_thinking},
                )
                for i, output in enumerate(request_outputs):
                    if output.outputs:
                        outputs_text[i] = output.outputs[0].text.strip()
                    else:
                        logging.warning(f"vLLM returned no output for prompt index {i}")
                        outputs_text[i] = ""
                return outputs_text
            except Exception as e:
                logging.error(f"Error during vLLM chat generation: {e}", exc_info=True)
                return [""] * num_prompts

        elif self.provider == "openrouter":
            tasks_kwargs_list = []
            for i, prompt_content in enumerate(prompts):
                tasks_kwargs_list.append(
                    {
                        "messages": [{"role": "user", "content": prompt_content}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stop": stop,
                        "sleep_time": sleep_time,
                        "request_idx": i,
                    }
                )

            return self._execute_openrouter_tasks_parallel(
                tasks_kwargs_list=tasks_kwargs_list,
                target_callable=self._execute_single_chat_api_turn,
                max_workers_input=max_workers,
                progress_description="Generating with OpenRouter (parallel)",
            )
        else:
            logging.error(
                f"Unsupported provider {self.provider} in UnifiedLLM.generate."
            )
            return [""] * num_prompts

    def _process_single_instance_chat(
        self,
        instance_idx: int,
        instance_prompts: List[str],
        start_template: str,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        sleep_time: Optional[float],
    ) -> str:
        current_history: List[Dict[str, str]] = []
        current_outputs: List[str] = []

        if start_template:  # Only add if start_template is non-empty
            current_history.append({"role": "system", "content": start_template})
            logging.debug(
                f"[Instance {instance_idx}] Starting chat history with start template: {start_template[:50]}..."
            )
        else:
            logging.debug(
                f"[Instance {instance_idx}] Starting chat history without a start template."
            )

        effective_max_tokens_per_turn = None
        if max_tokens is not None:
            if instance_prompts:  # Ensure there are prompts to divide by
                effective_max_tokens_per_turn = max_tokens // len(instance_prompts)
            else:  # If no instance_prompts, but start_template might be used for a single turn.
                # This case implies _execute_single_chat_api_turn would be called once.
                # However, the loop below iterates over instance_prompts.
                # If instance_prompts is empty, the loop won't run.
                # If start_template is the ONLY input, this function might need adjustment
                # or the calling logic ensures instance_prompts has at least one item if max_tokens division is critical.
                # For now, if instance_prompts is empty, max_tokens won't be divided by zero.
                # If max_tokens is meant for a single call (no instance_prompts), this should be 'max_tokens'.
                # Let's assume if instance_prompts is empty, no turns happen here.
                pass

            if (
                effective_max_tokens_per_turn is not None
                and effective_max_tokens_per_turn <= 0
            ):
                logging.warning(
                    f"[Instance {instance_idx}] Calculated effective_max_tokens per turn is non-positive ({effective_max_tokens_per_turn}) from total max_tokens ({max_tokens}) and {len(instance_prompts)} turns. Setting to default 128 for subsequent turns if any."
                )
                effective_max_tokens_per_turn = 128
            elif (
                max_tokens is not None and not instance_prompts
            ):  # max_tokens is set but no turns to divide it over
                # This state means the loop over instance_prompts won't run.
                # If the intent was a single call with start_template, this function isn't structured for it.
                # The current logic is fine if instance_prompts is the driver of turns.
                pass
        else:
            logging.debug(
                f"[Instance {instance_idx}] max_tokens is None for the entire chat. API will use its default for each turn."
            )

        logging.debug(
            f"[Instance {instance_idx}] Using effective max_tokens per turn: {effective_max_tokens_per_turn}"
        )

        for turn_idx, user_prompt in tqdm(
            enumerate(instance_prompts),
            desc=f"Processing instance {instance_idx + 1}",
            total=len(instance_prompts),
            leave=False,
        ):
            user_prompt_processed = user_prompt.strip()
            # if self.model_name.lower().startswith("qwen"):
            #     logging.debug(
            #         f"[Instance {instance_idx}, Turn {turn_idx}] Appending '/no_think' to user prompt."
            #     )
            #     user_prompt_processed = user_prompt_processed + " /no_think"

            current_history.append({"role": "user", "content": user_prompt_processed})
            logging.debug(
                f"[Instance {instance_idx}, Turn {turn_idx}] Processing user prompt: '{user_prompt_processed[:50]}...'"
            )

            assistant_response = self._execute_single_chat_api_turn(
                messages=current_history,
                max_tokens=effective_max_tokens_per_turn,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                sleep_time=sleep_time,
                request_idx=f"{instance_idx}-{turn_idx}",
            )
            logging.debug(
                f"[Instance {instance_idx}, Turn {turn_idx}] Received assistant response: '{assistant_response[:50]}...'"
            )

            if not assistant_response:
                logging.warning(
                    f"[Instance {instance_idx}, Turn {turn_idx}] LLM response failed or was empty for prompt: '{user_prompt_processed[:100]}...'"
                )
                current_outputs.append("NaN")
            else:
                history_content = self._remove_thinking_traces(assistant_response)
                current_history.append(
                    {"role": "assistant", "content": history_content}
                )
                current_outputs.append(assistant_response)

        if not current_outputs:
            # This can happen if instance_prompts was empty
            if not instance_prompts and start_template:
                logging.warning(
                    f"[Instance {instance_idx}] No user_prompts provided for this instance, so no chat turns executed beyond potential implicit use of start_template (if any). Returning empty string."
                )
            elif not instance_prompts and not start_template:
                logging.warning(
                    f"[Instance {instance_idx}] No start_template and no user_prompts provided. Returning empty string."
                )
            else:  # instance_prompts was not empty, but all turns failed
                logging.warning(
                    f"[Instance {instance_idx}] No outputs generated for this instance (all turns might have failed)."
                )
            return ""

        return "|".join(current_outputs)

    @timing_decorator
    def chat_generate(
        self,
        start_template: str,
        user_prompts_list: List[List[str]],
        max_tokens: Optional[int] = 512,
        temperature: float = 0.1,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        sleep_time: Optional[float] = None,
        max_workers: Optional[int] = None,
        enable_thinking: bool = False,
    ) -> List[str]:
        if not user_prompts_list:
            logging.warning(
                "Empty user_prompts_list provided to chat_generate. Returning empty list."
            )
            return []

        # start_template can be empty, handled by _process_single_instance_chat

        num_instances = len(user_prompts_list)
        logging.info(
            f"Starting chat generation for {num_instances} instances "
            f"with provider {self.provider} and model {self.model_name}."
        )

        if self.provider == "vllm":
            # Use vLLM's native chat functionality
            logging.info(
                "Using vLLM's native chat method for multi-turn conversations."
            )
            outputs = []

            for instance_idx, instance_prompts in enumerate(
                tqdm(
                    user_prompts_list,
                    desc="Processing vLLM chat instances",
                    total=num_instances,
                )
            ):
                try:
                    output = self._process_single_instance_vllm_chat(
                        instance_idx=instance_idx,
                        instance_prompts=instance_prompts,
                        start_template=start_template,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        enable_thinking=enable_thinking,
                    )
                    outputs.append(output)
                except Exception as e:
                    logging.error(
                        f"Error processing vLLM chat instance {instance_idx}: {e}",
                        exc_info=True,
                    )
                    outputs.append("")

            return outputs

        elif self.provider == "openrouter":
            # Use existing OpenRouter parallel processing
            logging.info("Using OpenRouter chat API for multi-turn conversations.")
            tasks_kwargs_list = []
            for i, instance_prompts_data in enumerate(user_prompts_list):
                tasks_kwargs_list.append(
                    {
                        "instance_idx": i,
                        "instance_prompts": instance_prompts_data,
                        "start_template": start_template,
                        "max_tokens": max_tokens,  # This is total for the instance, _process_single_instance_chat divides it
                        "temperature": temperature,
                        "top_p": top_p,
                        "stop": stop,
                        "sleep_time": sleep_time,
                    }
                )

            return self._execute_openrouter_tasks_parallel(
                tasks_kwargs_list=tasks_kwargs_list,
                target_callable=self._process_single_instance_chat,
                max_workers_input=max_workers,
                progress_description=f"Processing {num_instances} chat instances in parallel",
            )
        else:
            logging.error(
                f"Unsupported provider {self.provider} in UnifiedLLM.chat_generate."
            )
            return [""] * num_instances

    def fill_in_turn(self, instance_history, ground_truth_output, llm_output):
        """
        Fill in the assistant answers for a specific step.
        """

        def replace_answer(content: str, llm_output: str) -> str:
            if content is None:
                raise ValueError("Content cannot be None")
            # replace the answer tags in the llm_output with the ground truth output
            if llm_output is None:
                raise ValueError("LLM output cannot be None")
            logging.debug(
                f"Replacing answer in LLM output: {llm_output} with ground truth output: {ground_truth_output}"
            )
            # split the llm_output by <answer>
            parts = llm_output.split("<answer>")
            logging.debug(
                f"Splitting LLM output into parts: {parts} for ground truth output: {ground_truth_output}"
            )
            # get the first part and add the ground truth output
            llm_out = parts[0] + "<answer>" + str(ground_truth_output) + "</answer>"
            logging.debug(f"Replaced LLM output: {llm_out} ")
            return llm_out

        instance_history.append(
            {
                "role": "assistant",
                "content": replace_answer(ground_truth_output, llm_output),
            }
        )
        return instance_history

    def _implement_sliding_window(
        self,
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        if self.sliding_window_size is None or self.sliding_window_size <= 0:
            return history  # infinite window, return full history

        # Handle case where we only have system message or very short history
        if len(history) <= 1:  # Only system message or empty
            return history

        # Count conversation turns (user + assistant pairs) excluding system message
        # Each turn consists of 2 messages: user input + assistant response
        system_msg = history[0] if history[0]["role"] == "system" else None
        conversation_start_idx = 1 if system_msg else 0
        conversation_messages = history[conversation_start_idx:]

        # Calculate how many complete turns we have
        num_complete_turns = len(conversation_messages) // 2

        # If we have fewer turns than the window size, return full history
        if num_complete_turns <= self.sliding_window_size:
            return history

        # Keep only the last sliding_window_size turns (each turn = 2 messages)
        messages_to_keep = self.sliding_window_size * 2
        windowed_conversation = conversation_messages[-messages_to_keep:]

        # Reconstruct history with system message (if exists) + windowed conversation
        if system_msg:
            return [system_msg] + windowed_conversation
        else:
            return windowed_conversation

    def _perform_majority_vote(
        self,
        outputs: List[str],
    ):
        original_outputs = outputs.copy()

        # get the answer from the regex
        def extract_answer(output: str) -> str:
            if output is None:
                return ""
            match = re.search(r"<answer>(.*?)</answer>", output)
            if match:
                return match.group(1).strip()
            else:
                logging.warning(
                    f"Output '{output}' does not contain a valid <answer> tag. Returning empty string."
                )
            return ""

        outputs = [extract_answer(output) for output in outputs]
        # remove invalid outputs
        valid_outputs = []
        for out in outputs:
            try:
                out_int = int(out)
                valid_outputs.append(out_int)
            except ValueError:
                logging.warning(
                    f"Output '{out}' is not a valid integer, removing from majority vote."
                )
        if len(valid_outputs) == 0:
            logging.warning(
                "No valid outputs found for majority vote. returning random answer."
            )
            return f"<answer>{random.choice(original_outputs)}</answer>"

        # perform majority vote with random tie-breaking
        count = Counter(valid_outputs)
        max_count = max(count.values())
        top_answers = [answer for answer, c in count.items() if c == max_count]

        if len(top_answers) == 1:
            chosen = top_answers[0]
            logging.debug(
                f"Majority vote result: {chosen} with count {max_count} (no tie)"
            )
        else:
            chosen = random.choice(top_answers)
            logging.debug(
                f"Majority vote tie among {top_answers} with count {max_count}. Chose {chosen} randomly."
            )

        return f"<answer>{chosen}</answer>"

    def _process_step_wise_vllm_chat(
        self,
        start_template: str,
        user_prompts_list: List[List[str]],
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        enable_thinking: bool = False,
        evaluator: Optional[Any] = None,
        majority_vote: int = 1,
    ) -> List[str]:
        """
        Processes all instances step-by-step using vLLM's built-in chat method.
        For each step, all instances are processed together in a batch.
        """
        if not user_prompts_list:
            return []

        num_instances = len(user_prompts_list)
        max_steps = (
            max(len(prompts) for prompts in user_prompts_list)
            if user_prompts_list
            else 0
        )

        # Initialize conversation histories and outputs for each instance
        instance_histories: List[List[Dict[str, str]]] = [
            [] for _ in range(num_instances)
        ]
        instance_outputs: List[List[str]] = [[] for _ in range(num_instances)]

        # Add start template to all histories if provided
        if start_template:
            for history in instance_histories:
                history.append({"role": "system", "content": start_template})
            logging.debug(
                f"Added start template to all {num_instances} conversation histories"
            )

        # Calculate effective max tokens per turn
        effective_max_tokens_per_turn = None
        if max_tokens is not None:
            if max_steps > 0:
                if enable_thinking:
                    effective_max_tokens_per_turn = max_tokens
                else:
                    effective_max_tokens_per_turn = 10000
            if (
                effective_max_tokens_per_turn is not None
                and effective_max_tokens_per_turn <= 0
            ):
                logging.warning(
                    f"Calculated effective_max_tokens per turn is non-positive ({effective_max_tokens_per_turn}) "
                    f"from total max_tokens ({max_tokens}) and {max_steps} steps. Setting to default 128."
                )
                effective_max_tokens_per_turn = 128

        logging.debug(
            f"Using effective max_tokens per turn: {effective_max_tokens_per_turn}"
        )

        if self.incorrect_probability > 0.0:
            evaluator.add_noise_to_ground_truth(  # type: ignore
                err_probability=self.incorrect_probability
            )

        # Process each step across all instances
        for step_idx in tqdm(range(max_steps), desc="Processing steps"):

            num_tokens_generated = 0
            active_messages = []

            for instance_idx, instance_prompts in enumerate(user_prompts_list):
                user_prompt = instance_prompts[step_idx].strip()

                # Add user message to this instance's history
                instance_histories[instance_idx].append(
                    {"role": "user", "content": user_prompt}
                )

                # Add to batch for processing
                active_messages.append(instance_histories[instance_idx].copy())

            logging.debug(f"Step {step_idx}: Processing {num_instances} instances")

            # Check if we should only run LLM at target_step
            if self.target_step is not None and step_idx != self.target_step:
                # Skip LLM execution and use ground truth for history
                logging.debug(f"Skipping LLM execution at step {step_idx}, target_step is {self.target_step}")
                
                for instance_idx in range(num_instances):
                    if evaluator:
                        # Use ground truth from evaluator
                        gt = evaluator.entries["output"][step_idx][instance_idx]
                        gt_response = f"<answer>{gt}</answer>"
                        
                        instance_histories[instance_idx].append(
                            {"role": "assistant", "content": gt_response}
                        )
                        instance_outputs[instance_idx].append(gt_response)
                        
                        logging.debug(f"Step {step_idx}, Instance {instance_idx}: Used ground truth: {gt}")
                    else:
                        # No evaluator available, use placeholder
                        instance_histories[instance_idx].append(
                            {"role": "assistant", "content": "Ground Truth Used"}
                        )
                        instance_outputs[instance_idx].append("Ground Truth Used")

                # Still evaluate this step if evaluator exists
                # if evaluator:
                #     step_delimiter = getattr(
                #         getattr(evaluator, "parser", None), "STEP_DELIMITER", "|"
                #     )
                #     this_step_outputs = step_delimiter.join(
                #         instance_outputs[instance_idx][step_idx]
                #         for instance_idx in range(num_instances)
                #     )

                #     evaluator.evaluate_step(
                #         llm_output=this_step_outputs,
                #         step=step_idx,
                #         num_tokens_generated=0,  # No tokens generated when using ground truth
                #         enable_thinking=enable_thinking,
                #         filled_history=True,  # Always true when using ground truth
                #     )

                self.tokens_generated_per_step.append(0)
                continue  # Skip to next step

            try:
                request_outputs = []
                if majority_vote > 1:
                    for start_vote in range(0, majority_vote, 5):
                        current_n = (
                            5
                            if (majority_vote - start_vote) >= 5
                            else (majority_vote - start_vote)
                        )
                        batch_params = VLLM_SamplingParams(
                            temperature=temperature,
                            top_p=top_p,
                            min_tokens=2,
                            max_tokens=effective_max_tokens_per_turn,
                            stop=stop if stop else [],
                            n=current_n,
                        )
                        temp_outputs = self.client.chat(  # type: ignore
                            messages=active_messages,
                            sampling_params=batch_params,
                            use_tqdm=False,  # We already have outer progress bar
                            chat_template_kwargs={"enable_thinking": enable_thinking},
                        )
                        request_outputs.extend(temp_outputs)
                else:
                    single_params = VLLM_SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        min_tokens=2,
                        max_tokens=effective_max_tokens_per_turn,
                        stop=stop if stop else [],
                        n=1,
                    )
                    request_outputs = self.client.chat(  # type: ignore
                        messages=active_messages,
                        sampling_params=single_params,
                        use_tqdm=False,  # We already have outer progress bar
                        chat_template_kwargs={"enable_thinking": enable_thinking},
                    )

                # Process results and update histories
                for instance_idx in range(num_instances):
                    if (
                        instance_idx < len(request_outputs)
                        and request_outputs[instance_idx].outputs
                    ):
                        if majority_vote == 1:
                            num_tokens_generated += len(
                                request_outputs[instance_idx].outputs[0].token_ids
                            )
                            assistant_response = (
                                request_outputs[instance_idx].outputs[0].text.strip()
                            )
                        else:
                            # For majority vote, we need to aggregate outputs
                            assistant_responses = [
                                output.text.strip()
                                for output in request_outputs[instance_idx].outputs
                            ]
                            num_tokens_generated += sum(
                                len(output.token_ids)
                                for output in request_outputs[instance_idx].outputs
                            )
                            assistant_response = self._perform_majority_vote(
                                assistant_responses
                            )
                    else:
                        assistant_response = ""

                    if not assistant_response:
                        logging.warning(
                            f"Step {step_idx}, Instance {instance_idx}: vLLM response failed or was empty"
                        )
                        instance_histories[instance_idx].append(
                            {"role": "assistant", "content": "Please Continue..."}
                        )
                        instance_outputs[instance_idx].append("No Answer")
                    else:
                        history_content = self._remove_thinking_traces(
                            assistant_response
                        )

                        if self.fill_history:
                            # we fill the history with ground truth
                            gt = evaluator.entries["output"][step_idx][instance_idx]  # type: ignore

                            instance_histories[instance_idx] = self.fill_in_turn(
                                instance_histories[instance_idx], gt, history_content
                            )
                        else:
                            instance_histories[instance_idx].append(
                                {"role": "assistant", "content": history_content}
                            )

                        if history_content == "No Answer":
                            logging.warning(
                                f"Step {step_idx}, Instance {instance_idx}: Assistant response was 'No Answer'"
                            )
                            instance_outputs[instance_idx].append("No Answer")
                        else:
                            instance_outputs[instance_idx].append(assistant_response)

                    # Apply sliding window after completing a turn (user + assistant)
                    if self.sliding_window_size is not None:
                        # Count conversation turns: exclude system message, then divide by 2 (user + assistant pairs)
                        system_msg_count = (
                            1
                            if (
                                instance_histories[instance_idx]
                                and instance_histories[instance_idx][0]["role"]
                                == "system"
                            )
                            else 0
                        )
                        conversation_messages = (
                            len(instance_histories[instance_idx]) - system_msg_count
                        )
                        complete_turns = conversation_messages // 2

                        if complete_turns > self.sliding_window_size:
                            instance_histories[instance_idx] = (
                                self._implement_sliding_window(
                                    history=instance_histories[instance_idx],
                                )
                            )

                            logging.debug(
                                f"Applied sliding window to instance {instance_idx} after step {step_idx}. "
                                f"Kept last {self.sliding_window_size} turns, new history length: {len(instance_histories[instance_idx])} messages"
                            )

            except Exception as e:
                logging.error(
                    f"Error during step {step_idx} vLLM chat generation: {e}",
                    exc_info=True,
                )
                # Add error placeholder for all instances
                for instance_idx in range(num_instances):
                    instance_histories[instance_idx].append(
                        {"role": "assistant", "content": "No Answer"}
                    )
                    instance_outputs[instance_idx].append("NaN")

            # evaluate this step
            if evaluator:
                # Use the evaluator's declared step delimiter if available, else default to '|'
                step_delimiter = getattr(
                    getattr(evaluator, "parser", None), "STEP_DELIMITER", "|"
                )
                this_step_outputs = step_delimiter.join(
                    instance_outputs[instance_idx][step_idx]
                    for instance_idx in range(num_instances)
                )

                evaluator.evaluate_step(
                    llm_output=this_step_outputs,
                    step=step_idx,
                    num_tokens_generated=num_tokens_generated,
                    enable_thinking=enable_thinking,
                    filled_history=self.fill_history,
                )

                try:
                    acc = evaluator.metrics.step_correctness_array[step_idx]
                    if self.early_stopping and acc <= self.early_stopping_threshold:
                        logging.info(
                            f"Early stopping at step {step_idx} due to step correctness {acc} below threshold {self.early_stopping_threshold}"
                        )
                        break
                except IndexError:
                    logging.warning(
                        f"IndexError accessing prefix correctness for step {step_idx}. "
                        "This might happen if the evaluator has fewer steps than expected."
                    )

            self.tokens_generated_per_step.append(num_tokens_generated)

        # Convert outputs to the expected format ("|" separated strings)
        final_outputs = []
        for instance_idx in range(num_instances):
            if instance_outputs[instance_idx]:
                final_outputs.append("|".join(instance_outputs[instance_idx]))
            else:
                final_outputs.append("")

        return final_outputs

    def _process_step_wise_openrouter_chat(
        self,
        start_template: str,
        user_prompts_list: List[List[str]],
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        sleep_time: Optional[float],
        max_workers: Optional[int],
        enable_thinking: bool = False,
        evaluator: Optional[Any] = None,
    ) -> List[str]:
        """
        Processes all instances step-by-step using OpenRouter's chat API.
        For each step, all instances are processed together in parallel.
        """
        if not user_prompts_list:
            return []

        num_instances = len(user_prompts_list)
        max_steps = (
            max(len(prompts) for prompts in user_prompts_list)
            if user_prompts_list
            else 0
        )

        # Initialize conversation histories and outputs for each instance
        instance_histories: List[List[Dict[str, str]]] = [
            [] for _ in range(num_instances)
        ]
        instance_outputs: List[List[str]] = [[] for _ in range(num_instances)]

        # Add start template to all histories if provided
        if start_template:
            for history in instance_histories:
                history.append({"role": "system", "content": start_template})
            logging.debug(
                f"Added start template to all {num_instances} conversation histories"
            )

        # Calculate effective max tokens per turn
        effective_max_tokens_per_turn = None
        if max_tokens is not None:
            if max_steps > 0:
                if enable_thinking:
                    effective_max_tokens_per_turn = max_tokens
                else:
                    effective_max_tokens_per_turn = 10000  # cot
            if (
                effective_max_tokens_per_turn is not None
                and effective_max_tokens_per_turn <= 0
            ):
                logging.warning(
                    f"Calculated effective_max_tokens per turn is non-positive ({effective_max_tokens_per_turn}) "
                    f"from total max_tokens ({max_tokens}) and {max_steps} steps. Setting to default 128."
                )
                effective_max_tokens_per_turn = 128

        logging.debug(
            f"Using effective max_tokens per turn: {effective_max_tokens_per_turn}"
        )

        if self.incorrect_probability > 0.0:
            evaluator.add_noise_to_ground_truth(  # type: ignore
                err_probability=self.incorrect_probability
            )

        # Process each step across all instances
        for step_idx in tqdm(range(max_steps), desc="Processing steps"):

            # Check if we should only run LLM at target_step
            if self.target_step is not None and step_idx != self.target_step:
                # Skip LLM execution and use ground truth for history
                logging.debug(f"Skipping LLM execution at step {step_idx}, target_step is {self.target_step}")
                
                for instance_idx, instance_prompts in enumerate(user_prompts_list):
                    user_prompt = instance_prompts[step_idx].strip()
                    
                    # Add user message to this instance's history
                    instance_histories[instance_idx].append(
                        {"role": "user", "content": user_prompt}
                    )
                    
                    if evaluator:
                        # Use ground truth from evaluator
                        gt = evaluator.entries["output"][step_idx][instance_idx]
                        gt_response = f"<answer>{gt}</answer>"
                        
                        instance_histories[instance_idx].append(
                            {"role": "assistant", "content": gt_response}
                        )
                        instance_outputs[instance_idx].append(gt_response)
                        
                        logging.debug(f"Step {step_idx}, Instance {instance_idx}: Used ground truth: {gt}")
                    else:
                        # No evaluator available, use placeholder
                        instance_histories[instance_idx].append(
                            {"role": "assistant", "content": "Ground Truth Used"}
                        )
                        instance_outputs[instance_idx].append("Ground Truth Used")
                    
                    # Apply sliding window if needed
                    if self.sliding_window_size is not None:
                        system_msg_count = (
                            1
                            if (
                                instance_histories[instance_idx]
                                and instance_histories[instance_idx][0]["role"] == "system"
                            )
                            else 0
                        )
                        conversation_messages = (
                            len(instance_histories[instance_idx]) - system_msg_count
                        )
                        complete_turns = conversation_messages // 2

                        if complete_turns > self.sliding_window_size:
                            instance_histories[instance_idx] = (
                                self._implement_sliding_window(
                                    history=instance_histories[instance_idx],
                                )
                            )
                
                # Still evaluate this step if evaluator exists
                if evaluator:
                    step_delimiter = getattr(
                        getattr(evaluator, "parser", None), "STEP_DELIMITER", "|"
                    )
                    this_step_outputs = step_delimiter.join(
                        (
                            instance_outputs[instance_idx][step_idx]
                            if step_idx < len(instance_outputs[instance_idx])
                            else "No Answer"
                        )
                        for instance_idx in range(num_instances)
                    )

                    evaluator.evaluate_step(
                        llm_output=this_step_outputs,
                        step=step_idx,
                        num_tokens_generated=0,  # No tokens generated when using ground truth
                        enable_thinking=enable_thinking,
                        filled_history=True,  # Always true when using ground truth
                    )

                self.tokens_generated_per_step.append(0)
                continue  # Skip to next step

            # Collect all instance messages for this step
            tasks_kwargs_list = []
            active_instance_indices = []

            for instance_idx, instance_prompts in enumerate(user_prompts_list):
                user_prompt = instance_prompts[step_idx].strip()
                # if self.model_name.lower().startswith("qwen"):
                #     logging.debug(
                #         f"Step {step_idx}, Instance {instance_idx}: Appending '/no_think' to user prompt."
                #     )
                #     user_prompt = user_prompt + " /no_think"

                # Add user message to this instance's history
                instance_histories[instance_idx].append(
                    {"role": "user", "content": user_prompt}
                )

                tasks_kwargs_list.append(
                    {
                        "messages": instance_histories[instance_idx].copy(),
                        "max_tokens": effective_max_tokens_per_turn,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stop": stop,
                        "sleep_time": sleep_time,
                        "request_idx": f"{instance_idx}-{step_idx}",
                    }
                )
                active_instance_indices.append(instance_idx)

            if not tasks_kwargs_list:
                # No more prompts to process for any instance
                break

            logging.debug(
                f"Step {step_idx}: Processing {len(tasks_kwargs_list)} requests"
            )

            # Execute all requests for this step in parallel with token tracking
            step_responses, total_step_tokens = (
                self._execute_openrouter_tasks_parallel_with_tokens(
                    tasks_kwargs_list=tasks_kwargs_list,
                    max_workers_input=max_workers,
                    progress_description=f"Step {step_idx} OpenRouter requests",
                )
            )

            # Group responses by instance for majority voting
            instance_responses: Dict[int, str] = {}
            for response_idx, instance_idx in enumerate(active_instance_indices):
                instance_responses[instance_idx] = step_responses[response_idx]

            # Process results and update histories
            for instance_idx in instance_responses:
                assistant_response = instance_responses[instance_idx]

                if not assistant_response:
                    logging.warning(
                        f"Step {step_idx}, Instance {instance_idx}: OpenRouter response failed or was empty"
                    )
                    instance_histories[instance_idx].append(
                        {"role": "assistant", "content": "Please Continue..."}
                    )
                    instance_outputs[instance_idx].append("No Answer")
                else:
                    history_content = self._remove_thinking_traces(assistant_response)

                    if self.fill_history:
                        # we fill the history with ground truth
                        gt = evaluator.entries["output"][step_idx][instance_idx]  # type: ignore

                        instance_histories[instance_idx] = self.fill_in_turn(
                            instance_histories[instance_idx], gt, history_content
                        )
                    else:
                        instance_histories[instance_idx].append(
                            {"role": "assistant", "content": history_content}
                        )

                    if history_content == "No Answer":
                        logging.warning(
                            f"Step {step_idx}, Instance {instance_idx}: Assistant response was 'No Answer'"
                        )
                        instance_outputs[instance_idx].append("No Answer")
                    else:
                        instance_outputs[instance_idx].append(assistant_response)

                # Apply sliding window after completing a turn (user + assistant)
                if self.sliding_window_size is not None:
                    # Count conversation turns: exclude system message, then divide by 2 (user + assistant pairs)
                    system_msg_count = (
                        1
                        if (
                            instance_histories[instance_idx]
                            and instance_histories[instance_idx][0]["role"] == "system"
                        )
                        else 0
                    )
                    conversation_messages = (
                        len(instance_histories[instance_idx]) - system_msg_count
                    )
                    complete_turns = conversation_messages // 2

                    if complete_turns > self.sliding_window_size:
                        instance_histories[instance_idx] = (
                            self._implement_sliding_window(
                                history=instance_histories[instance_idx],
                            )
                        )

                        logging.debug(
                            f"Applied sliding window to instance {instance_idx} after step {step_idx}. "
                            f"Kept last {self.sliding_window_size} turns, new history length: {len(instance_histories[instance_idx])} messages"
                        )

            # Add total tokens generated in this step to the per-step tracking list
            self.tokens_generated_per_step.append(total_step_tokens)

            # evaluate this step
            if evaluator:
                step_delimiter = getattr(
                    getattr(evaluator, "parser", None), "STEP_DELIMITER", "|"
                )
                this_step_outputs = step_delimiter.join(
                    (
                        instance_outputs[instance_idx][step_idx]
                        if step_idx < len(instance_outputs[instance_idx])
                        else "No Answer"
                    )
                    for instance_idx in range(num_instances)
                )

                evaluator.evaluate_step(
                    llm_output=this_step_outputs,
                    step=step_idx,
                    num_tokens_generated=total_step_tokens,
                    enable_thinking=enable_thinking,
                    filled_history=self.fill_history,
                )

                try:
                    acc = evaluator.metrics.prefix_correctness_array[step_idx]
                    if self.early_stopping and acc <= self.early_stopping_threshold:
                        logging.info(
                            f"Early stopping at step {step_idx} due to prefix correctness {acc} below threshold {self.early_stopping_threshold}"
                        )
                        break
                except IndexError:
                    logging.warning(
                        f"IndexError accessing prefix correctness for step {step_idx}. "
                        "This might happen if the evaluator has fewer steps than expected."
                    )

        # Convert outputs to the expected format ("|" separated strings)
        final_outputs = []
        for instance_idx in range(num_instances):
            if instance_outputs[instance_idx]:
                final_outputs.append("|".join(instance_outputs[instance_idx]))
            else:
                final_outputs.append("")

        return final_outputs

    @timing_decorator
    def chat_generate_step_wise(
        self,
        start_template: str,
        user_prompts_list: List[List[str]],
        evaluator: Optional[Any] = None,
        max_tokens: Optional[int] = 512,
        temperature: float = 0.1,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        sleep_time: Optional[float] = None,
        max_workers: Optional[int] = None,
        enable_thinking: bool = False,
        majority_vote: int = 1,
    ) -> List[str]:
        """
        Alternative chat generation method that processes step-wise instead of instance-wise.
        For vLLM: processes all instances at each step together for better batching.
        For OpenRouter: falls back to the original method since it's already parallelized.
        """
        if not user_prompts_list:
            logging.warning(
                "Empty user_prompts_list provided to chat_generate_step_wise. Returning empty list."
            )
            return []

        num_instances = len(user_prompts_list)
        logging.info(
            f"Starting step-wise chat generation for {num_instances} instances "
            f"with provider {self.provider} and model {self.model_name}."
        )

        if self.provider == "vllm":
            logging.info(
                "Using step-wise vLLM processing: all instances processed together at each step."
            )
            return self._process_step_wise_vllm_chat(
                start_template=start_template,
                user_prompts_list=user_prompts_list,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                enable_thinking=enable_thinking,
                evaluator=evaluator,
                majority_vote=majority_vote,
            )

        elif self.provider == "openrouter":
            # Use step-wise OpenRouter processing for evaluation support
            logging.info(
                "Using step-wise OpenRouter processing: all instances processed step-by-step with evaluation."
            )
            return self._process_step_wise_openrouter_chat(
                start_template=start_template,
                user_prompts_list=user_prompts_list,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                sleep_time=sleep_time,
                max_workers=max_workers,
                enable_thinking=enable_thinking,
                evaluator=evaluator,
            )
        else:
            logging.error(
                f"Unsupported provider {self.provider} in UnifiedLLM.chat_generate_step_wise."
            )
            return [""] * num_instances

    def get_model_name_with_provider(self) -> str:
        return f"{self.provider}:{self.model_name}"

    def get_actual_tensor_parallel_size(self) -> Optional[int]:
        """Returns the actual tensor parallel size used, relevant for vLLM."""
        return self.actual_tp_size if self.provider == "vllm" else None

    def get_openrouter_credits(self) -> Optional[float]:
        """
        Gets the current credit balance from OpenRouter.
        Returns None if not using OpenRouter or if there's an error.
        """
        if self.provider != "openrouter":
            return None

        try:
            import requests

            # OpenRouter provides credit information via their API
            # We need to use requests directly since the OpenAI client doesn't have this endpoint
            headers = {
                "Authorization": f"Bearer {self.openrouter_config.api_key}",
                "Content-Type": "application/json",
            }
            response = requests.get(
                "https://openrouter.ai/api/v1/credits", headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                # According to API docs: data.total_usage contains the total usage
                credits = data.get("data", {}).get("total_usage")
                if credits is not None:
                    logging.debug(f"OpenRouter credit balance: {credits}")
                    return float(credits)
                else:
                    logging.warning("Credit balance not found in OpenRouter response")
                    return None
            else:
                logging.warning(
                    f"Failed to get OpenRouter credits: HTTP {response.status_code}"
                )
                return None
        except Exception as e:
            logging.error(f"Error getting OpenRouter credits: {e}")
            return None

    def calculate_openrouter_cost(
        self, initial_credits: Optional[float], final_credits: Optional[float]
    ) -> Optional[float]:
        """
        Calculates the cost of the run based on credit difference.

        Args:
            initial_credits: Credits before the run
            final_credits: Credits after the run

        Returns:
            Cost in credits, or None if calculation is not possible
        """
        if self.provider != "openrouter":
            return None

        if initial_credits is None or final_credits is None:
            logging.warning("Cannot calculate cost: missing credit information")
            return None

        cost = final_credits - initial_credits
        if cost < 0:
            logging.warning(
                f"Negative cost calculated: {cost}. This might indicate an error or credit addition during the run."
            )
            return None

        logging.info(f"OpenRouter run cost: {cost} credits")
        return cost
