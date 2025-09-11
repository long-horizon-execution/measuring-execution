from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class OpenRouterConfig:
    api_key: str = os.getenv("OPENROUTER_API_KEY")
    api_base: str = "https://openrouter.ai/api/v1"
    llm_sleep_time: float = 3.0
