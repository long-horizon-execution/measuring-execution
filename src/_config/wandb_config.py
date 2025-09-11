from dataclasses import dataclass

@dataclass
class WandBSettings:
    project: str = "long-horizon-execution"
    entity: str = None
    tags: list[str] = None
    run_name_prefix: str = ""
    notes: str = ""
    mode: str = "disabled"
    log_json_artifact: bool = True
    artifact_name: str = "experiment_run"
