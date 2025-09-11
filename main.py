# main_script.py
import logging

from jsonargparse import ArgumentParser
from src.config import MainConfig
from experiment_runner import ExperimentRunner


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    logging.info(f"Starting experiment application...")

    try:
        parser = ArgumentParser(parse_as_dict=False)
        parser.add_class_arguments(MainConfig, "cfg")
        args = parser.parse_args()
        cfg = args.cfg

        logging.info(f"Effective configuration:\n{parser.dump(args)}")

        runner = ExperimentRunner(config=cfg)
        runner.run()

        logging.info("Experiment application finished successfully.")
        return 0

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        return 1


# --- Entry point ---
if __name__ == "__main__":
    main()
