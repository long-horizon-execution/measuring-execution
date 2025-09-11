import json
import logging


def save_json_to_file(data, filepath, indent=4):
    """Saves a dictionary to a JSON file."""
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent)
        logging.info(f"Successfully saved JSON to {filepath}")
    except Exception as e:
        logging.error(f"Error saving JSON to {filepath}: {e}", exc_info=True)

