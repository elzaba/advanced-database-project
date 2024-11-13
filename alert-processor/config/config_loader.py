import os
import yaml
import logging.config

def load_logging_config():
    """
    Loads and applies logging configuration from YAML file.
    """
    logging_config_path = os.path.join(os.path.dirname(__file__), "logging_config.yml")
    with open(logging_config_path, "r") as f:
        logging_config = yaml.safe_load(f)

    # Ensure the log directory exists
    log_file_path = logging_config["handlers"]["file"]["filename"]
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)

    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("alert_processor")
    return logger

def load_settings():
    """
    Loads Prometheus URL and metrics configuration from YAML file.
    """
    # Load Prometheus URL from settings file
    settings_path = os.path.join(os.path.dirname(__file__), "metrics.yml")
    with open(settings_path, "r") as f:
        config = yaml.safe_load(f)
    prometheus_url = config["settings"]["prometheus_url"]

    # Load metrics configuration directly from metrics.yml
    metrics_path = os.path.join(os.path.dirname(__file__), "metrics.yml")
    with open(metrics_path, "r") as f:
        metrics_config = yaml.safe_load(f)

    return prometheus_url, metrics_config

# Initialize logger and load settings
logger = load_logging_config()
prometheus_url, metrics_config = load_settings()
