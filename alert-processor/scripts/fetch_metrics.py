import os
import json
from datetime import datetime, timedelta
from prometheus_api_client import PrometheusConnect
import yaml
import logging


# Initialize logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("alert_processor")
    return logger


logger = setup_logger()


# Load settings from metrics.yml
def load_metrics_config(config_file="../config/metrics.yml"):
    try:
        with open(config_file, "r") as file:
            metrics_config = yaml.safe_load(file)
            logger.info("Loaded metrics configuration successfully.")
            return metrics_config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None


# Fetch metrics from Prometheus
def fetch_metrics(prometheus_url, metrics_config):
    prometheus = PrometheusConnect(url=prometheus_url, disable_ssl=True)
    fetched_data = {}
    end_time = datetime.now()

    for metric in metrics_config['metrics']:
        metric_name = metric['name']
        query = metric.get('query', metric_name)
        time_range = metric.get('time_range', '8h')
        step = metric.get('step', '15s')

        # Parse the time range for hours or minutes
        if time_range.endswith('h'):
            time_range_hours = int(time_range.replace('h', ''))
            start_time = end_time - timedelta(hours=time_range_hours)
        elif time_range.endswith('m'):
            time_range_minutes = int(time_range.replace('m', ''))
            start_time = end_time - timedelta(minutes=time_range_minutes)
        else:
            logger.error(f"Invalid time_range format for metric {metric_name}: {time_range}")
            fetched_data[metric_name] = []
            continue

        # Fetch data from Prometheus
        try:
            response = prometheus.custom_query_range(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step=step
            )
            if response:
                fetched_data[metric_name] = response
                logger.info(f"Fetched data for {metric_name} successfully.")
            else:
                fetched_data[metric_name] = []
                logger.warning(f"No data found for {metric_name}.")
        except Exception as e:
            logger.error(f"Error fetching metric {metric_name}: {e}")
            fetched_data[metric_name] = []

    return fetched_data


# Save fetched data to JSON files
def save_data(fetched_data, data_folder="../data/raw"):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, data_folder)
    current_date = datetime.now().strftime("%Y-%m-%d")

    for metric_name, data in fetched_data.items():
        metric_dir = os.path.join(data_folder, metric_name)
        file_path = os.path.join(metric_dir, f"{current_date}.json")

        # Create directories if they don't exist
        try:
            os.makedirs(metric_dir, exist_ok=True)
            logger.info(f"Ensured directory exists for {metric_dir}")

            # Write data to the file, creating or appending as needed
            if os.path.exists(file_path):
                with open(file_path, 'r+') as file:
                    existing_data = json.load(file)
                    combined_data = existing_data + data
                    file.seek(0)
                    json.dump(combined_data, file, indent=2)
            else:
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=2)
            logger.info(f"Saved data for {metric_name} to {file_path}")

        except Exception as e:
            logger.error(f"Error saving data for {metric_name} to {file_path}: {e}")


# Main function to execute fetching and saving of metrics
def main():
    # Load Prometheus URL and metrics configuration
    prometheus_url = "http://localhost:9090"
    metrics_config = load_metrics_config()

    if not metrics_config:
        logger.error("Failed to load metrics configuration. Exiting.")
        return

    # Fetch metrics data
    fetched_data = fetch_metrics(prometheus_url, metrics_config)

    # Save fetched data to specified folder
    save_data(fetched_data)


if __name__ == "__main__":
    main()
