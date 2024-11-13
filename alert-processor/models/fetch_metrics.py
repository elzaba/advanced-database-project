import os
import json
from datetime import datetime, timedelta
from prometheus_api_client import PrometheusConnect
from ..config.config_loader import load_settings, load_logging_config


# Initialize logger
logger = load_logging_config()

def fetch_metrics(prometheus_url, metrics_config):
    # Initialize Prometheus connection with the provided URL
    prometheus = PrometheusConnect(url=prometheus_url, disable_ssl=True)

    fetched_data = {}
    end_time = datetime.now()

    # Loop through each metric in config and fetch data
    for metric in metrics_config['metrics']:
        metric_name = metric['name']
        query = metric.get('query', metric_name)
        time_range = metric.get('time_range', '1h')
        step = metric.get('step', '1m')

        # Check if the time range is in hours or minutes and calculate start_time accordingly
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


def save_data(fetched_data, data_folder="../data/raw"):
    os.makedirs(data_folder, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d")

    for metric_name, data in fetched_data.items():
        metric_dir = os.path.join(data_folder, metric_name)
        os.makedirs(metric_dir, exist_ok=True)
        file_path = os.path.join(metric_dir, f"{current_date}.json")

        try:
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


def main():
    # Load settings including Prometheus URL and metrics configuration
    prometheus_url, metrics_config = load_settings()

    # Fetch metrics
    fetched_data = fetch_metrics(prometheus_url, metrics_config)
    data_folder = os.path.join("..", "data", "raw")

    # Save fetched data
    print(fetched_data)
    save_data(fetched_data, data_folder)


if __name__ == "__main__":
    main()
