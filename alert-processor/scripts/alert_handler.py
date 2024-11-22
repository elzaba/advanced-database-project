import os
import json
import requests
import logging
from datetime import datetime

# Initialize logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("alert_handler")
    return logger

logger = setup_logger()

# Function to send alerts to Prometheus Alert API
def send_alert_to_prometheus(alerts, prometheus_url="http://localhost:9093/api/v1/alerts"):
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(prometheus_url, headers=headers, data=json.dumps(alerts))
        response.raise_for_status()
        logger.info(f"Successfully sent {len(alerts)} alerts to AlertManager.")
    except Exception as e:
        logger.error(f"Failed to send alerts to Prometheus: {e}")

# Function to read alerts from a file
def read_alerts_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            alerts = json.load(file)
        return alerts
    except Exception as e:
        logger.error(f"Failed to read alerts from {file_path}: {e}")
        return []

# Main function to fetch alerts and send them to Prometheus
def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    valid_alerts_path = os.path.join(base_dir, "../data/alerts/valid/valid_alerts_20241122_045007.json")
    invalid_alerts_path = os.path.join(base_dir, "../data/alerts/invalid/invalid_alerts_20241122_045007.json")

    # Read valid alerts
    valid_alerts = read_alerts_from_file(valid_alerts_path)
    if valid_alerts:
        # Append "source: alertProcess" to each alert
        for alert in valid_alerts:
            alert["labels"]["source"] = "alertProcess"
        # Send valid alerts to Prometheus
        send_alert_to_prometheus(valid_alerts)
    else:
        logger.info("No valid alerts to send.")

    # Read invalid alerts
    invalid_alerts = read_alerts_from_file(invalid_alerts_path)
    if invalid_alerts:
        # Append "source: alertProcess" to each alert
        for alert in invalid_alerts:
            alert["labels"]["source"] = "alertProcess"
        # Send invalid alerts to Prometheus
        send_alert_to_prometheus(invalid_alerts)
    else:
        logger.info("No invalid alerts to send.")

if __name__ == "__main__":
    main()
