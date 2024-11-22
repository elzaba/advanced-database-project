import requests
import json
import logging

# Initialize logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("alert_checker")
    return logger

logger = setup_logger()

# Fetch alerts from Prometheus
def fetch_alerts(prometheus_url):
    """
    Fetch alerts from the Prometheus Alert API.
    """
    alerts_endpoint = f"{prometheus_url}/api/v1/alerts"
    try:
        response = requests.get(alerts_endpoint)
        response.raise_for_status()  # Raise an error if the request fails
        alerts = response.json()
        return alerts
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch alerts from Prometheus: {e}")
        return None

# Main function to print the response structure
def main():
    prometheus_url = "http://localhost:9090"  # Replace with your Prometheus URL
    alerts = fetch_alerts(prometheus_url)

    if alerts:
        logger.info("Fetched alerts successfully.")
        # Pretty print the JSON response for inspection
        print(json.dumps(alerts, indent=4))
    else:
        logger.error("No alerts fetched.")

if __name__ == "__main__":
    main()
