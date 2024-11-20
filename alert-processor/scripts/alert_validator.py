import os
import logging
import requests
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle

# Initialize logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("alert_validator")
    return logger

logger = setup_logger()

# Load trained model
def load_model(model_path="../models/isolation_forest.pkl"):
    """
    Load the trained anomaly detection model.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, model_path)

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Fetch alerts from Prometheus
def fetch_alerts(prometheus_url="http://localhost:9090", alert_query="ALERTS"):
    """
    Fetch alerts from Prometheus.
    """
    try:
        response = requests.get(f"{prometheus_url}/api/v1/query", params={"query": alert_query})
        response.raise_for_status()
        alerts = response.json().get("data", {}).get("result", [])
        logger.info(f"Fetched {len(alerts)} alerts from Prometheus.")
        return alerts
    except Exception as e:
        logger.error(f"Error fetching alerts from Prometheus: {e}")
        return []

# Validate alerts with the model
def validate_alerts(model, alerts):
    """
    Validate alerts using the trained model.
    """
    validated_alerts = []

    for alert in alerts:
        # Extract the relevant features
        try:
            timestamp = alert["value"][0]
            value = float(alert["value"][1])
            feature_data = {
                "value": value,
                "hour": pd.to_datetime(int(timestamp), unit='s').hour,
                "day_of_week": pd.to_datetime(int(timestamp), unit='s').dayofweek,
                "month": pd.to_datetime(int(timestamp), unit='s').month
            }
            feature_df = pd.DataFrame([feature_data])

            # Predict using the model
            prediction = model.predict(feature_df)
            is_valid = prediction[0] == -1  # -1 indicates anomaly in IsolationForest

            if is_valid:
                validated_alerts.append(alert)
                logger.info(f"Valid alert: {alert}")
            else:
                logger.info(f"Invalid alert: {alert}")

        except Exception as e:
            logger.error(f"Error validating alert: {e}")

    return validated_alerts

# Send validated alerts back to Prometheus
def send_alert_to_prometheus(alert, prometheus_push_url="http://localhost:9091/metrics/job/alert-validator"):
    """
    Send validated alerts to Prometheus Pushgateway.
    """
    try:
        payload = f"validated_alerts{{alert_name=\"{alert['metric']['alertname']}\"}} 1\n"
        response = requests.post(prometheus_push_url, data=payload)
        response.raise_for_status()
        logger.info(f"Successfully sent validated alert to Prometheus: {alert}")
    except Exception as e:
        logger.error(f"Error sending alert to Prometheus: {e}")

# Main function
def main():
    # Load the trained model
    model = load_model()
    if not model:
        logger.error("Failed to load model. Exiting.")
        return

    # Fetch alerts from Prometheus
    alerts = fetch_alerts()

    if not alerts:
        logger.info("No alerts fetched from Prometheus. Exiting.")
        return

    # Validate alerts
    validated_alerts = validate_alerts(model, alerts)

    # Send validated alerts to Prometheus
    for alert in validated_alerts:
        send_alert_to_prometheus(alert)

if __name__ == "__main__":
    main()
