import os
import json
import requests
import pickle
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import joblib
import logging
from scipy.stats import zscore
from pathlib import Path
import argparse

# Paths and URLs
METRICS_CONFIG_PATH = Path("../config/metrics.yml")
MODELS_DIR = Path("../models")
ALERTMANAGER_URL = "http://localhost:9093/api/v1/alerts"
DATA_DIR = Path("../data/alerts")

# Initialize logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("alert_validator")
    return logger

logger = setup_logger()

def load_metrics_config():
    """Load metric configurations from metrics.yml."""
    try:
        with open(METRICS_CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)
            logging.info("Successfully loaded metrics configuration.")
            return config
    except Exception as e:
        logging.error(f"Failed to load metrics configuration: {e}")
        raise

def load_model_and_features(model_name, metric_name):
    """Load the specified model and feature details."""
    try:
        model_path = MODELS_DIR / metric_name / f"{model_name}.pkl"
        features_path = MODELS_DIR / metric_name / "features.pkl"
        model = joblib.load(model_path)
        with open(features_path, "rb") as f:
            features = joblib.load(f)
        logging.info(f"Model '{model_name}' and features loaded successfully.")
        return model, features
    except Exception as e:
        logging.error(f"Error loading model or features: {e}")
        raise

def fetch_prometheus_alerts(prometheus_url):
    """Fetch active alerts from Prometheus."""
    try:
        response = requests.get(f"{prometheus_url}/api/v1/alerts")
        response.raise_for_status()
        logging.info("Fetched active alerts from Prometheus.")
        return response.json()["data"]["alerts"]
    except Exception as e:
        logging.error(f"Failed to fetch alerts: {e}")
        raise

def fetch_metric_data(prometheus_url, query, time_range, step):
    """Fetch metric data for feature engineering."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=time_range)
    # URL encode the query
    # encoded_query = quote_plus(query)
    params = {
        "query": query,
        "start": start_time.isoformat() + 'Z',
        "end": end_time.isoformat() + 'Z',
        "step": step,
    }
    try:
        response = requests.get(f"{prometheus_url}/api/v1/query_range", params=params)
        response.raise_for_status()
        result = response.json()["data"]["result"]
        if result:
            df = pd.DataFrame(result[0]["values"], columns=["timestamp", "value"]).astype({"value": float})
            logging.info("Successfully fetched metric data.")
            return df
        else:
            logging.warning("No data found for the given metric query.")
            return pd.DataFrame(columns=["timestamp", "value"])
    except Exception as e:
        logging.error(f"Failed to fetch metric data: {e}")
        raise

def preprocess_alert(alert, metric_config):
    """Generate features for an incoming alert."""
    query = metric_config["query"]
    time_range = 10
    step = metric_config["step"]

    prometheus_url = metric_config.get("prometheus_url", "http://localhost:9090")

    metric_data = fetch_metric_data(prometheus_url, query, time_range, step)

    if metric_data.empty:
        raise ValueError("No data available for the metric query.")

    # Feature Engineering
    metric_data["timestamp"] = pd.to_datetime(metric_data["timestamp"], unit="s")
    metric_data.set_index("timestamp", inplace=True)
    metric_data["z_score"] = zscore(metric_data["value"].fillna(metric_data["value"].mean()))
    metric_data["rate_of_change"] = metric_data["value"].diff()
    metric_data["ema_mean"] = metric_data["value"].ewm(span=10, adjust=False).mean()
    metric_data["ema_std"] = metric_data["value"].ewm(span=10, adjust=False).std()
    metric_data["upper_threshold"] = metric_data["ema_mean"] + 3 * metric_data["ema_std"]
    metric_data["lower_threshold"] = metric_data["ema_mean"] - 3 * metric_data["ema_std"]

    # Rolling mean and std for dynamic thresholds
    rolling_mean = metric_data["value"].rolling(window=10).mean()
    rolling_std = metric_data["value"].rolling(window=10).std()

    # Static thresholds from config
    critical_threshold = metric_config["thresholds"]["critical"]
    warning_threshold = metric_config["thresholds"]["warning"]

    # Return the latest data as features
    features = metric_data.iloc[-1][["value", "ema_mean", "ema_std", "upper_threshold", "lower_threshold", "z_score", "rate_of_change"]].values
    return features.reshape(1, -1), warning_threshold, critical_threshold

def validate_alert(alert, model, metric_config):
    """Validate an alert using the trained model and statistical thresholds."""
    try:
        logging.debug(f"Starting validation for alert: {alert['labels']['alertname']}")
        logging.debug(f"Alert details: {alert}")

        # Preprocess the alert to extract features and thresholds
        alert_features, critical_threshold, warning_threshold = preprocess_alert(alert, metric_config)

        logging.debug(f"Extracted alert features: {alert_features}")
        logging.info(f"Critical threshold: {critical_threshold}, Warning threshold: {warning_threshold}")

        # Convert alert_features to a DataFrame for compatibility
        feature_names = ["value", "ema_mean", "ema_std", "upper_threshold", "lower_threshold", "z_score", "rate_of_change"]
        alert_features_df = pd.DataFrame(alert_features, columns=feature_names)
        
        logging.debug(f"Formatted alert features for model: {alert_features_df}")

        # Predict alert severity
        prediction = model.predict(alert_features_df)
        logging.debug(f"Isolation Forest prediction: {prediction}")

        # Extract the current value
        current_value = alert_features[0, 0]
        logging.info(f"Current metric value: {current_value}")

        # Determine severity based on thresholds and model prediction
        if prediction[0] == -1:  # Anomaly detected by Isolation Forest
            if current_value > critical_threshold:
                severity = "critical"
            elif current_value > warning_threshold:
                severity = "warning"
            else:
                severity = "noise"
            logging.info(f"Alert '{alert['labels']['alertname']}' classified as {severity} with current value: {current_value}")
        else:
            severity = "noise"
            logging.info(f"Alert '{alert['labels']['alertname']}' classified as noise by Isolation Forest model.")

        return severity
    except Exception as e:
        logging.error(f"Error validating alert '{alert['labels']['alertname']}': {e}", exc_info=True)
        return "noise"

def save_alerts(alerts, category):
    """Save alerts to appropriate files if not empty."""
    if not alerts:
        logger.info(f"No {category} alerts to save.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = DATA_DIR / category / f"{category}_alerts_{timestamp}.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(alerts, file, indent=2)

    logger.info(f"Saved {len(alerts)} {category} alerts to '{file_path}'.")

def forward_alerts(alert, severity):
    """Forward validated alerts to Notification Channels."""
    try:
        alert["labels"]["severity"] = severity
        alert["labels"]["source"] = "alertProcessor"
        formatted_alert = [
            {
                "labels": alert["labels"],
                "annotations": alert.get("annotations", {}),
                "startsAt": alert.get("activeAt", "")
            }
        ]
        logging.info(f"Formatted alert: {json.dumps(formatted_alert, indent=2)}")
        response = requests.post(ALERTMANAGER_URL, json=formatted_alert, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        logging.info(f"Alert '{alert['labels']['alertname']}' forwarded to Notification Channels.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to forward alert: {e}")
        logging.error(f"Response: {response.text}")    

# Main function
def main():
    """Main function for alert handling and validation."""
    config = load_metrics_config()
    valid_alerts, invalid_alerts = [], []

    for metric in config["metrics"]:
        model_name = metric["best_model"]["name"]
        metric_name = metric["name"]

        model, _ = load_model_and_features(model_name, metric_name)

        alerts = fetch_prometheus_alerts("http://localhost:9090")
        for alert in alerts:
            if alert["labels"]["alertname"] in metric["alert_names"]:
                severity = validate_alert(alert, model, metric)
                if severity != "noise":  # Forward only warning or critical alerts
                    valid_alerts.append(alert)
                    forward_alerts(alert, severity)
                else:
                    invalid_alerts.append(alert) 

    # Save alerts
    save_alerts(valid_alerts, "valid")
    save_alerts(invalid_alerts, "invalid")                   

if __name__ == "__main__":
    main()
