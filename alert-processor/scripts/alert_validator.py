import os
import json
import requests
import pickle
import pandas as pd
import yaml
from datetime import datetime
import logging

from scipy.stats import zscore


# Initialize logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("alert_validator")
    return logger


logger = setup_logger()


# Load metrics configuration from metrics.yml
def load_metrics_config(config_path="../config/metrics.yml"):
    """
    Load metrics configuration from the specified file.
    """
    try:
        with open(config_path, "r") as file:
            metrics_config = yaml.safe_load(file)
        logger.info("Loaded metrics configuration successfully.")
        return metrics_config
    except Exception as e:
        logger.error(f"Failed to load metrics configuration. Exiting: {e}")
        return None


# Fetch alerts from Prometheus
def fetch_prometheus_alerts(prometheus_url="http://localhost:9090/api/v1/alerts"):
    """
    Fetch alerts from Prometheus.
    """
    try:
        response = requests.get(prometheus_url)
        response.raise_for_status()
        data = response.json()
        alerts = data.get("data", {}).get("alerts", [])
        logger.info(f"Fetched {len(alerts)} alerts from Prometheus.")
        return alerts
    except Exception as e:
        logger.error(f"Error fetching alerts from Prometheus: {e}")
        return []


# Load the model dynamically based on best_model from metrics.yml
def load_model_and_features(metric_name, model_folder="../models"):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    model_folder_path = os.path.join(base_dir, model_folder, metric_name)

    try:
        # Load metrics configuration to get best model information
        metrics_config = load_metrics_config()
        if not metrics_config:
            raise ValueError("Metrics configuration not found")

        # Get the best model's name from metrics.yml
        best_model_name = None
        for metric in metrics_config["metrics"]:
            if metric["name"] == metric_name:
                best_model_name = metric["best_model"]["name"]
                break

        if not best_model_name:
            raise ValueError(f"No best model found for metric '{metric_name}' in metrics.yml")

        # Construct the path for the best model
        model_path = os.path.join(model_folder_path, f"{best_model_name}.pkl")
        features_path = os.path.join(model_folder_path, "features.pkl")

        # Load the model and features
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with open(features_path, "rb") as features_file:
            features = pickle.load(features_file)

        logger.info(f"Loaded best model '{best_model_name}' and features for metric: {metric_name}")
        return model, features
    except Exception as e:
        logger.error(f"Error loading best model or features for metric '{metric_name}': {e}")
        return None, None


# Validate an alert using the loaded model
def validate_alert(alert, model, features):
    try:
        # Extract relevant features from the alert
        alert_features = {
            "value": float(alert["value"]),
            "hour": datetime.fromisoformat(alert["activeAt"][:-1]).hour,
            "day_of_week": datetime.fromisoformat(alert["activeAt"][:-1]).weekday(),
            "month": datetime.fromisoformat(alert["activeAt"][:-1]).month,
        }

        # Create a DataFrame for the model
        feature_df = pd.DataFrame([alert_features])

        # Check if the required features match
        if not all(feature in feature_df.columns for feature in features):
            raise ValueError(f"Missing required features: {features}")

        # Validate the alert
        prediction = model.predict(feature_df[features])
        return "valid" if prediction[0] == -1 else "invalid"
    except Exception as e:
        logger.error(f"Error validating alert '{alert['labels']['alertname']}': {e}")
        return "error"



# Load historical data for the given metric
def load_historical_data(metric_name, data_folder="../data/historical"):
    """
    Load historical data for a given metric from a CSV file.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    historical_data_path = os.path.join(base_dir, data_folder, metric_name, "historical_data.csv")
    try:
        if os.path.exists(historical_data_path):
            historical_data = pd.read_csv(historical_data_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded historical data for {metric_name} from {historical_data_path}")
            return historical_data
        else:
            logger.warning(f"Historical data not found for {metric_name}. Path: {historical_data_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading historical data for {metric_name}: {e}")
        return None


# Validate an alert using the loaded model
def validate_alert(alert, model, features, metric_name):
    try:
        # Extract relevant features from the alert
        alert_features = {
            "value": float(alert["value"]),
            "hour": datetime.fromisoformat(alert["activeAt"][:-1]).hour,
            "day_of_week": datetime.fromisoformat(alert["activeAt"][:-1]).weekday(),
            "month": datetime.fromisoformat(alert["activeAt"][:-1]).month,
        }

        # Load the metric configuration and get the thresholds
        metrics_config = load_metrics_config()
        if not metrics_config:
            raise ValueError("Metrics configuration not found")

        # Get the thresholds for the metric
        threshold_data = None
        for metric in metrics_config["metrics"]:
            if metric["name"] == metric_name:
                threshold_data = metric.get("thresholds", {})
                break

        if not threshold_data:
            raise ValueError(f"Threshold data not found for metric '{metric_name}' in metrics.yml")

        # Apply thresholds
        warning_threshold = threshold_data.get("warning", None)
        critical_threshold = threshold_data.get("critical", None)

        if warning_threshold is None or critical_threshold is None:
            raise ValueError(f"Missing thresholds for metric '{metric_name}' in metrics.yml")

        alert_features["warning_threshold"] = warning_threshold
        alert_features["critical_threshold"] = critical_threshold

        # Create a DataFrame for the model
        feature_df = pd.DataFrame([alert_features])

        # Ensure required features are present
        if not all(feature in feature_df.columns for feature in features):
            raise ValueError(f"Missing required features: {features}")

        # Validate the alert
        prediction = model.predict(feature_df[features])

        # Check if prediction is a single value or Series and extract the value
        if isinstance(prediction, pd.Series):
            prediction_value = prediction.iloc[0]  # Get the first prediction if it's a Series
        elif isinstance(prediction, pd.DataFrame):
            prediction_value = prediction.iloc[0, 0]  # Get the first value in DataFrame
        else:
            prediction_value = prediction  # If it's not a Series or DataFrame, assume it's a single value

        # Handle ambiguous truth value of prediction (check for NaNs, empty)
        if isinstance(prediction_value, (pd.Series, pd.DataFrame)):
            if prediction_value.empty:
                raise ValueError("Prediction result is empty.")
            prediction_value = prediction_value.iloc[0]  # Extract the first value

        # Return validity based on prediction result
        return "valid" if prediction_value == -1 else "invalid"
    except Exception as e:
        logger.error(f"Error validating alert '{alert['labels']['alertname']}': {e}")
        return "error"


# Main function
def main():
    # Load metrics configuration
    metrics_config = load_metrics_config()
    if not metrics_config:
        return

    # Fetch alerts from Prometheus
    alerts = fetch_prometheus_alerts()
    if not alerts:
        logger.error("No alerts fetched. Exiting.")
        return

    validated_alerts = []

    # Validate each alert
    for alert in alerts:
        alert_name = alert["labels"].get("alertname")
        if not alert_name:
            logger.warning(f"Alert without 'alertname': {alert}")
            continue

        # Find the matching metric in metrics.yml
        metric = next((m for m in metrics_config["metrics"] if alert_name in m["alert_names"]), None)
        if not metric:
            logger.warning(f"No matching metric for alert: {alert_name}")
            alert["validity"] = "unknown"
            validated_alerts.append(alert)
            continue

        # Load the best model and features for this metric
        model, features = load_model_and_features(metric["name"])
        if not model or not features:
            alert["validity"] = "error"
            validated_alerts.append(alert)
            continue

        # Load historical data for this metric
        historical_data = load_historical_data(metric["name"])
        if historical_data is None:
            alert["validity"] = "error"
            validated_alerts.append(alert)
            continue

        # Validate the alert
        alert["validity"] = validate_alert(alert, model, features, historical_data)
        validated_alerts.append(alert)

    # Log the validated alerts
    logger.info(f"Validated alerts: {json.dumps(validated_alerts, indent=2)}")


if __name__ == "__main__":
    main()
