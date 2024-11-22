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

# Load historical data for the given metric
def load_historical_data(metric_name, data_folder="../data/historical"):
    """
    Load historical data for a given metric from a CSV file.
    Removes duplicates based on timestamp.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    historical_data_path = os.path.join(base_dir, data_folder, metric_name, "historical_data.csv")
    try:
        if os.path.exists(historical_data_path):
            historical_data = pd.read_csv(historical_data_path, index_col=0, parse_dates=True)

            # Remove duplicates based on 'timestamp' and keep the first occurrence
            historical_data = historical_data.loc[~historical_data.index.duplicated(keep='first')]

            logger.info(f"Loaded historical data for {metric_name} from {historical_data_path}")
            return historical_data
        else:
            logger.warning(f"Historical data not found for {metric_name}. Path: {historical_data_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading historical data for {metric_name}: {e}")
        return None

# Calculate missing features for live alert based on historical data
def calculate_alert_features(alert, historical_data, window_size=3):
    """
    Calculate rolling mean, rolling std, z-score, and static alerts for live alert using historical data.
    """
    alert_timestamp = pd.to_datetime(alert["activeAt"][:-1])

    # Convert the alert to a DataFrame
    alert_df = pd.DataFrame([{
        'timestamp': alert_timestamp,
        'value': float(alert['value'])
    }]).set_index('timestamp')

    # Calculate historical rolling mean and std for reference
    historical_rolling_mean = historical_data['value'].rolling(window=window_size).mean()
    historical_rolling_std = historical_data['value'].rolling(window=window_size).std()

    # Apply rolling mean and std to alert data using historical reference
    alert_df['rolling_mean'] = historical_rolling_mean.loc[:alert_timestamp].iloc[-1]
    alert_df['rolling_std'] = historical_rolling_std.loc[:alert_timestamp].iloc[-1]

    # Calculate Z-Score for alert
    alert_df['z_score'] = (alert_df['value'] - alert_df['rolling_mean']) / alert_df['rolling_std']

    # Calculate static alert based on thresholds
    alert_df['upper_threshold'] = alert_df['rolling_mean'] + 3 * alert_df['rolling_std']
    alert_df['lower_threshold'] = alert_df['rolling_mean'] - 3 * alert_df['rolling_std']
    alert_df['static_alert'] = (alert_df['value'] > alert_df['upper_threshold']) | (alert_df['value'] < alert_df['lower_threshold'])

    return alert_df.iloc[0].to_dict()

# Validate an alert using the loaded model
def validate_alert(alert, model, features, historical_data, metric_name):
    try:
        # Calculate required features using historical data
        alert_features = calculate_alert_features(alert, historical_data)

        # Extract the relevant features for the model
        feature_df = pd.DataFrame([alert_features])[features]

        # Predict using the model
        prediction = model.predict(feature_df)

        # Return validity based on prediction result
        return "valid" if prediction[0] == -1 else "invalid"
    except Exception as e:
        logger.error(f"Error validating alert '{alert['labels']['alertname']}': {e}")
        return "error"

# Store alerts based on their validity status
def store_alerts(valid_alerts, invalid_alerts, unknown_alerts, error_alerts):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    alerts_folder = os.path.join(base_dir, "../data/alerts")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories for storing alerts
    valid_folder = os.path.join(alerts_folder, "valid", f"valid_alerts_{timestamp_str}.json")
    invalid_folder = os.path.join(alerts_folder, "invalid", f"invalid_alerts_{timestamp_str}.json")
    unknown_folder = os.path.join(alerts_folder, "unknown", f"unknown_alerts_{timestamp_str}.json")
    error_folder = os.path.join(alerts_folder, "error", f"error_alerts_{timestamp_str}.json")

    os.makedirs(os.path.dirname(valid_folder), exist_ok=True)
    os.makedirs(os.path.dirname(invalid_folder), exist_ok=True)
    os.makedirs(os.path.dirname(unknown_folder), exist_ok=True)
    os.makedirs(os.path.dirname(error_folder), exist_ok=True)

    # Update source to 'alertProcess'
    for alert in valid_alerts + invalid_alerts + unknown_alerts + error_alerts:
        alert["source"] = "alertProcess"

    # Save alerts to respective JSON files
    with open(valid_folder, "w") as valid_file:
        json.dump(valid_alerts, valid_file, indent=2)
    with open(invalid_folder, "w") as invalid_file:
        json.dump(invalid_alerts, invalid_file, indent=2)
    with open(unknown_folder, "w") as unknown_file:
        json.dump(unknown_alerts, unknown_file, indent=2)
    with open(error_folder, "w") as error_file:
        json.dump(error_alerts, error_file, indent=2)

    logger.info(f"Saved {len(valid_alerts)} valid alerts to '{valid_folder}'")
    logger.info(f"Saved {len(invalid_alerts)} invalid alerts to '{invalid_folder}'")
    logger.info(f"Saved {len(unknown_alerts)} unknown alerts to '{unknown_folder}'")
    logger.info(f"Saved {len(error_alerts)} error alerts to '{error_folder}'")

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

    valid_alerts = []
    invalid_alerts = []
    unknown_alerts = []
    error_alerts = []

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
            unknown_alerts.append(alert)
            continue

        # Load the best model and features for this metric
        model, features = load_model_and_features(metric["name"])
        if not model or not features:
            alert["validity"] = "error"
            error_alerts.append(alert)
            continue

        # Load historical data for this metric
        historical_data = load_historical_data(metric["name"])
        if historical_data is None:
            alert["validity"] = "error"
            error_alerts.append(alert)
            continue

        # Validate the alert
        alert["validity"] = validate_alert(alert, model, features, historical_data, metric["name"])
        if alert["validity"] == "valid":
            valid_alerts.append(alert)
        elif alert["validity"] == "invalid":
            invalid_alerts.append(alert)

    # Store alerts based on their validity
    store_alerts(valid_alerts, invalid_alerts, unknown_alerts, error_alerts)

if __name__ == "__main__":
    main()
