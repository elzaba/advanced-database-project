import requests
import yaml
import numpy as np
from datetime import datetime, timedelta
import time  # Added for scheduling

# Prometheus server and alert.rules file path
PROMETHEUS_URL = "http://localhost:9090/api/v1/query_range"
RULES_FILE_PATH = './prometheus/alert.rules'  # Path to alert.rules file

# Define metrics to monitor
metrics = {
    "kafka_server_brokertopicmetrics_bytesinpersec": {"base_threshold": 500000},
    "kafka_server_brokertopicmetrics_bytesoutpersec": {"base_threshold": 500000},
}

def fetch_metric_data(metric, duration="1h", step="1m"):
    """Fetch data for a given metric from Prometheus."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    response = requests.get(PROMETHEUS_URL, params={
        'query': metric,
        'start': start_time.isoformat() + 'Z',
        'end': end_time.isoformat() + 'Z',
        'step': step
    })

    if response.status_code == 200:
        results = response.json().get("data", {}).get("result", [])
        values = [float(value[1]) for result in results for value in result.get("values", [])]
        return values
    else:
        print(f"Failed to fetch data for metric {metric}")
        return []

def reload_prometheus():
    """Reload Prometheus to apply updated alert rules."""
    try:
        response = requests.post("http://localhost:9090/-/reload")
        response.raise_for_status()
        print("Prometheus configuration reloaded successfully.")
    except requests.exceptions.HTTPError as err:
        print(f"Failed to reload Prometheus configuration: {err}")

def calculate_thresholds(values):
    """Calculate alert thresholds based on statistical analysis."""
    if not values:
        return None

    # Basic statistics
    mean_value = np.mean(values)
    std_dev = np.std(values)
    
    # Calculate thresholds
    low_threshold = mean_value + std_dev
    medium_threshold = mean_value + 2 * std_dev
    high_threshold = mean_value + 3 * std_dev
    critical_threshold = mean_value + 4 * std_dev
    
    # Calculate EMA for smoother thresholds
    ema = np.mean(values[-10:])  # Simple EMA calculation for last 10 values
    low_threshold = max(low_threshold, ema * 1.1)
    return low_threshold, medium_threshold, high_threshold, critical_threshold

def update_alert_rules(thresholds):
    """Update alert.rules with new thresholds."""
    with open(RULES_FILE_PATH, 'r') as file:
        rules = yaml.safe_load(file)

    for group in rules['groups']:
        for rule in group['rules']:
            metric_key = rule['expr'].split(' ')[0]
            if metric_key in thresholds:
                severity = rule['labels']['severity']
                rule['expr'] = f"{metric_key} > {thresholds[metric_key][severity]}"
    
    with open(RULES_FILE_PATH, 'w') as file:
        yaml.dump(rules, file, default_flow_style=False)
    print("Alert rules updated with new thresholds.")
    reload_prometheus()

def main():
    """Run dynamic alert processor periodically to update thresholds."""
    while True:
        # Store thresholds by metric and severity level
        severity_thresholds = {}

        for metric, config in metrics.items():
            values = fetch_metric_data(metric)
            if values:
                thresholds = calculate_thresholds(values)
                if thresholds:
                    severity_thresholds[metric] = {
                        'low': thresholds[0],
                        'medium': thresholds[1],
                        'high': thresholds[2],
                        'critical': thresholds[3]
                    }
                    print(f"Thresholds for {metric}: {severity_thresholds[metric]}")

        update_alert_rules(severity_thresholds)

        # Run every 5 minutes
        time.sleep(300)

if __name__ == "__main__":
    main()
