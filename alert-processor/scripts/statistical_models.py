import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import zscore
import logging

# Initialize Logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("compare_statistical_models")
    return logger

logger = setup_logger()

# Load Test Data
def load_test_data(metric_name, data_folder="../data/processed"):
    """
    Load test data for the specified metric.
    """
    test_data_path = os.path.join(data_folder, metric_name, "test.csv")
    if not os.path.exists(test_data_path):
        logger.error(f"Test data not found for metric: {metric_name}")
        return None

    test_data = pd.read_csv(test_data_path, parse_dates=True, index_col=0)
    logger.info(f"Loaded test data for {metric_name} with {len(test_data)} rows.")
    return test_data

# Z-Score Anomaly Detection
def z_score_anomaly_detection(data, threshold=3):
    """
    Perform anomaly detection using Z-score.
    """
    data['z_score'] = zscore(data['value'].fillna(data['value'].mean()))
    data['z_score_anomaly'] = (data['z_score'].abs() > threshold).astype(int)
    return data

# Moving Average Anomaly Detection
def moving_average_anomaly_detection(data, window=3, threshold_factor=3):
    """
    Perform anomaly detection using moving average with dynamic thresholds.
    """
    data['moving_avg'] = data['value'].rolling(window=window).mean()
    data['moving_std'] = data['value'].rolling(window=window).std()

    data['upper_threshold'] = data['moving_avg'] + threshold_factor * data['moving_std']
    data['lower_threshold'] = data['moving_avg'] - threshold_factor * data['moving_std']

    data['moving_avg_anomaly'] = ((data['value'] > data['upper_threshold']) | (data['value'] < data['lower_threshold'])).astype(int)
    return data

# Evaluate Model
def evaluate_statistical_model(test_data, anomaly_column):
    """
    Evaluate statistical model performance using classification metrics.
    """
    if 'final_anomaly' not in test_data.columns:
        logger.error("Ground truth (final_anomaly) column missing in test data.")
        return

    y_true = test_data['final_anomaly']
    y_pred = test_data[anomaly_column]

    logger.info(f"Evaluation results for {anomaly_column}:")
    logger.info("\n" + classification_report(y_true, y_pred))

    logger.info("Confusion Matrix:")
    logger.info("\n" + str(confusion_matrix(y_true, y_pred)))

# Main Function
def main():
    metric_name = "kafka_broker_bytes_in_per_sec"  # Replace with your metric name
    test_data = load_test_data(metric_name)

    if test_data is None:
        return

    # Z-score Anomaly Detection
    logger.info("Evaluating Z-score Anomaly Detection...")
    test_data = z_score_anomaly_detection(test_data)
    evaluate_statistical_model(test_data, 'z_score_anomaly')

    # Moving Average Anomaly Detection
    logger.info("Evaluating Moving Average Anomaly Detection...")
    test_data = moving_average_anomaly_detection(test_data)
    evaluate_statistical_model(test_data, 'moving_avg_anomaly')

if __name__ == "__main__":
    main()