import os
import pandas as pd
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import ruptures as rpt  # For Bayesian Changepoint Detection


# Initialize logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("advanced_statistical_models")
    return logger


logger = setup_logger()


def load_test_data(test_data_folder="../data/processed"):
    """
    Load test data from processed folder.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    test_data_folder = os.path.join(base_dir, test_data_folder)

    test_data = {}
    try:
        for metric_name in os.listdir(test_data_folder):
            test_file_path = os.path.join(test_data_folder, metric_name, "test.csv")
            if os.path.exists(test_file_path):
                df = pd.read_csv(test_file_path, index_col=0, parse_dates=True)
                test_data[metric_name] = df
                logger.info(f"Loaded test data for {metric_name} with {len(df)} rows.")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
    return test_data


def evaluate_arima(df, metric_name):
    """
    Use ARIMA for anomaly detection based on residuals.
    """
    try:
        model = ARIMA(df['value'], order=(1, 0, 1)).fit()
        residuals = df['value'] - model.fittedvalues
        threshold = 3 * residuals.std()
        df['arima_anomaly'] = (residuals.abs() > threshold).astype(int)

        logger.info(f"Evaluating ARIMA anomalies for {metric_name}...")
        evaluate_statistical_model(df, 'arima_anomaly')
    except Exception as e:
        logger.error(f"Error with ARIMA for {metric_name}: {e}")


def evaluate_seasonal_decompose(df, metric_name):
    """
    Use seasonal decomposition residuals for anomaly detection.
    """
    try:
        decomposition = seasonal_decompose(df['value'], model='additive', period=12)
        residuals = decomposition.resid
        threshold = 3 * residuals.std()
        df['decompose_anomaly'] = (residuals.abs() > threshold).astype(int)

        logger.info(f"Evaluating Seasonal Decomposition anomalies for {metric_name}...")
        evaluate_statistical_model(df, 'decompose_anomaly')
    except Exception as e:
        logger.error(f"Error with seasonal decomposition for {metric_name}: {e}")


def evaluate_bayesian_changepoint(df, metric_name):
    """
    Use Bayesian Changepoint Detection for anomaly detection.
    """
    try:
        algo = rpt.Pelt(model="rbf").fit(df['value'].values)
        breakpoints = algo.predict(pen=10)
        df['bcpd_anomaly'] = 0
        for bp in breakpoints:
            if bp < len(df):
                df.loc[df.index[bp], 'bcpd_anomaly'] = 1

        logger.info(f"Evaluating Bayesian Changepoint anomalies for {metric_name}...")
        evaluate_statistical_model(df, 'bcpd_anomaly')
    except Exception as e:
        logger.error(f"Error with Bayesian Changepoint Detection for {metric_name}: {e}")


def evaluate_statistical_model(df, anomaly_col):
    """
    Evaluate a statistical model using classification metrics.
    """
    try:
        y_true = df['final_anomaly']
        y_pred = df[anomaly_col]

        logger.info(f"Evaluation results for {anomaly_col}:")
        logger.info("\n" + classification_report(y_true, y_pred, zero_division=1))
        logger.info("Confusion Matrix:")
        logger.info("\n" + str(confusion_matrix(y_true, y_pred)))
    except Exception as e:
        logger.error(f"Error evaluating statistical model: {e}")


def main():
    # Load test data
    test_data = load_test_data()

    for metric_name, df in test_data.items():
        logger.info(f"Processing metric: {metric_name}")

        # Drop NaNs in 'value'
        df = df.dropna(subset=['value'])

        # Evaluate ARIMA-based anomaly detection
        evaluate_arima(df, metric_name)

        # Evaluate Seasonal Decomposition
        evaluate_seasonal_decompose(df, metric_name)

        # Evaluate Bayesian Changepoint Detection
        evaluate_bayesian_changepoint(df, metric_name)


if __name__ == "__main__":
    main()
