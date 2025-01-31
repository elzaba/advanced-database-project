import os
import yaml
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from scipy.stats import zscore


# Initialize logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("preprocessor_data")
    return logger


logger = setup_logger()


# Load configuration from metrics.yml
def load_metrics_config(config_path="../config/metrics.yml"):
    """
    Load metrics configuration from a YAML file.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(base_dir, config_path)

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded metrics configuration from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading metrics configuration: {e}")
        return {}


# Load raw data from JSON files
def load_raw_data(data_folder="../data/raw"):
    """
    Load raw data from JSON files stored in the specified data folder.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, data_folder)

    raw_data = {}
    try:
        for metric_name in os.listdir(data_folder):
            metric_dir = os.path.join(data_folder, metric_name)
            if not os.path.isdir(metric_dir):
                continue

            combined_data = []
            for file_name in os.listdir(metric_dir):
                if file_name.startswith('.'):
                    continue
                file_path = os.path.join(metric_dir, file_name)
                try:
                    with open(file_path, "r") as file:
                        data = json.load(file)
                        combined_data.extend(data)
                        logger.info(f"Loaded data from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading data from {file_path}: {e}")

            raw_data[metric_name] = combined_data
    except Exception as e:
        logger.error(f"Error loading raw data from {data_folder}: {e}")

    return raw_data


# Save historical data separately
def save_historical_data(df, metric_name, data_folder="../data/historical"):
    """
    Save historical data into a separate CSV file for each metric.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    historical_dir = os.path.join(base_dir, data_folder, metric_name)

    try:
        os.makedirs(historical_dir, exist_ok=True)
        historical_file = os.path.join(historical_dir, "historical_data.csv")

        if os.path.exists(historical_file):
            existing_data = pd.read_csv(historical_file, index_col=0, parse_dates=True)
            df = pd.concat([existing_data, df]).drop_duplicates().sort_index()

        df.to_csv(historical_file, index=True)
        logger.info(f"Historical data saved for metric '{metric_name}' to {historical_file}")
    except Exception as e:
        logger.error(f"Error saving historical data for metric '{metric_name}': {e}")

def resample_data(df):
    """
    Handles class imbalance using SMOTE
    """
    # Separate features and labels
    X = df.drop(columns=["label"])
    y = df["label"]

    # Apply SimpleImputer to handle missing values before resampling
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Apply SMOTE for oversampling the minority classes
    smote = SMOTE(sampling_strategy="auto", random_state=42)  # Oversample minority classes
    X_res, y_res = smote.fit_resample(X_imputed, y)

    # Optionally, undersample the majority class to balance the data further
    # undersample = RandomUnderSampler(sampling_strategy="auto", random_state=42)  # Reduce noise class
    # X_res, y_res = undersample.fit_resample(X_res, y_res)

    # Return the resampled data
    df_resampled = pd.DataFrame(X_res, columns=X.columns)
    df_resampled["label"] = y_res
    return df_resampled

# Preprocess data for each metric
def preprocess_data(raw_data, thresholds, span=10):
    """
    Preprocess raw data by cleaning, engineering features, and identifying anomalies.
    """
    processed_data = {}

    for metric_name, data in raw_data.items():
        try:
            if metric_name not in thresholds:
                logger.warning(f"No thresholds defined for metric '{metric_name}'")
                continue

            warning_threshold = thresholds[metric_name].get("warning")
            critical_threshold = thresholds[metric_name].get("critical")

            # Extract timestamps and values
            all_records = []
            for record in data:
                if "values" in record:
                    for timestamp, value in record["values"]:
                        all_records.append({"timestamp": timestamp, "value": value})

            # Convert to DataFrame
            if not all_records:
                logger.warning(f"No data available for metric {metric_name}")
                processed_data[metric_name] = pd.DataFrame()
                continue

            df = pd.DataFrame(all_records)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df.set_index("timestamp", inplace=True)

            # Save historical data
            save_historical_data(df, metric_name)

            # Data Cleaning
            df["value"] = df["value"].fillna(method="ffill")

            # Apply Z-score for basic anomaly detection
            df["z_score"] = zscore(df["value"].fillna(df["value"].mean()))
            df["rate_of_change"] = df["value"].diff()

            # Feature Engineering
            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek
            df["month"] = df.index.month
            # Exponential Moving Averages (EMA)
            df["ema_mean"] = df["value"].ewm(span=span, adjust=False).mean()
            df["ema_std"] = df["value"].ewm(span=span, adjust=False).std()

            # Dynamic Thresholds based on EMA
            df["upper_threshold"] = df["ema_mean"] + 3 * df["ema_std"]
            df["lower_threshold"] = df["ema_mean"] - 3 * df["ema_std"]

            # Labeling: 1 = Noise, -1 = Anomaly
            df["label"] = 1  # Default to Noise
            df.loc[(df["value"] > warning_threshold) & (df["value"] <= critical_threshold), "label"] = -1
            df.loc[df["value"] > critical_threshold, "label"] = -1

            # Apply resampling to handle class imbalance
            df_resampled = resample_data(df)

            # Save processed and resampled data
            processed_data[metric_name] = df_resampled
            logger.info(f"Processed data for metric {metric_name}")
        except Exception as e:
            logger.error(f"Error processing data for {metric_name}: {e}")
            processed_data[metric_name] = pd.DataFrame()

    return processed_data


# Save preprocessed data
def save_processed_data(processed_data, data_folder="../data/processed"):
    """
    Save processed data as CSV files in the specified data folder.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, data_folder)

    for metric_name, df in processed_data.items():
        metric_dir = os.path.join(data_folder, metric_name)
        file_path = os.path.join(metric_dir, "processed.csv")

        try:
            os.makedirs(metric_dir, exist_ok=True)
            df.to_csv(file_path)
            logger.info(f"Processed data saved for {metric_name} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data for {metric_name}: {e}")


# Main function
def main():
    # Load configuration and raw data
    config = load_metrics_config()
    thresholds = {metric["name"]: metric.get("thresholds", {}) for metric in config.get("metrics", [])}

    raw_data = load_raw_data()
    if not raw_data:
        logger.error("No raw data found. Exiting.")
        return

    # Preprocess data
    processed_data = preprocess_data(raw_data, thresholds)
    save_processed_data(processed_data)


if __name__ == "__main__":
    main()
