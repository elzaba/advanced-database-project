import os
import json

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore

# Initialize logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("preprocessor_data")
    return logger


logger = setup_logger()


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
                file_path = os.path.join(metric_dir, file_name)
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        combined_data.extend(data)
                        logger.info(f"Loaded data from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading data from {file_path}: {e}")

            raw_data[metric_name] = combined_data
    except Exception as e:
        logger.error(f"Error loading raw data from {data_folder}: {e}")

    return raw_data


# Preprocess data for each metric
def preprocess_data(raw_data):
    """
    Preprocess raw data by cleaning, engineering features, and identifying anomalies.
    """
    processed_data = {}

    for metric_name, data in raw_data.items():
        try:
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
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['value'] = df['value'].astype(float)
            df.set_index('timestamp', inplace=True)

            # Data Cleaning
            df['value'].fillna(method='ffill', inplace=True)  # Handle missing values
            z_scores = np.abs(zscore(df['value']))
            df = df[z_scores < 3]  # Remove outliers

            # Feature Engineering
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['rolling_mean'] = df['value'].rolling(window=3).mean()
            df['rolling_std'] = df['value'].rolling(window=3).std()
            df['lag_1'] = df['value'].shift(1)
            df['lag_2'] = df['value'].shift(2)

            # Seasonal Decomposition
            try:
                decomposition = seasonal_decompose(df['value'], model='additive', period=3)
                df['trend'] = decomposition.trend
                df['seasonal'] = decomposition.seasonal
                df['residual'] = decomposition.resid
            except Exception as e:
                logger.warning(f"Seasonal decomposition failed for {metric_name}: {e}")

            # Dynamic Thresholding for Anomaly Detection
            df['upper_threshold'] = df['rolling_mean'] + 3 * df['rolling_std']
            df['lower_threshold'] = df['rolling_mean'] - 3 * df['rolling_std']
            df['anomaly'] = (df['value'] > df['upper_threshold']) | (df['value'] < df['lower_threshold'])

            processed_data[metric_name] = df
            logger.info(f"Processed data for {metric_name}")
        except Exception as e:
            logger.error(f"Error processing data for {metric_name}: {e}")
            processed_data[metric_name] = pd.DataFrame()

    return processed_data


def save_processed_data(processed_data, data_folder="../data/processed"):
    """
    Save processed data as CSV files in the specified data folder.
    If a file already exists, append the new data to it.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, data_folder)

    for metric_name, df in processed_data.items():
        metric_dir = os.path.join(data_folder, metric_name)
        file_path = os.path.join(metric_dir, "processed.csv")

        try:
            os.makedirs(metric_dir, exist_ok=True)

            if os.path.exists(file_path):
                # If the file exists, load it and append new data
                existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df = pd.concat([existing_data, df]).drop_duplicates().sort_index()

            # Save the combined data
            df.to_csv(file_path)
            logger.info(f"Appended and saved processed data for {metric_name} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data for {metric_name} to {file_path}: {e}")



# Main function
def main():
    # Load raw data
    raw_data = load_raw_data()

    if not raw_data:
        logger.error("No raw data found. Exiting.")
        return

    # Preprocess raw data
    processed_data = preprocess_data(raw_data)

    # Save processed data
    save_processed_data(processed_data)


if __name__ == "__main__":
    main()
