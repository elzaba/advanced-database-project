import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import logging


# Initialize logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_model")
    return logger


logger = setup_logger()


# Load preprocessed data
def load_preprocessed_data(data_folder="../data/processed"):
    """
    Load preprocessed data from the specified folder.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, data_folder)

    combined_data = pd.DataFrame()
    try:
        for metric_name in os.listdir(data_folder):
            metric_dir = os.path.join(data_folder, metric_name)
            file_path = os.path.join(metric_dir, "processed.csv")

            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df["metric"] = metric_name  # Add a column to differentiate metrics
                combined_data = pd.concat([combined_data, df], ignore_index=True)
                logger.info(f"Loaded data from {file_path}")
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {e}")

    return combined_data


# Train an Isolation Forest model
def train_model(data):
    """
    Train an Isolation Forest model to detect anomalies.
    """
    # Select features for training
    features = ["value", "hour", "day_of_week", "month", "rolling_mean", "rolling_std", "lag_1", "lag_2"]

    data = data.dropna(subset=features)  # Drop rows with missing values

    X = data[features]

    # Train Isolation Forest model
    model = IsolationForest(random_state=42, contamination=0.05)
    model.fit(X)

    # Predict anomalies
    data.loc[:, 'predicted_anomaly'] = model.predict(X)
    data.loc[:, 'predicted_anomaly'] = data['predicted_anomaly'].map({1: 0, -1: 1})  # Map to binary

    logger.info(f"Isolation Forest trained successfully with {len(X)} samples.")

    return model, data


# Save the model
def save_model(model, model_path="../models/isolation_forest.pkl"):
    """
    Save the trained model to a file.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, model_path)

    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")


# Load the model
def load_model(model_path="../models/isolation_forest.pkl"):
    """
    Load a trained model from a file.
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


# Predict anomalies using the model
def predict_anomalies(model, data):
    """
    Predict anomalies using the trained model.
    """
    features = ["value", "hour", "day_of_week", "month", "rolling_mean", "rolling_std", "lag_1", "lag_2"]
    data = data.dropna(subset=features)  # Drop rows with missing values

    X = data[features]
    predictions = model.predict(X)
    data["predicted_anomaly"] = predictions
    data["predicted_anomaly"] = data["predicted_anomaly"].map({1: 0, -1: 1})  # Map to binary (1 for anomaly, 0 for normal)

    return data


# Main function
def main():
    # Load preprocessed data
    data = load_preprocessed_data()

    if data.empty:
        logger.error("No preprocessed data found. Exiting.")
        return

    # Train model
    model, predictions = train_model(data)

    # Save the model
    save_model(model)

    # Predict anomalies (for demonstration, use the same data)
    model = load_model()
    if model:
        predictions = predict_anomalies(model, data)
        logger.info(f"Predictions:\n{predictions[['value', 'anomaly', 'predicted_anomaly']].head()}")


if __name__ == "__main__":
    main()
