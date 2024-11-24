import os
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

    data = {}
    try:
        for metric_name in os.listdir(data_folder):
            metric_dir = os.path.join(data_folder, metric_name)
            file_path = os.path.join(metric_dir, "processed.csv")

            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                data[metric_name] = df
                logger.info(f"Loaded data from {file_path}")
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {e}")

    return data


# Split data into training and testing sets
def split_data(data, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets and save them as separate files.
    """
    train_test_data = {}
    for metric_name, df in data.items():
        try:
            # Drop rows with missing values in the required features
            features = ["value", "ema_mean", "ema_std", "upper_threshold", "lower_threshold", "z_score", "rate_of_change"]
            df = df.dropna(subset=features)

            # Split data
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            train_test_data[metric_name] = {"train": train_df, "test": test_df}

            # Save split data
            base_dir = os.path.abspath(os.path.dirname(__file__))
            metric_dir = os.path.join(base_dir, "../data/processed", metric_name)
            os.makedirs(metric_dir, exist_ok=True)
            train_df.to_csv(os.path.join(metric_dir, "train.csv"))
            test_df.to_csv(os.path.join(metric_dir, "test.csv"))

            logger.info(f"Data split and saved for metric '{metric_name}': {len(train_df)} train rows, {len(test_df)} test rows")
        except Exception as e:
            logger.error(f"Error splitting data for metric '{metric_name}': {e}")

    return train_test_data


# Train models
def train_models(data):
    """
    Train multiple models to detect anomalies.
    """
    models = {}
    for metric_name, datasets in data.items():
        try:
            train_df = datasets["train"]
            
            # Features for the model
            features = ["value", "ema_mean", "ema_std", "upper_threshold", "lower_threshold", "z_score", "rate_of_change"]
            X_train = train_df[features]
            y_train = train_df["label"]

            model_store = {}

            # Isolation Forest for anomaly detection
            isolation_forest = IsolationForest(random_state=42, contamination=0.05)
            isolation_forest.fit(X_train)
            model_store["IsolationForest"] = isolation_forest

            # Random Forest Classifier
            random_forest = RandomForestClassifier(random_state=42)
            random_forest.fit(X_train, y_train)
            model_store["RandomForest"] = random_forest

            # Logistic Regression
            logistic_regression = LogisticRegression(random_state=42, max_iter=500)
            logistic_regression.fit(X_train, y_train)
            model_store["LogisticRegression"] = logistic_regression

            models[metric_name] = {"models": model_store, "features": features}
            logger.info(f"Models trained successfully for metric '{metric_name}' with {len(X_train)} samples.")
        except Exception as e:
            logger.error(f"Error training models for metric '{metric_name}': {e}")

    return models


# Save the models and feature metadata
def save_models_and_metadata(models, model_folder="../models"):
    """
    Save the trained models and feature metadata to files in the respective metric folder.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    for metric_name, data in models.items():
        metric_dir = os.path.join(base_dir, model_folder, metric_name)
        os.makedirs(metric_dir, exist_ok=True)

        try:
            # Save models
            for model_name, model in data["models"].items():
                model_path = os.path.join(metric_dir, f"{model_name}.pkl")
                with open(model_path, "wb") as file:
                    pickle.dump(model, file)
                logger.info(f"{model_name} model saved to {model_path}")

            # Save feature metadata
            features_path = os.path.join(metric_dir, "features.pkl")
            with open(features_path, "wb") as file:
                pickle.dump(data["features"], file)
            logger.info(f"Feature metadata saved to {features_path}")
        except Exception as e:
            logger.error(f"Error saving models or metadata for metric '{metric_name}': {e}")


# Main function
def main():
    # Load preprocessed data
    data = load_preprocessed_data()

    if not data:
        logger.error("No preprocessed data found. Exiting.")
        return

    # Split data into train and test sets
    split_data_dict = split_data(data)

    # Train models
    models = train_models(split_data_dict)

    # Save models and metadata
    save_models_and_metadata(models)


if __name__ == "__main__":
    main()
