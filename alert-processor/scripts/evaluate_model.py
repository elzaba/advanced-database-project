import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import logging
import yaml


# Initialize logger
def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate_model")
    return logger


logger = setup_logger()


# Load test data
def load_test_data(data_folder="../data/processed"):
    """
    Load test data from the specified folder.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, data_folder)

    test_data = {}
    try:
        for metric_name in os.listdir(data_folder):
            metric_dir = os.path.join(data_folder, metric_name)
            file_path = os.path.join(metric_dir, "test.csv")

            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                test_data[metric_name] = df
                logger.info(f"Loaded test data from {file_path}")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")

    return test_data


# Load all trained models and features
def load_models_and_features(model_folder="../models", metric_name=None):
    """
    Load the trained models and feature metadata for a specific metric.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    metric_dir = os.path.join(base_dir, model_folder, metric_name)

    models = {}
    features = []
    try:
        # Load features
        features_path = os.path.join(metric_dir, "features.pkl")
        with open(features_path, "rb") as features_file:
            features = pickle.load(features_file)

        # Load all models
        for model_name in ["IsolationForest", "RandomForest", "LogisticRegression", "OneClassSVM"]:
            model_path = os.path.join(metric_dir, f"{model_name}.pkl")
            if os.path.exists(model_path):
                with open(model_path, "rb") as model_file:
                    models[model_name] = pickle.load(model_file)

        logger.info(f"Loaded models and features successfully for metric: {metric_name}")
    except Exception as e:
        logger.error(f"Error loading models or features for metric '{metric_name}': {e}")

    return models, features


# Evaluate a specific model
def evaluate_model(test_data, model, features, is_one_class=False):
    """
    Evaluate a specific model on test data.
    """
    try:
        # Drop rows with missing features
        test_data = test_data.dropna(subset=features)
        X_test = test_data[features]

        # True labels
        y_true = test_data["label"]

        # Predict anomalies
        if is_one_class:
            predictions = model.predict(X_test)
            predictions = (predictions == -1).astype(int)  # Convert -1 to 1 for anomalies
        else:
            predictions = model.predict(X_test)

        # Generate classification report and confusion matrix
        report = classification_report(y_true, predictions, output_dict=True)
        confusion = confusion_matrix(y_true, predictions)

        logger.info(f"Classification Report:\n{classification_report(y_true, predictions)}")
        logger.info(f"Confusion Matrix:\n{confusion}")

        # Return F1-score for anomalies
        return report["1"]["f1-score"]
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return 0


# Update metrics.yml with best model information
def update_metrics_config(metric_name, best_model, best_f1_score, config_path="../config/metrics.yml"):
    """
    Update the metrics.yml file with the best model for the given metric.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(base_dir, config_path)

    try:
        # Load existing configuration
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Update best model information
        for metric in config.get("metrics", []):
            if metric["name"] == metric_name:
                metric["best_model"] = {"name": best_model, "f1_score": best_f1_score}
                break

        # Save updated configuration
        with open(config_path, "w") as file:
            yaml.safe_dump(config, file)

        logger.info(f"Updated best model for metric '{metric_name}' in {config_path}")
    except Exception as e:
        logger.error(f"Error updating metrics.yml: {e}")


# Main function to evaluate all models for each metric
def main():
    # Load test data
    test_data = load_test_data()
    if not test_data:
        logger.error("No test data found. Exiting.")
        return

    # Evaluate models for each metric
    for metric_name, df in test_data.items():
        logger.info(f"Evaluating models for metric: {metric_name}")

        # Load models and features
        models, features = load_models_and_features(metric_name=metric_name)
        if not models or not features:
            logger.warning(f"No models or features available for metric: {metric_name}")
            continue

        # Evaluate each model
        best_model = None
        best_f1_score = 0
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            is_one_class = model_name == "OneClassSVM"  # Special handling for One-Class SVM
            f1_score = evaluate_model(df, model, features, is_one_class=is_one_class)

            logger.info(f"{model_name} F1-Score for anomalies: {f1_score:.4f}")
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model = model_name

        logger.info(f"Best model for metric '{metric_name}' is: {best_model} with F1-Score: {best_f1_score:.4f}")

        # Update metrics.yml with the best model information
        # update_metrics_config(metric_name, best_model, best_f1_score)


if __name__ == "__main__":
    main()
