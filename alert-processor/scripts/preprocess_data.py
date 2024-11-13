import os
import json
import pandas as pd
from datetime import datetime


def preprocess_data_from_directory(raw_data_dir='../data/raw/kafka_messages_in_per_sec',
                                   processed_data_dir='../data/processed'):
    """
    Preprocesses metric data from JSON files in a specified directory.

    Args:
        raw_data_dir (str): Directory containing raw JSON data files.
        processed_data_dir (str): Directory to save the processed data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with resampled message rates.
    """
    data = []

    # Read each JSON file in the raw data directory
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(raw_data_dir, filename)
            with open(filepath, 'r') as f:
                raw_data = json.load(f)

                # Process each entry in the JSON data
                for metric_entry in raw_data:
                    instance = metric_entry['metric'].get('instance', 'unknown_instance')
                    topic = metric_entry['metric'].get('topic', 'unknown_topic')
                    for value in metric_entry['values']:
                        timestamp, rate = value
                        data.append({
                            'timestamp': datetime.fromtimestamp(timestamp),
                            'instance': instance,
                            'topic': topic,
                            'messages_in_per_sec': float(rate)
                        })

    # Debug: Check if data list is populated
    if not data:
        print("No data found to preprocess.")
        return pd.DataFrame()

    # Create DataFrame from the extracted data
    df = pd.DataFrame(data)
    print("DataFrame before resampling:", df.head())

    # Set timestamp as the index for resampling
    df.set_index('timestamp', inplace=True)

    # Resample data to hourly frequency and calculate the mean for each instance/topic
    try:
        df_resampled = df.groupby(['instance', 'topic']).resample('1h').mean().reset_index(level='timestamp')
    except Exception as e:
        print(f"Error during resampling: {e}")
        return pd.DataFrame()

    print("DataFrame after resampling:", df_resampled.head())

    # Save the processed data
    os.makedirs(processed_data_dir, exist_ok=True)
    output_filepath = os.path.join(processed_data_dir, 'preprocessed_data.csv')
    df_resampled.to_csv(output_filepath, index=False)
    print(f"Processed data saved to {output_filepath}")

    return df_resampled


# Example usage
processed_data = preprocess_data_from_directory()
print(processed_data)
