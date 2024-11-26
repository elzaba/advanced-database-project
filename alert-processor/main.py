import sys
import os
import time
from scripts.fetch_metrics import main as fetch_metrics_main

# Ensure the alert-processor root directory is added to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

if __name__ == "__main__":
    # Fetch ALERT_INTERVAL from the environment variable, default to 43200 seconds (12 hours)
    alert_interval = int(os.getenv("ALERT_INTERVAL", 43200))
    
    while True:
        try:
            # Execute the fetch_metrics_main function
            fetch_metrics_main()
        except Exception as e:
            print(f"Error during metrics fetch: {e}")
        
        # Wait for the specified interval before fetching metrics again
        print(f"Sleeping for {alert_interval} seconds before the next fetch.")
        time.sleep(alert_interval)

