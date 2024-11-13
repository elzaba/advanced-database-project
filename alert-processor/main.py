import sys
import os

# Ensure the alert-processor root directory is added to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and execute the main function in fetch_metrics
from scripts.fetch_metrics import main as fetch_metrics_main

if __name__ == "__main__":
    fetch_metrics_main()
