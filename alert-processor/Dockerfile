# Base image with Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY main.py .
COPY requirements.txt .
COPY scripts/ scripts/
RUN pip install -r requirements.txt

# Run the alert processor script
CMD ["python", "main.py"]
