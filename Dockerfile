# Use a lightweight official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Install basic system dependencies required for building certain Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to optimize Docker layer caching
COPY requirements.txt /code/requirements.txt

# Install dependencies (This is where your heavy Legal-BERT/Torch libraries install)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all the backend files into the container
COPY . /code

# Set permissions so Hugging Face's internal user can read/write files
RUN chmod -R 777 /code

# Hugging Face Spaces explicitly listens on port 7860
EXPOSE 7860

# Run Uvicorn pointing to your server file
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
