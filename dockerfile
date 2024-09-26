# Use the official Python image as the base
FROM python:3.12-slim

# Set environment variables to avoid interactive prompts during installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies, including ffmpeg and imagemagick for moviepy
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    imagemagick \
    libgl1-mesa-glx \
    libxrender1 \
    libsm6 \
    libxext6 \
    && apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .


# Create directories for video processing
RUN mkdir -p downloads clips

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the Celery worker by default
CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]
