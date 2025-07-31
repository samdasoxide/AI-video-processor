# AI Video Processor - Intelligent Video Clipping Pipeline

**AI-powered video processing system that automatically analyzes YouTube videos and extracts the most engaging, high-quality clips. Combines OpenAI Whisper transcription, GPT-4 content analysis, sentiment analysis, and computer vision for intelligent clip selection. Features video enhancement, distributed worker architecture with Docker/Celery, Redis caching, and Supabase storage for scalable content curation.**

## Overview

This is a sophisticated AI-powered video processing system designed to automatically analyze YouTube videos and extract the most engaging, high-quality clips for content creators and marketers. The project combines cutting-edge artificial intelligence with distributed computing to create a scalable pipeline that transforms long-form video content into optimized short clips.

## Core Functionality

The system operates as an intelligent video curation engine that takes YouTube URLs as input and produces carefully selected, enhanced video clips as output. At its heart, the `VideoProcessor` class orchestrates a complex workflow that begins by downloading videos using `yt-dlp` with optimized settings for quality and performance. The system intelligently selects video formats up to 1080p resolution while ensuring efficient concurrent downloads.

## AI-Powered Content Analysis

The project leverages multiple AI technologies to understand and analyze video content. OpenAI's Whisper model provides accurate speech-to-text transcription, while GPT-4/GPT-4o performs sophisticated content analysis to identify the most valuable segments. The system doesn't just rely on basic metrics—it conducts deep sentiment analysis using Hugging Face transformers, evaluates audio quality characteristics, and performs computer vision analysis on video frames to assess visual quality, detect faces, and measure technical parameters like brightness, contrast, and sharpness.

## Intelligent Clip Selection

The most innovative aspect is the AI-driven clip identification process. The system feeds comprehensive analysis data—including full transcripts, sentiment scores, audio features, and visual quality metrics—into GPT-4, which acts as an intelligent curator. The AI evaluates content based on multiple criteria: standalone value, engagement potential, technical quality, and content richness. Each potential clip receives a multi-dimensional scoring system that weighs content quality (50%), technical excellence (30%), and engagement potential (20%).

## Advanced Video Processing

Beyond content analysis, the system includes sophisticated video enhancement capabilities. Individual frames undergo quality improvements through LAB color space processing with CLAHE (Contrast Limited Adaptive Histogram Equalization) and sharpening filters. The system automatically selects optimal encoding settings, utilizing hardware acceleration (H.264 NVENC) when GPU resources are available, falling back to software encoding otherwise.

## Scalable Architecture

The project is architected for production-scale deployment using a distributed worker system. Docker containerization ensures consistent environments, while Celery workers with Redis as the message broker enable horizontal scaling. The `docker-compose.yml` configuration sets up multiple services: worker nodes for processing, a beat scheduler for periodic tasks, Redis for caching and message queuing, and Flower for monitoring worker performance.

## Data Management and Storage

All processed content, metadata, and analytics are stored in Supabase, providing both database functionality and cloud storage. The system maintains comprehensive records including full transcripts, sentiment analysis results, quality metrics, and clip metadata. This creates a searchable archive of processed content with rich metadata for future analysis and optimization.

## Production-Ready Features

The system includes enterprise-level features such as Redis caching for performance optimization, retry mechanisms with exponential backoff for upload reliability, comprehensive logging for monitoring and debugging, and parallel processing capabilities that can handle multiple videos simultaneously. Error handling is robust, with graceful degradation and detailed logging for troubleshooting.

## System Architecture

### Periodic Ingestion Service
- **Supabase Bucket (Cloud Storage)**: This cloud storage periodically contains new CSV files with video URLs.
- **Ingestion Service**: A background service periodically checks the Supabase storage for new CSV files containing YouTube URLs. 
    - **Trigger**: When a new CSV file is detected, the service reads the file and extracts the video URLs.
    - **Next Step**: It sends these URLs to the Task Queue for processing.

### Task Queue (RabbitMQ/Kafka)
- **Task Queue Manager**: After the video URLs are collected, they are sent to a task queue like RabbitMQ or Kafka, where each video URL is added as a task to the queue.
    - **Purpose**: The queue ensures that tasks are distributed to worker nodes efficiently and manages load by distributing tasks among available workers.

### Worker Nodes (Dockerized or Kubernetes Pods)
This is where the core processing happens. Each worker node is responsible for a specific stage of video processing. 

#### Video Downloading
- **Download Worker**: Fetches the video from the URL (using `yt-dlp`) and handles retries for failed downloads.
    - **Data Flow**: Downloads the video file and stores it in Supabase storage.
    - **Next Step**: Once the download is successful, the next worker node is triggered for transcription.

#### Transcription (Whisper)
- **Transcription Worker**: Converts the audio in the video into text using Whisper (speech-to-text).
    - **Data Flow**: The transcription (text) and video metadata (e.g., video title, duration, uploader) are saved in Supabase storage.
    - **Next Step**: The transcription is forwarded to the next worker for processing.

#### NLP Processing (GPT-4)
- **NLP Worker**: This worker uses GPT-4 (or another model) to process the transcription and extract key points, summaries, and relevant segments of the video.
    - **Data Flow**: The worker analyzes the transcription text, extracting key segments (based on time markers) and saving this information to Supabase.
    - **Next Step**: The tagged segments are passed to the clipping worker.

#### Clipping
- **Clipping Worker**: Based on the tagged sections from the NLP worker, this worker generates clips of the most important parts of the video.
    - **Data Flow**: The generated clips (in formats like MP4) are saved in Supabase storage alongside the original video.
    - **Final Step**: The worker node marks the video processing as complete in the Supabase database.

### Supabase (Database & Cloud Storage)
- **Supabase Database**: Stores metadata, transcripts, and status tracking for video processing tasks.
    - **Tracks**: The pipeline progress for each video, storing information such as download status, transcription completion, and whether clips were generated.
- **Supabase Cloud Storage**: Holds the original video files, transcription text, processed metadata, and the final clips.

### Monitoring & Logging
- **Monitoring (Prometheus & Grafana)**: Monitors the health and performance of the pipeline, tracking CPU, memory usage, task queue performance, and worker node efficiency.
    - **Alerts**: Real-time alerts notify if a node crashes or if tasks are backing up in the queue.
- **Logging (ELK Stack)**: Logs key events and errors at each step (e.g., download failures, transcription issues). This ensures the pipeline can recover from errors and be debugged if issues arise.

### Data Flow Summary
- **1. Supabase Bucket (Ingestion)** → **2. Task Queue** → **3. Worker Nodes**:
   - Video Downloading → Transcription → NLP Processing → Clipping
- **4. Supabase Storage/Database**: Each step stores results back to Supabase.
- **5. Monitoring & Logging**: Continuously tracks and logs the pipeline's status.

## Getting Started

### Prerequisites
- Python 3.12+
- Docker and Docker Compose
- OpenAI API key
- Supabase account and credentials
- Redis (included in Docker setup)

### Installation
1. Clone the repository
2. Copy environment variables and configure API keys
3. Run with Docker Compose:
   ```bash
   docker-compose up -d
   ```
4. Access Flower monitoring dashboard at `http://localhost:5555`

This AI video processor represents a complete solution for automated content curation, combining the latest in AI technology with production-ready engineering to transform how organizations handle video content at scale.

<img width="523" alt="Screenshot 2024-10-02 at 9 51 11 AM" src="https://github.com/user-attachments/assets/fa78d6a4-4270-4069-9fa5-26adbc1eba66">
