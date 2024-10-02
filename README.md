
Here is a more detailed step-by-step textual representation of the system design for the periodic video processing pipeline:

---

### **1. Periodic Ingestion Service**
- **Supabase Bucket (Cloud Storage)**: This cloud storage periodically contains new CSV files with video URLs.
- **Ingestion Service**: A background service periodically checks the Supabase storage for new CSV files containing YouTube URLs. 
    - **Trigger**: When a new CSV file is detected, the service reads the file and extracts the video URLs.
    - **Next Step**: It sends these URLs to the Task Queue for processing.

---

### **2. Task Queue (RabbitMQ/Kafka)**
- **Task Queue Manager**: After the video URLs are collected, they are sent to a task queue like RabbitMQ or Kafka, where each video URL is added as a task to the queue.
    - **Purpose**: The queue ensures that tasks are distributed to worker nodes efficiently and manages load by distributing tasks among available workers.

---

### **3. Worker Nodes (Dockerized or Kubernetes Pods)**
This is where the core processing happens. Each worker node is responsible for a specific stage of video processing. 

#### **3.1 Video Downloading**
- **Download Worker**: Fetches the video from the URL (using `yt-dlp`) and handles retries for failed downloads.
    - **Data Flow**: Downloads the video file and stores it in Supabase storage.
    - **Next Step**: Once the download is successful, the next worker node is triggered for transcription.

#### **3.2 Transcription (Whisper)**
- **Transcription Worker**: Converts the audio in the video into text using Whisper (speech-to-text).
    - **Data Flow**: The transcription (text) and video metadata (e.g., video title, duration, uploader) are saved in Supabase storage.
    - **Next Step**: The transcription is forwarded to the next worker for processing.

#### **3.3 NLP Processing (GPT-4)**
- **NLP Worker**: This worker uses GPT-4 (or another model) to process the transcription and extract key points, summaries, and relevant segments of the video.
    - **Data Flow**: The worker analyzes the transcription text, extracting key segments (based on time markers) and saving this information to Supabase.
    - **Next Step**: The tagged segments are passed to the clipping worker.

#### **3.4 Clipping**
- **Clipping Worker**: Based on the tagged sections from the NLP worker, this worker generates clips of the most important parts of the video.
    - **Data Flow**: The generated clips (in formats like MP4) are saved in Supabase storage alongside the original video.
    - **Final Step**: The worker node marks the video processing as complete in the Supabase database.

---

### **4. Supabase (Database & Cloud Storage)**
- **Supabase Database**: Stores metadata, transcripts, and status tracking for video processing tasks.
    - **Tracks**: The pipeline progress for each video, storing information such as download status, transcription completion, and whether clips were generated.
- **Supabase Cloud Storage**: Holds the original video files, transcription text, processed metadata, and the final clips.

---

### **5. Monitoring & Logging**
- **Monitoring (Prometheus & Grafana)**: Monitors the health and performance of the pipeline, tracking CPU, memory usage, task queue performance, and worker node efficiency.
    - **Alerts**: Real-time alerts notify if a node crashes or if tasks are backing up in the queue.
- **Logging (ELK Stack)**: Logs key events and errors at each step (e.g., download failures, transcription issues). This ensures the pipeline can recover from errors and be debugged if issues arise.

---

### **Data Flow Summary**
- **1. Supabase Bucket (Ingestion)** → **2. Task Queue** → **3. Worker Nodes**:
   - Video Downloading → Transcription → NLP Processing → Clipping
- **4. Supabase Storage/Database**: Each step stores results back to Supabase.
- **5. Monitoring & Logging**: Continuously tracks and logs the pipeline's status.

---


<img width="523" alt="Screenshot 2024-10-02 at 9 51 11 AM" src="https://github.com/user-attachments/assets/fa78d6a4-4270-4069-9fa5-26adbc1eba66">
