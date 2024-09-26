import os
import ssl
from dotenv import load_dotenv
import certifi
from openai import OpenAI
import torch
import whisper
from moviepy.editor import VideoFileClip
from moviepy.audio.fx.all import audio_normalize
import cv2
import logging
import psutil
import multiprocessing
from supabase import create_client, Client
import yt_dlp
import pickle
import json
import numpy as np
import subprocess
import mimetypes

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load environment variables and set up SSL
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['SSL_CERT_FILE'] = certifi.where()

# Initialize clients and models
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up directories
base_dir = os.path.expanduser("~/Downloads/madvision/magic-local")
videos_output = os.path.join(base_dir, "videos_output")
clips_output = os.path.join(base_dir, "clips_output")
os.makedirs(videos_output, exist_ok=True)
os.makedirs(clips_output, exist_ok=True)

# Supabase storage buckets
SOURCE_VIDEOS_BUCKET = "source_videos"
CLIPS_BUCKET = "clips2"

# yt-dlp options
ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': os.path.join(videos_output, '%(id)s.%(ext)s'),
    'noplaylist': True,
    'keepvideo': True,
    'keepaudio': True,
    'quiet': True,
    'no_warnings': True,
    'progress_hooks': [lambda d: logging.info(f"Download status: {d['status']} for {d['filename']}")],
}

# Create a queue to manage the flow of processed videos
from queue import Queue
processed_queue = Queue()

def log_resource_usage():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    logging.info(f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_percent}%")

def get_optimal_concurrency():
    return min(multiprocessing.cpu_count(), 8)  # Up to 8 concurrent processes

def download_video(url):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            logging.info(f"Starting download of video from URL: {url}")
            info = ydl.extract_info(url, download=True)
            video_path = os.path.join(videos_output, f"{info['id']}.mp4")
            logging.info(f"Download completed: {video_path}")
            return video_path, info['id'], info
        except Exception as e:
            logging.error(f"Error downloading {url}: {str(e)}")
            return None, None, None

def upload_to_supabase_storage(file_path, bucket):
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        file_name = os.path.basename(file_path)
        storage_path = f"{bucket}/{file_name}"
        
        # Set the correct content type
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = 'application/octet-stream'
        
        response = supabase.storage.from_(bucket).upload(
            storage_path, 
            file_data, 
            file_options={"content-type": content_type}
        )
        if 'error' in response:
            logging.error(f"Error uploading {file_path} to Supabase storage: {response['error']}")
            return None
        else:
            logging.info(f"Uploaded {file_path} to Supabase storage at {storage_path}")
            return storage_path
    except Exception as e:
        logging.error(f"Error uploading {file_path} to Supabase storage: {e}")
        return None

def verify_supabase_upload(file_path, bucket):
    try:
        file_name = os.path.basename(file_path)
        storage_path = f"{bucket}/{file_name}"
        
        # Check if the file exists in Supabase storage
        res = supabase.storage.from_(bucket).list(storage_path)
        if res and len(res) > 0:
            logging.info(f"Verified upload of {file_path} to Supabase storage at {storage_path}")
            return True
        else:
            logging.error(f"Failed to verify upload of {file_path} to Supabase storage")
            return False
    except Exception as e:
        logging.error(f"Error verifying upload of {file_path} to Supabase storage: {e}")
        return False

def process_video(video_path, video_id):
    try:
        logging.info(f"Starting transcription for {video_path}")
        transcription = whisper_model.transcribe(video_path)

        logging.info(f"Identifying content type for {video_path}")
        content_type = identify_content_type(transcription['text'])
        logging.info(f"Identified content type: {content_type}")

        # Get video duration
        with VideoFileClip(video_path) as video:
            video_duration = video.duration
        logging.info(f"Video duration: {video_duration} seconds")

        logging.info(f"Finding insightful clips for {video_path}")
        clips = find_insightful_clips(transcription['text'], content_type, video_duration=video_duration)
        logging.info(f"Found {len(clips)} clips")

        processed_clips = []
        for i, clip in enumerate(clips):
            logging.info(f"Extracting and enhancing clip {i+1} for {video_path}")
            output_path = os.path.join(clips_output, f"{video_id}_clip_{i+1}.mp4")
            extract_and_enhance_clip(video_path, clip['start'], clip['end'], output_path)
            processed_clips.append({
                'path': output_path,
                'summary': clip['summary'],
                'start': clip['start'],
                'end': clip['end'],
                'relevance_score': clip['relevance_score']
            })

        logging.info(f"Processed {len(processed_clips)} clips for {video_path}")
        return processed_clips
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")
        return []

def identify_content_type(transcript):
    prompt = f"Analyze the following transcript and identify the type of content (e.g., podcast, interview, educational video, etc.):\n\n{transcript[:1000]}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-turbo" if you prefer
            messages=[
                {"role": "system", "content": "You are an AI that identifies types of video content."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        return content.strip()
    except Exception as e:
        logging.error(f"Error in identify_content_type: {e}")
        return "Unknown"

def find_insightful_clips(transcript, content_type, min_duration=60, max_duration=180, min_clips=3, max_clips=10, video_duration=0):
    prompt = f"""
Analyze the following {content_type} transcript and identify between {min_clips} and {max_clips} of the most valuable, engaging, and interesting sections. Focus on selecting unique segments with natural start and end points, similar to YouTube chapters. For each section:

1. Ensure it starts and ends at natural breaks in speech, preferably at the beginning of a new thought or topic.
2. Choose sections that can stand alone as engaging content.
3. Look for sections that contain any of the following:
   - Key insights or unique perspectives
   - Interesting facts or ideas
   - Practical advice or information
   - Emotionally engaging or inspiring moments
   - Points that might spark curiosity or discussion
4. Pay attention to the start and end of each clip:
   - Start the clip with a complete sentence or thought that introduces the topic.
   - End the clip with a concluding statement or a natural pause in the conversation.

IMPORTANT RULES:
- Do NOT include any clips that contain introductions or previews of the content.
- Ensure that clips do not overlap with each other.
- Prioritize uniqueness and engagement in your selection.

Return a JSON array of clips, where each clip is an object with 'start' and 'end' times in seconds, a 'summary' field, and a 'relevance_score' field (0-100).
The 'relevance_score' should indicate how engaging or valuable the clip is, with 100 being the highest quality.
Ensure each clip is between {min_duration} and {max_duration} seconds long.
Do not exceed the video's duration of {video_duration} seconds.
Vary the clip lengths within the allowed range for diversity.

Transcript:
{transcript}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-turbo" if you prefer
            messages=[
                {"role": "system", "content": "You are an AI assistant that analyzes video transcripts to find engaging and potentially valuable content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        content = response.choices[0].message.content
        logging.info(f"API Response: {content}")

        try:
            # Strip out code block markers if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            
            clips = json.loads(content)
            # Sort clips by start time to ensure non-overlapping
            clips.sort(key=lambda x: x['start'])
            
            # Remove overlapping clips, prioritizing higher relevance scores
            non_overlapping_clips = []
            for clip in clips:
                if clip['end'] > video_duration:
                    logging.warning(f"Clip end time {clip['end']} exceeds video duration {video_duration}. Adjusting end time.")
                    clip['end'] = video_duration
                if not non_overlapping_clips or clip['start'] >= non_overlapping_clips[-1]['end']:
                    non_overlapping_clips.append(clip)
                elif clip['relevance_score'] > non_overlapping_clips[-1]['relevance_score']:
                    non_overlapping_clips[-1] = clip
            
            # Sort final clips by relevance_score
            non_overlapping_clips.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Select top clips, ensuring we have at least min_clips and at most max_clips
            selected_clips = non_overlapping_clips[:max(min_clips, min(len(non_overlapping_clips), max_clips))]
            
            # If less than min_clips, attempt to add more clips
            if len(selected_clips) < min_clips:
                additional_clips = non_overlapping_clips[min_clips:max_clips]
                selected_clips.extend(additional_clips[:min_clips - len(selected_clips)])
                selected_clips = selected_clips[:max_clips]

            logging.info(f"Selected {len(selected_clips)} non-overlapping clips")
            return selected_clips
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error: {e}")
            logging.error(f"Problematic content: {content}")
            return []
    except Exception as e:
        logging.error(f"Error in find_insightful_clips: {e}")
        return []

def auto_reframe(clip, target_aspect_ratio=1.0):
    def detect_faces(frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    w, h = clip.size
    target_w = int(h * target_aspect_ratio)
    
    # Analyze multiple frames to detect faces
    num_samples = 5
    face_positions = []
    for t in np.linspace(0, clip.duration, num_samples):
        frame = clip.get_frame(t)
        faces = detect_faces(frame)
        if len(faces) > 0:
            for (x, y, fw, fh) in faces:
                face_center = x + fw // 2
                face_positions.append(face_center)
    
    if face_positions:
        # Calculate the average face center
        avg_face_center = int(np.mean(face_positions))
        
        # Determine the crop region
        left = max(0, min(w - target_w, avg_face_center - target_w // 2))
        right = left + target_w
        
        # Adjust if the crop goes out of bounds
        if right > w:
            right = w
            left = right - target_w
    else:
        # If no faces detected, default to center crop
        left = (w - target_w) // 2
        right = left + target_w

    # Crop without resizing
    cropped_clip = clip.crop(x1=left, y1=0, x2=right, y2=h)
    
    return cropped_clip

def extract_and_enhance_clip(video_path, start_time, end_time, output_path):
    clip = None
    try:
        # Extract the clip
        clip = VideoFileClip(video_path).subclip(start_time, end_time)

        # Auto reframe clip without resizing
        clip = auto_reframe(clip)

        # Add fade out
        clip = clip.fadeout(0.5)

        # Normalize the audio
        clip = clip.fx(audio_normalize)

        # Write the final clip with improved quality (lower CRF value)
        clip.write_videofile(output_path,
                             codec="libx264",
                             audio_codec="aac",
                             logger=None,
                             ffmpeg_params=["-pix_fmt", "yuv420p", "-g", "1", "-crf", "20"])

        logging.info(f"Clip extracted and enhanced successfully: {output_path}")
    except Exception as e:
        logging.error(f"Error extracting and enhancing clip: {e}")
    finally:
        if clip:
            clip.close()

# Add this function to fetch video URLs from Supabase
def get_video_urls():
    try:
        response = supabase.table('new_videos').select('id, video_url').in_('id', [5, 25, 50, 75, 100, 950]).execute()
        data = response.data
        if not data:
            logging.info("No video records found for the specified IDs.")
            return []
        return [(item['id'], item['video_url']) for item in data]
    except Exception as e:
        logging.error(f"Exception while fetching video URLs: {e}")
        return []

# Update the process_batch function
def process_batch():
    video_data = get_video_urls()
    for video_id, url in video_data:
        logging.info(f"Processing Video ID {video_id}: {url}")
        video_path, yt_video_id, video_info = download_video(url)
        if video_path:
            clips = process_video(video_path, yt_video_id)
            if clips:
                processed_queue.put((url, video_id, yt_video_id, video_path, video_info, clips))
            else:
                logging.warning(f"No clips generated for Video ID {video_id}: {video_path}")
        else:
            logging.error(f"Failed to download video from URL: {url}")

# Modify the upload_to_supabase function
def upload_to_supabase(url, video_id, yt_video_id, video_path, video_info, clips):
    try:
        # Upload the source video to Supabase storage
        video_storage_path = upload_to_supabase_storage(video_path, SOURCE_VIDEOS_BUCKET)
        if not video_storage_path:
            logging.error(f"Failed to upload source video {video_path} to Supabase")
            return

        # Verify the upload
        if not verify_supabase_upload(video_storage_path, SOURCE_VIDEOS_BUCKET):
            logging.error(f"Verification failed for uploaded video {video_path}")
            return

        # Update the videos table
        video_data = {
            "id": video_id,
            "filename": os.path.basename(video_storage_path),
            "storage_path": video_storage_path,
            "url": url,
            "title": video_info.get('title', ''),
            "description": video_info.get('description', ''),
            "channel": video_info.get('uploader', ''),
            "published_at": video_info.get('upload_date', ''),
        }
        supabase.table('videos').upsert(video_data).execute()

        # Upload clips to Supabase storage and update clips table
        for clip in clips:
            clip_storage_path = upload_to_supabase_storage(clip['path'], CLIPS_BUCKET)
            if clip_storage_path:
                if not verify_supabase_upload(clip_storage_path, CLIPS_BUCKET):
                    logging.error(f"Verification failed for uploaded clip {clip['path']}")
                    continue
                clip_data = {
                    "file_path": clip_storage_path,
                    "original_youtube_url": url,
                    "summary": clip['summary'],
                    "start_time": clip['start'],
                    "end_time": clip['end'],
                    "relevance_score": clip['relevance_score'],
                    "video_id": video_id,
                }
                supabase.table("clips").insert(clip_data).execute()
            else:
                logging.error(f"Failed to upload clip {clip['path']} to Supabase")

        # Delete local files after successful upload and verification
        os.remove(video_path)
        logging.info(f"Deleted local source video: {video_path}")
        for clip in clips:
            os.remove(clip['path'])
            logging.info(f"Deleted local clip: {clip['path']}")

        logging.info(f"Completed processing and uploading for Video ID {video_id}")
    except Exception as e:
        logging.error(f"Error uploading to Supabase: {e}")

if __name__ == "__main__":
    main()

# Add this function to fetch video URLs from Supabase
def get_video_urls():
    try:
        response = supabase.table('new_videos').select('id, video_url').in_('id', [5, 25, 50, 75, 100, 950]).execute()
        data = response.data
        if not data:
            logging.info("No video records found for the specified IDs.")
            return []
        return [(item['id'], item['video_url']) for item in data]
    except Exception as e:
        logging.error(f"Exception while fetching video URLs: {e}")
        return []

# Update the process_batch function
def process_batch():
    video_data = get_video_urls()
    for video_id, url in video_data:
        logging.info(f"Processing Video ID {video_id}: {url}")
        video_path, yt_video_id, video_info = download_video(url)
        if video_path:
            clips = process_video(video_path, yt_video_id)
            if clips:
                processed_queue.put((url, video_id, yt_video_id, video_path, video_info, clips))
            else:
                logging.warning(f"No clips generated for Video ID {video_id}: {video_path}")
        else:
            logging.error(f"Failed to download video from URL: {url}")

# Modify the upload_to_supabase function
def upload_to_supabase(url, video_id, yt_video_id, video_path, video_info, clips):
    try:
        # Upload the source video to Supabase storage
        video_storage_path = upload_to_supabase_storage(video_path, SOURCE_VIDEOS_BUCKET)
        if not video_storage_path:
            logging.error(f"Failed to upload source video {video_path} to Supabase")
            return

        # Verify the upload
        if not verify_supabase_upload(video_storage_path, SOURCE_VIDEOS_BUCKET):
            logging.error(f"Verification failed for uploaded video {video_path}")
            return

        # Update the videos table
        video_data = {
            "id": video_id,
            "filename": os.path.basename(video_storage_path),
            "storage_path": video_storage_path,
            "url": url,
            "title": video_info.get('title', ''),
            "description": video_info.get('description', ''),
            "channel": video_info.get('uploader', ''),
            "published_at": video_info.get('upload_date', ''),
        }
        supabase.table('videos').upsert(video_data).execute()

        # Upload clips to Supabase storage and update clips table
        for clip in clips:
            clip_storage_path = upload_to_supabase_storage(clip['path'], CLIPS_BUCKET)
            if clip_storage_path:
                if not verify_supabase_upload(clip_storage_path, CLIPS_BUCKET):
                    logging.error(f"Verification failed for uploaded clip {clip['path']}")
                    continue
                clip_data = {
                    "file_path": clip_storage_path,
                    "original_youtube_url": url,
                    "summary": clip['summary'],
                    "start_time": clip['start'],
                    "end_time": clip['end'],
                    "relevance_score": clip['relevance_score'],
                    "video_id": video_id,
                }
                supabase.table("clips").insert(clip_data).execute()
            else:
                logging.error(f"Failed to upload clip {clip['path']} to Supabase")

        # Delete local files after successful upload and verification
        os.remove(video_path)
        logging.info(f"Deleted local source video: {video_path}")
        for clip in clips:
            os.remove(clip['path'])
            logging.info(f"Deleted local clip: {clip['path']}")

        logging.info(f"Completed processing and uploading for Video ID {video_id}")
    except Exception as e:
        logging.error(f"Error uploading to Supabase: {e}")

if __name__ == "__main__":
    main()