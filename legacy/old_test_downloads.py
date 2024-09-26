import os
import ssl
from dotenv import load_dotenv
import certifi
from openai import OpenAI
import asyncio
import torch
import whisper
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import cv2
from tqdm import tqdm
import logging
import psutil
import multiprocessing
from moviepy.audio.fx.all import audio_normalize
from supabase import create_client, Client
import yt_dlp
import asyncio
import aiohttp

# Load environment variables and set up SSL
load_dotenv()
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['SSL_CERT_FILE'] = certifi.where()

# Create a queue to manage the flow of processed videos
processed_queue = asyncio.Queue()

# Initialize clients and models
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

# yt-dlp options
ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': os.path.join(videos_output, '%(id)s.%(ext)s'),
    'noplaylist': True,
}

# Test URLs
TEST_URLS = [
    "https://www.youtube.com/watch?v=kG5Qb9sr0YQ",
    "https://www.youtube.com/watch?v=bc6uFV9CJGg",
    "https://www.youtube.com/watch?v=6u4JVz7iQTY&t=55s"
]

def log_resource_usage():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    logging.info(f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_percent}%")

def get_optimal_concurrency():
    return min(multiprocessing.cpu_count(), 8)  # Up to 8 concurrent processes

async def download_video(url):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            return os.path.join(videos_output, f"{info['id']}.mp4"), info['id']
        except Exception as e:
            logging.error(f"Error downloading {url}: {str(e)}")
            return None, None

async def process_video(video_path, video_id):
    try:
        logging.info(f"Starting transcription for {video_path}")
        transcription = whisper_model.transcribe(video_path)
        
        logging.info(f"Identifying content type for {video_path}")
        content_type = await identify_content_type(transcription['text'])
        
        logging.info(f"Finding insightful clips for {video_path}")
        clips = await find_insightful_clips(transcription['text'], content_type)
        
        processed_clips = []
        for i, clip in enumerate(clips):
            logging.info(f"Extracting and enhancing clip {i+1} for {video_path}")
            output_path = os.path.join(clips_output, f"{video_id}_clip_{i+1}.mp4")
            await extract_and_enhance_clip(video_path, clip['start'], clip['end'], output_path)
            processed_clips.append({
                'path': output_path,
                'transcript': transcription['text'][clip['start']:clip['end']],
                'start': clip['start'],
                'end': clip['end']
            })
        
        return processed_clips
    except Exception as e:
        logging.error(f"Error in process_video for {video_path}: {str(e)}")
        raise
        
        return processed_clips
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {str(e)}")
        return []

async def identify_content_type(transcript):
    prompt = f"Analyze the following transcript and identify the type of content (e.g., podcast, interview, educational video, etc.):\n\n{transcript[:1000]}"
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are an AI that identifies types of video content."},
                    {"role": "user", "content": prompt}
                ]
            }
        ) as response:
            result = await response.json()
            return result['choices'][0]['message']['content']

def transcribe_with_whisper(video_path):
    model = whisper.load_model("tiny")  # Use the tiny model for faster processing
    result = model.transcribe(video_path, fp16=False)
    return result

# Modify the get_or_create_transcription function to include a force_transcribe parameter
def get_or_create_transcription(video_path, force_transcribe=False):
    cache_path = video_path + '.transcription.pkl'
    if os.path.exists(cache_path) and not force_transcribe:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    transcription = transcribe_with_whisper(video_path)
    with open(cache_path, 'wb') as f:
        pickle.dump(transcription, f)
    return transcription

def auto_reframe(clip, target_aspect_ratio):
    def detect_face(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            return faces[0]
        return None

    w, h = clip.size
    if w / h > target_aspect_ratio:
        new_w = int(h * target_aspect_ratio)
        frame = clip.get_frame(0)
        face = detect_face(frame)
        if face is not None:
            x, y, fw, fh = face
            center_x = x + fw // 2
            x1 = max(0, min(w - new_w, center_x - new_w // 2))
        else:
            x1 = (w - new_w) // 2
        return clip.crop(x1=x1, y1=0, x2=x1+new_w, y2=h)
    else:
        new_h = int(w / target_aspect_ratio)
        y1 = (h - new_h) // 2
        return clip.crop(x1=0, y1=y1, x2=w, y2=y1+new_h)

async def find_insightful_clips(transcript, content_type, min_duration=60, max_duration=180, min_clips=2, max_clips=8):
    prompt = f"""
    Analyze the following {content_type} transcript and identify between {min_clips} and {max_clips} of the most valuable or interesting sections. These sections should have potential to engage viewers, even if they're not all equally insightful. For each section:

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

    IMPORTANT: Always select at least {min_clips} clips, even if they don't seem highly insightful. Choose the best available options.

    Return a JSON array of clips, where each clip is an object with 'start' and 'end' times in seconds, a 'summary' field, and a 'relevance_score' field (0-100).
    The 'relevance_score' should indicate how engaging or valuable the clip is, with 100 being the highest quality.
    Ensure each clip is between {min_duration} and {max_duration} seconds long.
    Vary the clip lengths within the allowed range for diversity.

    Transcript:
    {transcript}
    """

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                json={
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are an AI assistant that analyzes video transcripts to find engaging and potentially valuable content."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.8,
                }
            ) as response:
                result = await response.json()
                content = result['choices'][0]['message']['content']
                logging.info(f"API Response: {content}")
                
                try:
                    clips = json.loads(content)
                    # Sort clips by relevance_score in descending order
                    clips.sort(key=lambda x: x['relevance_score'], reverse=True)
                    
                    # Ensure we have at least min_clips
                    if len(clips) < min_clips:
                        logging.warning(f"Only {len(clips)} clips found. This is less than the minimum of {min_clips}.")
                        # If we have less than min_clips, we'll create additional clips by splitting the transcript
                        total_duration = sum(clip['end'] - clip['start'] for clip in clips)
                        remaining_duration = len(transcript.split()) / 2  # Rough estimate of duration based on word count
                        additional_clips_needed = min_clips - len(clips)
                        chunk_duration = remaining_duration / additional_clips_needed
                        
                        for i in range(additional_clips_needed):
                            start_time = total_duration + (i * chunk_duration)
                            end_time = start_time + chunk_duration
                            clips.append({
                                'start': start_time,
                                'end': end_time,
                                'summary': f"Additional clip {i+1} to meet minimum clip requirement",
                                'relevance_score': 50  # Neutral score for additional clips
                            })
                    
                    # Select top clips, ensuring we have at least min_clips and at most max_clips
                    selected_clips = clips[:max_clips]
                    return selected_clips
                except json.JSONDecodeError as e:
                    logging.error(f"JSON Decode Error: {e}")
                    logging.error(f"Problematic content: {content}")
                    # If JSON parsing fails, create default clips
                    return [
                        {
                            'start': 0,
                            'end': min_duration,
                            'summary': "Default clip 1 due to parsing error",
                            'relevance_score': 50
                        },
                        {
                            'start': min_duration,
                            'end': 2 * min_duration,
                            'summary': "Default clip 2 due to parsing error",
                            'relevance_score': 50
                        }
                    ]
    except Exception as e:
        logging.error(f"Error in API call: {e}")
        # If API call fails, create default clips
        return [
            {
                'start': 0,
                'end': min_duration,
                'summary': "Default clip 1 due to API error",
                'relevance_score': 50
            },
            {
                'start': min_duration,
                'end': 2 * min_duration,
                'summary': "Default clip 2 due to API error",
                'relevance_score': 50
            }
        ]

def detect_sentence_boundaries(audio_path, min_silence_len=500, silence_thresh=-40):
    audio = AudioSegment.from_wav(audio_path)
    non_silent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return [start / 1000 for start, _ in non_silent_ranges]  # Convert to seconds

def refine_clip_boundaries(clip, sentence_boundaries):
    start, end = clip['start'], clip['end']
    refined_start = min(sentence_boundaries, key=lambda x: abs(x - start))
    refined_end = min(sentence_boundaries, key=lambda x: abs(x - end))
    return refined_start, refined_end

async def extract_and_enhance_clip(video_path, start_time, end_time, output_path):
    try:
        # Extract the clip
        clip = VideoFileClip(video_path).subclip(start_time, end_time)
        
        # Auto reframe clip
        clip = auto_reframe(clip, 1)  # For 1:1 aspect ratio
        
        # Add fade in and fade out
        clip = clip.fadein(0.5).fadeout(0.5)
        
        # Normalize the audio
        clip = clip.fx(audio_normalize)
        # Write the final clip
        clip.write_videofile(output_path,
                             codec="libx264",
                             audio_codec="aac",
                             logger=None,
                             ffmpeg_params=["-pix_fmt", "yuv420p"])  # Add this line
        
        logging.info(f"Clip extracted and enhanced successfully: {output_path}")
    except Exception as e:
        logging.error(f"Error extracting and enhancing clip: {e}")
    finally:
        clip.close()  # Ensure the clip is properly closed

async def process_batch(urls):
    for url in urls:
        logging.info(f"Processing URL: {url}")
        video_path, video_id = await download_video(url)
        if video_path:
            try:
                clips = await process_video(video_path, video_id)
                if clips:
                    await processed_queue.put((url, video_id, video_path, clips))
                    
                    # Move clip file deletion here
                    for clip in clips:
                        os.remove(clip['path'])
                
                # Move video file deletion here, after processing
                os.remove(video_path)
            except Exception as e:
                logging.error(f"Error processing video {video_path}: {str(e)}")

async def update_supabase(url, video_id, video_path, clips):
    # Update videos table
    supabase.table('videos').update({"filename": os.path.basename(video_path)}).eq('url', url).execute()
    
    # Update clips table
    for clip in clips:
        clip_data = {
            "file_path": clip['path'],
            "original_youtube_url": url,
            "transcript": clip['transcript'],
        }
        supabase.table("clips").insert(clip_data).execute()

async def ensure_test_urls_in_supabase():
    for url in TEST_URLS:
        result = supabase.table('videos').select('url').eq('url', url).execute()
        if not result.data:
            # If the URL doesn't exist, insert it with minimal required fields
            try:
                supabase.table('videos').insert({
                    "url": url,
                    # Add other required fields here if necessary
                    # For example: "title": "Placeholder Title"
                }).execute()
                logging.info(f"Added URL to Supabase: {url}")
            except Exception as e:
                logging.error(f"Failed to add URL to Supabase: {url}. Error: {str(e)}")
    logging.info("Finished processing test URLs.")

async def main():
    await ensure_test_urls_in_supabase()
    await process_batch(TEST_URLS)
    
    # Process any remaining items in the queue (if needed for future expansion)
    while not processed_queue.empty():
        url, video_id, video_path, clips = await processed_queue.get()
        logging.info(f"Processing queued item: {url}")
    
    log_resource_usage()

if __name__ == "__main__":
    asyncio.run(main())