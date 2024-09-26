import os
import asyncio
import aiohttp
from celery import Celery
from dotenv import load_dotenv
from supabase import create_client, Client
import yt_dlp
import whisper
import openai
import cv2
from moviepy.editor import VideoFileClip
from moviepy.audio.fx.all import audio_normalize
import logging
import json
from redis import Redis
from celery.exceptions import Ignore

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Celery
app = Celery('video_processor', broker=os.getenv("CELERY_BROKER_URL"))
redis_client = Redis.from_url(os.getenv("CELERY_BROKER_URL")) 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Celery
app.conf.update(
    result_backend=os.getenv("CELERY_RESULT_BACKEND"),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Schedule the task to run every hour
app.conf.beat_schedule = {
    'process-videos-every-10s': {
        'task': 'video_processor.process_video_batch',
        # 'schedule': crontab(minute=0, hour='*'),
        'schedule': 10.0,
    },
}

# Configuration
BATCH_SIZE = 10
MAX_CONCURRENT_DOWNLOADS = 5
MAX_CONCURRENT_PROCESSES = 3

downloads = os.path.join("downloads")
clips_output = os.path.join("clips_output")
os.makedirs(downloads, exist_ok=True)
os.makedirs(clips_output, exist_ok=True)


def fetch_unprocessed_videos():
    logging.info("Creating Supabase instance")
    response = supabase.table('videos').select('id', 'url').eq("processed", False).limit(BATCH_SIZE).execute()
    return response.data

async def download_video(session, video):
    ydl_opts = {
        'outtmpl': os.path.join(downloads, '%(id)s.%(ext)s'),
        'format': 'best[ext=mp4]'
    }
    loop = asyncio.get_event_loop()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = await loop.run_in_executor(None, ydl.extract_info, video['url'], True)
    return os.path.join(downloads, f"{info["id"]}.{info["ext"]}")

async def transcribe_video(video_path):
    logging.info(f"Starting transcription for {video_path}")
    loop = asyncio.get_event_loop()
    model = whisper.load_model("base")
    result = await loop.run_in_executor(None, model.transcribe, video_path)
    return result["text"]

async def identify_insightful_clips(transcript):
    logging.info(f"Identifying insighful clips")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai.api_key}"},
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "Identify the most insightful 1-3 minute segments from this transcript. Return a list of tuples with start and end times in seconds."},
                    {"role": "user", "content": transcript}
                ]
            }
        ) as response:
            result = await response.json()
            return eval(result['choices'][0]['message']['content'])


async def find_insightful_clips(transcript, content_type, min_duration=10, max_duration=30, min_clips=2, max_clips=8):
    logging.info(f"Finding insightul clips")
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
                    "model": "gpt-4o",
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
                    selected_clips = clips[:3]
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


async def identify_content_type(transcript):
    logging.info(f"Identifying video content type")
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


async def process_clip(video_path, start_time, end_time):
    # This function would contain the OpenCV processing logic
    # For brevity, we'll just simulate the processing time
    await asyncio.sleep(5)
    return f"processed_{os.path.basename(video_path)}"

def upload_to_supabase(file_path, bucket):
    logging.info("Uploading audio to supabase")
    with open(file_path, 'rb') as f:
        file_content = f.read()
        file_name = os.path.basename(file_path)
        supabase.storage.from_(bucket).upload(file_name, file_content, file_options={"content-type": "audio/mpeg"})
    return f"{bucket}/{file_name}"

async def process_single_video(video):
    logging.info("Processing single video")
    try:
        async with aiohttp.ClientSession() as session:
            video_path = await download_video(session, video)
        
        transcript = await transcribe_video(video_path)
        content_type = await identify_content_type(transcript)
        clips_info = await find_insightful_clips(transcript, content_type)
        
        clip_tasks = []
        for i, clip in enumerate(clips_info):
            logging.info(f"Extracting and enhancing clip {i+1} for {video_path}")
            clip_path = os.path.join(clips_output, f"{video['id']}_clip_{i+1}.mp4")
            await extract_and_enhance_clip(video_path, clip['start'], clip['end'], clip_path)
            clip_url = upload_to_supabase(clip_path, 'clips')
            clip_info = {
                'original_youtube_url': video['url'],
                'clip_url': clip_url,
                'start': clip['start'],
                'end': clip['end'],
                'transcript': transcript[clip['start']:clip['end']],
            }
            clip_tasks.append(clip_info)
            supabase.table('clips').insert(clip_info).execute()
        
        # await asyncio.gather(*clip_tasks)
        
        upload_to_supabase(video_path, 'source_videos')
        
        supabase.table('videos').update({'processed': True}).eq('id', video['id']).execute()
        
        # Clean up local files
        os.remove(video_path)
        for clip in os.listdir('clips_output'):
            os.remove(f'clips_output/{clip}')
        
    except Exception as e:
        print(f"Error processing video {video['id']}: {str(e)}")
        supabase.table('videos').update({'processed': False, 'error': str(e)}).eq('id', video['id']).execute()


def auto_reframe(clip, target_aspect_ratio):
    logging.info("reframing videos")
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
    

async def extract_and_enhance_clip(video_path, start_time, end_time, output_path):
    logging.info(f"Extracting and enhancing clip for {video_path} and {output_path}")
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

    return f"processed_{os.path.basename(video_path)}"

@app.task(bind=True)
def process_video_batch(self):
    logging.info("Attempting to acquire lock for video batch processing")
    lock = redis_client.lock("video_batch_processing_lock", timeout=600)  # 10 minutes timeout

    try:
        have_lock = lock.acquire(blocking=False)
        if have_lock:
            logging.info("Lock acquired, starting video batch processing")
            _process_video_batch()
        else:
            logging.info("Another task is already processing. Skipping this execution.")
            raise Ignore()
    finally:
        if have_lock:
            lock.release()
            logging.info("Lock released after video batch processing")

def _process_video_batch():
    logging.info("Processing Video batch")
    async def async_batch_process():
        videos = fetch_unprocessed_videos()
        if not videos:
            print("No videos to process.")
            return

        # Create semaphores to limit concurrent operations
        download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        process_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSES)

        async def process_with_semaphores(video):
            async with download_semaphore:
                async with process_semaphore:
                    await process_single_video(video)

        tasks = [process_with_semaphores(video) for video in videos]
        await asyncio.gather(*tasks)

        print(f"Batch processing completed for {len(videos)} videos.")

    asyncio.run(async_batch_process())

if __name__ == '__main__':
    app.start()