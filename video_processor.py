import os
from typing import List, Dict, Tuple, Optional
import yt_dlp
import whisper

# Remove old: import openai
from openai import AsyncOpenAI  # <-- New import
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from supabase import create_client, Client
from datetime import datetime
import json
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
from scipy.signal import find_peaks
import torch
from transformers import pipeline
import redis
from PIL import Image
import io


###########################################################
# DATA CLASSES
###########################################################
@dataclass
class VideoClip:
    start_time: float
    end_time: float
    transcript: str
    confidence_score: float
    keywords: List[str]
    engagement_metrics: Dict[str, float]
    visual_quality_score: float
    audio_quality_score: float


@dataclass
class ProcessedClip:
    clip_id: str
    source_video_id: str
    start_time: float
    end_time: float
    storage_path: str
    transcript: str
    metadata: Dict


###########################################################
# VIDEO PROCESSOR
###########################################################
class VideoProcessor:
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        openai_key: str,
        storage_bucket: str = "clips",
        redis_url: Optional[str] = None,
    ):
        # Supabase setup
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.storage_bucket = storage_bucket

        # Whisper model (with GPU if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model("base", device=device)

        # Initialize the new async OpenAI client
        self.openai_client = AsyncOpenAI(api_key=openai_key)

        # Logger
        self.logger = logging.getLogger(__name__)

        # Transformers pipelines for additional analysis
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", device=0 if torch.cuda.is_available() else -1
        )
        self.feature_extractor = pipeline(
            "feature-extraction", device=0 if torch.cuda.is_available() else -1
        )

        # Redis caching (optional)
        self.redis_client = redis.from_url(redis_url) if redis_url else None

        # Additional device reference (for encoding decisions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # OpenCV optimizations
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)

    ###########################################################
    # PUBLIC METHOD: PROCESS YOUTUBE VIDEO
    ###########################################################
    async def process_youtube_video(self, video_url: str) -> str:
        """Process a single YouTube video with an asynchronous OpenAI client."""
        cache_key = f"video:{video_url}"

        # 1) Check if result is already cached
        if self.redis_client:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                self.logger.info(f"Cache hit for {video_url}")
                return json.loads(cached_result)["video_id"]

        # 2) Download video with optimized settings (runs in thread pool)
        video_info = await self._download_video_optimized(video_url)
        video_id = video_info["id"]

        # 3) Transcribe & analyze audio, plus extract video features, in parallel
        async with asyncio.TaskGroup() as tg:
            transcript_task = tg.create_task(
                self._transcribe_and_analyze(video_info["filepath"])
            )
            video_features_task = tg.create_task(
                self._extract_video_features(video_info["filepath"])
            )

        transcript_result = transcript_task.result()
        video_features = video_features_task.result()

        # 4) Identify best clips with sophisticated criteria (using GPT-4o, for example)
        clips = await self._identify_clips_enhanced(transcript_result, video_features)

        # 5) Process clips in parallel
        processed_clips = await self._parallel_clip_processing(
            video_info["filepath"], clips, video_id
        )

        # 6) Store all metadata in Supabase
        await self._store_enhanced_metadata(
            video_id, processed_clips, transcript_result, video_features
        )

        # 7) Cache the final result
        if self.redis_client:
            self.redis_client.setex(
                cache_key, 3600, json.dumps({"video_id": video_id})  # 1 hour cache
            )

        return video_id

    ###########################################################
    # DOWNLOAD VIDEO (OPTIMIZED)
    ###########################################################
    async def _download_video_optimized(self, url: str) -> Dict:
        """Optimized video download with smart format selection and concurrency."""
        ydl_opts = {
            "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "outtmpl": "/tmp/%(id)s.%(ext)s",
            "concurrent_fragment_downloads": 3,
            "postprocessor_args": {"preset": "faster"},
        }

        with ThreadPoolExecutor() as executor:
            info = await asyncio.get_event_loop().run_in_executor(
                executor, self._download_with_ydl, url, ydl_opts
            )
        return info

    def _download_with_ydl(self, url: str, opts: Dict) -> Dict:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
            return {"id": info["id"], "filepath": filepath}

    ###########################################################
    # TRANSCRIBE & ANALYZE AUDIO
    ###########################################################
    async def _transcribe_and_analyze(self, filepath: str) -> Dict:
        """Enhanced transcription with Whisper plus audio quality and sentiment analysis."""
        # 1) Whisper transcription
        result = self.whisper_model.transcribe(filepath)

        # 2) Audio quality analysis
        audio_features = await self._analyze_audio_quality(filepath)

        # 3) Perform sentiment analysis on each segment’s text
        segments_text = [
            seg["text"] for seg in result.get("segments", []) if "text" in seg
        ]
        sentiments = self.sentiment_analyzer(segments_text) if segments_text else []

        return {
            "transcript": result,
            "audio_features": audio_features,
            "sentiments": sentiments,
        }

    async def _analyze_audio_quality(self, filepath: str) -> Dict:
        with VideoFileClip(filepath) as video:
            audio = video.audio
            features = {
                "sample_rate": audio.fps,
                "duration": audio.duration,
            }
            return features

    ###########################################################
    # EXTRACT VIDEO FEATURES
    ###########################################################
    async def _extract_video_features(self, filepath: str) -> Dict:
        features = {"frame_qualities": [], "scene_changes": [], "face_detections": []}

        with VideoFileClip(filepath) as video:
            total_frames = int(video.fps * video.duration)
            sample_rate = max(1, total_frames // 100)
            frame_index = 0

            for i, frame in enumerate(video.iter_frames()):
                if i % sample_rate == 0:
                    frame_features = await self._analyze_frame(frame)
                    features["frame_qualities"].append(frame_features)
                frame_index += 1

        return features

    async def _analyze_frame(self, frame: np.ndarray) -> Dict:
        img = Image.fromarray(frame)

        quality_metrics = {
            "brightness": float(np.mean(frame)),
            "contrast": float(np.std(frame)),
            "sharpness": float(self._calculate_sharpness(img)),
        }

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        return {"quality": quality_metrics, "face_count": len(faces)}

    def _calculate_sharpness(self, image: Image) -> float:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    ###########################################################
    # IDENTIFY CLIPS (ENHANCED) WITH NEW CLIENT
    ###########################################################
    async def _identify_clips_enhanced(
        self, transcript_result: Dict, video_features: Dict
    ) -> List[VideoClip]:
        """
        Use the new AsyncOpenAI client to call GPT-4o (or gpt-4) with advanced context.
        """
        analysis_prompt = self._create_enhanced_prompt(
            transcript_result, video_features
        )

        # GPT-4 or GPT-4o request using the new async method
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",  # Or "gpt-4", depending on which you have access to
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.7,
        )

        # Parse GPT response into candidate clips
        clip_candidates: List[VideoClip] = self._parse_gpt_response(response)

        # Filter & score
        scored_clips = []
        for clip in clip_candidates:
            total_score = self._calculate_engagement_score(
                clip, transcript_result, video_features
            )
            clip.engagement_metrics["total_score"] = total_score
            if total_score > 0.7:
                scored_clips.append(clip)

        # Sort by final score, pick top 10
        top_clips = sorted(
            scored_clips,
            key=lambda x: x.engagement_metrics["total_score"],
            reverse=True,
        )[:10]

        return top_clips

    def _create_enhanced_prompt(
        self, transcript_result: Dict, video_features: Dict
    ) -> str:
        return f"""
        Analyze this video content to identify the most valuable 1-3 minute segments.
        Consider:
        1. Content Value
        2. Technical Quality
        3. Engagement Factors
        4. Stand-alone Value
        
        Transcript and audio features:
        {json.dumps(transcript_result, indent=2)}
        
        Video quality features:
        {json.dumps(video_features, indent=2)}
        
        For each identified clip, provide:
        - Start/end timestamps
        - Short transcript excerpt
        - Confidence score (0-1)
        - Key topics/keywords
        - Value proposition
        - Expected engagement metrics
        """

    def _parse_gpt_response(self, response) -> List[VideoClip]:
        """
        Parse the GPT-4/4o JSON/text response into a list of VideoClip objects.
        """
        clips = []
        try:
            content_str = response["choices"][0]["message"]["content"]
            data = json.loads(content_str)  # expecting a JSON array
            for item in data:
                clips.append(
                    VideoClip(
                        start_time=float(item.get("start_time", 0.0)),
                        end_time=float(item.get("end_time", 0.0)),
                        transcript=item.get("transcript", ""),
                        confidence_score=float(item.get("confidence_score", 0.0)),
                        keywords=item.get("keywords", []),
                        engagement_metrics=item.get("engagement_metrics", {}),
                        visual_quality_score=0.0,
                        audio_quality_score=0.0,
                    )
                )
        except Exception as e:
            self.logger.error(f"Failed to parse GPT-4 response: {e}")
        return clips

    ###########################################################
    # SCORING / ENGAGEMENT ANALYSIS
    ###########################################################
    def _calculate_engagement_score(
        self, clip: VideoClip, transcript_result: Dict, video_features: Dict
    ) -> float:
        content_score = self._analyze_content_quality(clip, transcript_result)
        technical_score = self._analyze_technical_quality(clip, video_features)
        engagement_score = self._analyze_engagement_potential(clip, transcript_result)

        weights = {"content": 0.5, "technical": 0.3, "engagement": 0.2}
        total_score = (
            content_score * weights["content"]
            + technical_score * weights["technical"]
            + engagement_score * weights["engagement"]
        )
        return total_score

    def _analyze_content_quality(
        self, clip: VideoClip, transcript_result: Dict
    ) -> float:
        length_factor = min(len(clip.transcript) / 200, 1.0)
        conf_factor = clip.confidence_score
        return (length_factor + conf_factor) / 2.0

    def _analyze_technical_quality(
        self, clip: VideoClip, video_features: Dict
    ) -> float:
        frames = video_features.get("frame_qualities", [])
        if not frames:
            return 0.5
        avg_brightness = np.mean([f["quality"]["brightness"] for f in frames])
        brightness_score = min(avg_brightness / 255.0, 1.0)
        return brightness_score

    def _analyze_engagement_potential(
        self, clip: VideoClip, transcript_result: Dict
    ) -> float:
        sentiments = transcript_result.get("sentiments", [])
        if not sentiments:
            return 0.5
        positive_count = sum(1 for s in sentiments if s["label"] == "POSITIVE")
        score = positive_count / len(sentiments)
        return score

    ###########################################################
    # PROCESS MULTIPLE CLIPS IN PARALLEL
    ###########################################################
    async def _parallel_clip_processing(
        self, source_filepath: str, clips: List[VideoClip], video_id: str
    ) -> List[ProcessedClip]:
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(
                    self._process_clip_enhanced(source_filepath, clip, video_id)
                )
                for clip in clips
            ]
        return [task.result() for task in tasks]

    ###########################################################
    # PROCESS A SINGLE CLIP (ENHANCED)
    ###########################################################
    async def _process_clip_enhanced(
        self, source_filepath: str, clip: VideoClip, video_id: str
    ) -> ProcessedClip:
        with VideoFileClip(source_filepath) as video:
            sub_clip = video.subclip(clip.start_time, clip.end_time)
            processed_frames = []
            for frame in sub_clip.iter_frames():
                enhanced_frame = await self._enhance_frame(frame)
                processed_frames.append(enhanced_frame)

            clip_id = f"{video_id}_{int(clip.start_time)}"
            output_path = f"/tmp/{clip_id}.mp4"
            encoder = "h264_nvenc" if torch.cuda.is_available() else "libx264"

            fps = sub_clip.fps if sub_clip.fps else 24
            height, width, _ = processed_frames[0].shape

            # OpenCV-based writer for direct control
            fourcc = (
                cv2.VideoWriter_fourcc(*"H264")
                if encoder == "h264_nvenc"
                else cv2.VideoWriter_fourcc(*"mp4v")
            )
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for frm in processed_frames:
                out.write(frm)
            out.release()

            # Upload to Supabase
            storage_path = await self._upload_with_retry(output_path, clip_id)

            metadata = {
                "confidence_score": clip.confidence_score,
                "keywords": clip.keywords,
                "engagement_metrics": clip.engagement_metrics,
                "quality_scores": {
                    "visual": clip.visual_quality_score,
                    "audio": clip.audio_quality_score,
                },
            }

            return ProcessedClip(
                clip_id=clip_id,
                source_video_id=video_id,
                start_time=clip.start_time,
                end_time=clip.end_time,
                storage_path=storage_path,
                transcript=clip.transcript,
                metadata=metadata,
            )

    ###########################################################
    # ENHANCE FRAME (QUALITY IMPROVEMENTS)
    ###########################################################
    async def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_float = frame.astype(np.float32) / 255.0

        # LAB + CLAHE
        lab = cv2.cvtColor(frame_float, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(np.uint8(l * 255)) / 255.0
        enhanced_lab = cv2.merge([l_clahe, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 9.0
        sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)

        # Clip & convert to uint8
        sharpened = np.clip(sharpened, 0, 1)
        return (sharpened * 255).astype(np.uint8)

    ###########################################################
    # UPLOAD WITH RETRY
    ###########################################################
    async def _upload_with_retry(
        self, file_path: str, clip_id: str, max_retries: int = 3
    ) -> str:
        storage_path = f"{self.storage_bucket}/{clip_id}.mp4"
        attempt = 0

        while attempt < max_retries:
            try:
                with open(file_path, "rb") as f:
                    self.supabase.storage.from_(self.storage_bucket).upload(
                        f"{clip_id}.mp4", f
                    )
                return storage_path
            except Exception as e:
                self.logger.error(f"Upload attempt {attempt+1} failed: {e}")
                attempt += 1
                if attempt == max_retries:
                    raise
                await asyncio.sleep(2**attempt)  # exponential backoff

        return storage_path

    ###########################################################
    # STORE METADATA (ENHANCED)
    ###########################################################
    async def _store_enhanced_metadata(
        self,
        video_id: str,
        processed_clips: List[ProcessedClip],
        transcript_result: Dict,
        video_features: Dict,
    ):
        video_data = {
            "video_id": video_id,
            "full_transcript": transcript_result["transcript"].get("text", ""),
            "audio_features": json.dumps(transcript_result.get("audio_features", {})),
            "sentiments": json.dumps(transcript_result.get("sentiments", [])),
            "video_features": json.dumps(video_features),
            "processed_at": datetime.utcnow().isoformat(),
            "clip_count": len(processed_clips),
        }
        self.supabase.table("videos").insert(video_data).execute()

        clip_data = []
        for clip in processed_clips:
            clip_data.append(
                {
                    "clip_id": clip.clip_id,
                    "source_video_id": clip.source_video_id,
                    "start_time": clip.start_time,
                    "end_time": clip.end_time,
                    "storage_path": clip.storage_path,
                    "transcript": clip.transcript,
                    "metadata": json.dumps(clip.metadata),
                }
            )
        self.supabase.table("clips").insert(clip_data).execute()


###########################################################
# PROCESS MULTIPLE VIDEOS (ENTRY POINT)
###########################################################
async def process_videos(urls: List[str], processor: VideoProcessor):
    """Process multiple videos concurrently using the VideoProcessor."""
    tasks = [processor.process_youtube_video(url) for url in urls]
    return await asyncio.gather(*tasks)
