import os
import subprocess
import tempfile
import base64
from typing import List, Tuple
from dotenv import load_dotenv

import whisper
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
VLM_MODEL = os.getenv("VLM_MODEL", "openai/gpt-4o-mini")

if not OPENROUTER_API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY in your .env file")

def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_audio(video_path: str, start_time: float, end_time: float, output_path: str):
    """Extract audio from a specific segment using ffmpeg."""
    duration = end_time - start_time
    command = [
        "ffmpeg",
        "-y", # Overwrite output
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-vn", # Disable video
        "-acodec", "libmp3lame",
        "-q:a", "2",
        output_path
    ]
    # Run ffmpeg, suppress output for cleaner logs
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_keyframe(video_path: str, timestamp: float, output_path: str):
    """Extract a single frame at a specific timestamp."""
    command = [
        "ffmpeg",
        "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def summarize_scene(llm: ChatOpenAI, transcript: str, image_path: str) -> str:
    """Use VLM to summarize the scene based on keyframe and transcript."""
    base64_image = encode_image(image_path)
    
    prompt = f"""
    Опиши, что происходит в этой сцене, кто в ней участвует, какие эмоции и мотивы 
    прослеживаются на основе изображения и текста диалога:
    
    Текст диалога/аудио:
    {transcript if transcript.strip() else '[Диалоги отсутствуют]'}
    """
    
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )
    
    try:
        response = llm.invoke([msg])
        return response.content
    except Exception as e:
        print(f"Error during VLM summarization: {e}")
        return "Не удалось сгенерировать описание для этой сцены из-за ошибки API."

def process_video(video_path: str):
    print(f"Opening video: {video_path}")
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    
    print("Detecting scenes. This might take a while...")
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    print(f"Detected {len(scene_list)} scenes.")
    
    print("Loading Whisper model (base)...")
    whisper_model = whisper.load_model("base")
    
    print("Initializing VLM Client...")
    llm = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=VLM_MODEL,
        max_tokens=600
    )
    
    print("Initializing ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings, 
        collection_name="video_scenes"
    )
    
    # Process each scene
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        middle_time = start_time + (end_time - start_time) / 2
        
        print(f"\n--- Processing Scene {i+1}/{len(scene_list)} ---")
        print(f"Time: {start_time:.2f}s - {end_time:.2f}s")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "scene.mp3")
            image_path = os.path.join(temp_dir, "keyframe.jpg")
            
            # Step 1: Extract Audio
            extract_audio(video_path, start_time, end_time, audio_path)
            
            # Step 2: Transcribe
            transcript = ""
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000: # Ensure valid audio
                result = whisper_model.transcribe(audio_path, fp16=False)
                transcript = result["text"].strip()
                
            print(f"Transcript: {transcript if transcript else '[No speaking]'}")
            
            # Step 3: Extract Keyframe
            extract_keyframe(video_path, middle_time, image_path)
            
            # Step 4: Summarize with VLM
            if not os.path.exists(image_path):
                print("Failed to extract keyframe. Skipping scene.")
                continue
                
            summary = summarize_scene(llm, transcript, image_path)
            print(f"Summary: {summary}")
            
            # Step 5: Save to ChromaDB
            final_text = f"Описание сцены:\n{summary}\n\nОригинальный транскрипт:\n{transcript}"
            
            metadata = {
                "video_filename": os.path.basename(video_path),
                "start_time": start_time,
                "end_time": end_time
            }
            
            vectorstore.add_texts(
                texts=[final_text],
                metadatas=[metadata]
            )
            print("Successfully added to ChromaDB.")
            
    print("\nProcessing complete! Vectors are persisted in ./chroma_db")

if __name__ == "__main__":
    # Ensure user has installed all external CLI stuff like ffmpeg.
    # We will look for a default "movie.mp4" or ask user.
    test_video = "movie.mp4"
    if not os.path.exists(test_video):
        print(f"Please place a video file named '{test_video}' in the current directory.")
        exit(1)
        
    process_video(test_video)
