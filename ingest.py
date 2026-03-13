import os
import subprocess
import tempfile
import base64
from typing import List, Tuple
from dotenv import load_dotenv

import gigaam
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
VLM_MODEL = os.getenv("VLM_MODEL", "google/gemini-2.5-flash-lite")

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

def summarize_scene(llm: ChatOpenAI, transcript: str, image_path: str, plot_text: str) -> str:
    """Use VLM to summarize the scene based on keyframe, transcript and plot."""
    base64_image = encode_image(image_path)
    
    prompt = f"""Вот глобальный сюжет фильма:
{plot_text}

Вот транскрипт текущей сцены:
{transcript if transcript.strip() else '[Диалоги отсутствуют]'}

Опиши подробно, что происходит на этом кадре, кто участвует и какова мотивация героев именно в этот момент. Обязательно свяжи происходящее в сцене с глобальным сюжетом.\n
Важно: если в сцене есть диалоги, то они должны быть связаны с глобальным сюжетом фильма.
Также подробно опиши саму сцену, что находится в кадре"""
    
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
    
    print("Loading GigaAM v2 model...")
    gigaam_model = gigaam.load_model("v2_rnnt")
    
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
    
    print("Reading plot.txt...")
    plot_text = ""
    if os.path.exists("plot.txt"):
        with open("plot.txt", "r", encoding="utf-8") as f:
            plot_text = f.read()
    else:
        print("plot.txt not found, using empty plot context.")

    print("\n--- Phase 1: Extracting Audio and Transcribing all scenes ---")
    transcripts = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "scene.mp3")
            extract_audio(video_path, start_time, end_time, audio_path)
            
            transcript = ""
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                transcript = gigaam_model.transcribe(audio_path)
                
            transcripts.append(transcript)
            print(f"Scene {i+1}/{len(scene_list)} Transcript: {transcript if transcript else '[No speaking]'}")

    print("\n--- Phase 2: Generating Summaries and Saving to ChromaDB ---")
    # Process each scene for summarization
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        middle_time = start_time + (end_time - start_time) / 2
        
        print(f"\nProcessing Scene {i+1}/{len(scene_list)}")
        print(f"Time: {start_time:.2f}s - {end_time:.2f}s")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "keyframe.jpg")
            transcript = transcripts[i]
            
            # Step 1: Extract Keyframe
            extract_keyframe(video_path, middle_time, image_path)
            
            # Step 2: Summarize with VLM
            if not os.path.exists(image_path):
                print("Failed to extract keyframe. Skipping scene.")
                continue
                
            summary = summarize_scene(llm, transcript, image_path, plot_text)
            print(f"Summary: {summary}")
            
            # Step 3: Save to ChromaDB
            final_text = f"Описание сцены:\n{summary}\n\nОригинальный транскрипт:\n{transcript}"
            
            prev_dialogue = transcripts[i-1] if i > 0 else ""
            next_dialogue = transcripts[i+1] if i < len(transcripts) - 1 else ""
            
            metadata = {
                "video_filename": os.path.basename(video_path),
                "start_time": start_time,
                "end_time": end_time,
                "prev_dialogue": prev_dialogue,
                "next_dialogue": next_dialogue
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
