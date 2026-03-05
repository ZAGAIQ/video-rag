# Video RAG MVP

A Minimum Viable Product (MVP) for a Video Retrieval-Augmented Generation (RAG) system. This application allows users to search for specific scenes within a video clip using natural language queries (in Russian), based on semantic meaning, actions, and emotions rather than just relying on generic metadata.

## Overview

The system processes video files to detect distinct scenes, extracts keyframes and audio, transcribes the dialogue (using Whisper), and generates comprehensive scene summaries using a Vision-Language Model (VLM). These multimodal insights are embedded into a vector database (ChromaDB) to enable fast, highly-relevant semantic search.

### Key Features
- **Automated Scene Detection**: Uses `PySceneDetect` to chop videos into logical, distinct segments.
- **Audio Transcription**: Uses `OpenAI Whisper` to convert speech to text for each scene.
- **Multimodal Summarization**: Uses an LLM/VLM (via OpenRouter) to combine the keyframe and internal dialogue to create a deep, contextual summary.
- **Multilingual Semantic Search**: Uses `HuggingFace` embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) to execute high-quality vector queries in Russian and English.
- **Interactive UI**: A built-in Streamlit app provides an intuitive interface to search with natural language, see the AI's explanation for *why* a scene matched, and instantly play the retrieved scene.

## Prerequisites

- **Python 3.10+** (Tested on Python 3.11)
- **FFmpeg**: Required for audio and keyframe extraction. Make sure it's installed and available in your system's PATH.
- OpenRouter API Key (or a compatible OpenAI endpoint) to access the Vision-Language Model (e.g., GPT-4o-mini).

## Setup & Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Environment Variables:
   Create a `.env` file in the root of the project with your API keys:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   VLM_MODEL=openai/gpt-4o-mini
   ```

## Usage

### Step 1: Ingestion Pipeline
Place your target video file in the project's root directory and name it `movie.mp4` (or update `test_video` in `ingest.py`). Then populate the vector database by running:

```bash
python ingest.py
```
*Note: This process may take a while depending on your hardware and video length, as it runs Whisper transcription, VLM queries for summaries, and generates ChromaDB embeddings.*

### Step 2: Search Interface
Start the Streamlit application to search for scenes:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Enter your prompt in Russian (e.g., "двое мужчин обсуждают технологии") and the app will retrieve the most relevant video fragments, choose the best one automatically, and play it directly in the browser!
