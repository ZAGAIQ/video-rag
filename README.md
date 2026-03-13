# Video RAG MVP

This is a Minimum Viable Product (MVP) for a Video Retrieval-Augmented Generation (RAG) system. It allows users to search for specific scenes in a video based on semantic meaning, actions, emotions, and dialogue, rather than just basic metadata.

## Features

- **Scene Detection**: Automatically segments video into distinct scenes using `PySceneDetect`.
- **Audio Transcription**: Extracts audio and transcribes it using the Russian state-of-the-art `GigaAM` model.
- **Contextual Scene Summarization**: Uses a Vision-Language Model (VLM, e.g., via OpenRouter/Gemini) to generate rich semantic descriptions of keyframes, incorporating global plot context and dialogue.
- **Sliding Window Context**: Captures previous and next scene dialogues to provide a continuous, unbroken context window to the LLM during search.
- **Semantic Search**: Stores scene summaries, transcripts, and metadata in `ChromaDB` using `SentenceTransformers` embeddings for powerful vector search.
- **Interactive UI**: A built-in Streamlit web interface for searching and playing back the exact video scenes you're looking for.

## Prerequisites

- Python 3.10+
- `ffmpeg` installed and added to your system's PATH.

## Installation

1. Clone or download this repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   VLM_MODEL=google/gemini-2.5-flash-lite
   ```

## Usage

### 1. Ingestion (Processing a video)
1. Place a text file named `plot.txt` in the root directory containing the global plot of the video (this significantly improves scene summarization).
2. Place your target video file in the root directory and name it `movie.mp4` (or update the filename in `ingest.py` / Streamlit sidebar).
3. Run the ingestion script to process the video, segment scenes, transcribe audio, summarize, and build the ChromaDB vector store:
   ```bash
   python ingest.py
   ```
   *Note: This process can take a significant amount of time depending on the length of the video and your hardware.*

### 2. Search and UI
1. Once ingestion is complete, start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to the provided local URL (usually `http://localhost:8501`).
3. Enter your search query (e.g., "сцена под дождем, где обсуждают план") into the search bar.
4. The system will retrieve the most relevant scenes, provide an AI-generated explanation of why they match, and allow you to play the specific scene directly in the app.

## Project Structure

- `app.py`: Streamlit frontend for searching and playing video scenes.
- `ingest.py`: Backend script for processing video, generating transcripts/summaries, and populating the vector database.
- `requirements.txt`: List of Python dependencies.
- `plot.txt`: (Create yourself) The global plot of the video for contextual summarization.
- `chroma_db/`: Directory where the vector database is stored (created automatically during ingestion).
