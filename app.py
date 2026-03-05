import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
VLM_MODEL = os.getenv("VLM_MODEL", "openai/gpt-4o-mini")

st.set_page_config(page_title="Video RAG MVP", layout="wide")

@st.cache_resource
def load_vectorstore():
    """Load ChromaDB with HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings, 
        collection_name="video_scenes"
    )

@st.cache_resource
def load_llm():
    """Load the LLM for chat explanations via OpenRouter."""
    if not OPENROUTER_API_KEY:
        st.error("OPENROUTER_API_KEY is missing. Please check your .env file.")
        st.stop()
        
    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=VLM_MODEL,
        temperature=0.3
    )

def generate_explanation(llm, query: str, context: str) -> str:
    """Generate an explanation of why the scene matches using LLM."""
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=(
            "Ты — киновед и помощник по поиску видео. Пользователь ищет определенную сцену. "
            "На основе предоставленного контекста (описание сцен и диалоги) ответь на запрос "
            "пользователя и объясни, почему именно эта сцена (или сцены) подходит лучше всего.\n\n"
            "Запрос пользователя: {query}\n\n"
            "Контекст найденных сцен:\n{context}\n\n"
            "Твой ответ (будь краток и по делу):"
        )
    )
    prompt = prompt_template.format(query=query, context=context)
    response = llm.invoke(prompt)
    return response.content

st.title("🎬 Video RAG Search MVP")
st.markdown("Ищите фильмы и сцены по смыслу, действиям и эмоциям!")

# Sidebar for config or info
with st.sidebar:
    st.header("Настройки")
    st.info("Это MVP система поиска сцен по смыслу.")
    video_file_path = st.text_input("Путь к видеофайлу (для плеера)", value="movie.mp4")

vectorstore = load_vectorstore()
llm = load_llm()

query = st.text_input("Что вы ищете? (например, 'сцена под дождем, где обсуждают план')")

if query:
    if "last_query" not in st.session_state or st.session_state.last_query != query:
        with st.spinner("Ищем подходящие сцены..."):
            # Retrieve top 3 relevant documents
            results = vectorstore.similarity_search(query, k=3)
            st.session_state.results = results
            st.session_state.last_query = query
            
            if results:
                # Prepare context for the LLM
                context_str = ""
                for i, doc in enumerate(results):
                    context_str += f"--- Сцена {i+1} ---\n"
                    context_str += f"Файл: {doc.metadata.get('video_filename')}\n"
                    context_str += f"Время: {doc.metadata.get('start_time'):.1f}s - {doc.metadata.get('end_time'):.1f}s\n"
                    context_str += f"Содержимое: {doc.page_content}\n\n"
                
                # Generate explanation
                st.session_state.explanation = generate_explanation(llm, query, context_str)
            else:
                st.session_state.explanation = ""

    results = st.session_state.get("results", [])
    explanation = st.session_state.get("explanation", "")
    
    if not results:
        st.warning("Ничего не найдено. Возможно, база данных пуста.")
    else:
        st.subheader("Ответ ИИ")
        st.markdown(explanation)
        
        st.markdown("---")
        st.subheader("Найденные сцены")
        
        # Let the user select which scene to play
        scene_options = [f"Сцена {i+1} ({doc.metadata.get('start_time'):.1f}s - {doc.metadata.get('end_time'):.1f}s)" for i, doc in enumerate(results)]
        
        # Try to parse the best scene from the LLM explanation
        import re
        best_scene_idx = 0
        match = re.search(r'сцена\s+(\d)', explanation.lower()) or re.search(r'сцену\s+(\d)', explanation.lower()) or re.search(r'сцене\s+(\d)', explanation.lower())
        if match:
            parsed_idx = int(match.group(1)) - 1
            if 0 <= parsed_idx < len(scene_options):
                best_scene_idx = parsed_idx
                
        selected_scene_idx = st.radio("Выберите сцену для просмотра в плеере:", options=range(len(scene_options)), format_func=lambda x: scene_options[x], horizontal=True, index=best_scene_idx)
        
        selected_scene = results[selected_scene_idx]
        start_time = selected_scene.metadata.get("start_time", 0.0)
        end_time = selected_scene.metadata.get("end_time", 0.0)
        
        # Layout: Left column for Video, Right column for Info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Плеер: Сцена {selected_scene_idx+1}")
            if os.path.exists(video_file_path):
                st.video(video_file_path, start_time=int(start_time))
            else:
                st.error(f"Файл видео '{video_file_path}' не найден. Плеер недоступен.")
        
        with col2:
            st.subheader("Исходное извлечение")
            st.text(selected_scene.page_content)
