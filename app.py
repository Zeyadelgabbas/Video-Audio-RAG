import streamlit as st
from pathlib import Path
import time

from Src.config import config
from Src.video_pipeline import VideoProcessor
from Src.vector_store import VectorStore
from Src.rag_chat import RAGChat
from Src.logger import get_logger

logging = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Video RAG System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()

if 'rag_chat' not in st.session_state:
    st.session_state.rag_chat = RAGChat(st.session_state.vector_store)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def display_statistics():
    """Display system statistics in sidebar"""
    stats = st.session_state.processor.get_statistics()
    
    st.sidebar.markdown("### 📊 System Statistics")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("Videos", stats['database'].get('total_videos', 0))
        st.metric("Chunks", stats['database'].get('total_chunks', 0))
    
    with col2:
        duration_hours = stats['database'].get('total_duration_hours', 0)
        st.metric("Total Hours", f"{duration_hours:.1f}")
        st.metric("Embeddings", stats['vector_store'].get('total_chunks', 0))
    
    # List of videos
    video_names = stats['vector_store'].get('video_names', [])
    if video_names:
        st.sidebar.markdown("### 🎬 Available Videos")
        for video in video_names:
            st.sidebar.text(f"• {video}")


def process_videos_page():
    """Page for processing videos"""
    st.markdown('<div class="main-header">🎥 Video Processing</div>', unsafe_allow_html=True)
    
    # Instructions
    st.info(f"""
    **How to process videos:**
    1. Place video files in: `{config.VIDEOS_INPUT_PATH}`
    2. Supported formats: {', '.join(config.SUPPORTED_VIDEO_FORMATS)}
    3. Click 'Process Videos' button below
    4. Processed videos will be moved to: `{config.VIDEOS_FINISHED_PATH}`
    """)
    
    # Check for videos in input folder
    video_files = []
    for ext in config.SUPPORTED_VIDEO_FORMATS:
        video_files.extend(list(config.VIDEOS_INPUT_PATH.glob(f"*{ext}")))
    
    if video_files:
        st.success(f"✅ Found {len(video_files)} video(s) ready to process:")
        
        # Display list of videos
        for i, video in enumerate(video_files, 1):
            file_size_mb = video.stat().st_size / (1024 * 1024)
            st.text(f"{i}. {video.name} ({file_size_mb:.1f} MB)")
        
        st.markdown("---")
        
        # Process button
        if st.button("🚀 Process Videos", type="primary"):
            with st.spinner("Processing videos... This may take a while."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create placeholder for logs
                log_container = st.container()
                
                with log_container:
                    result = st.session_state.processor.process_folder()
                
                progress_bar.progress(100)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📋 Processing Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total", result['total'])
                with col2:
                    st.metric("✅ Successful", result['successful'])
                with col3:
                    st.metric("❌ Failed", result['failed'])
                with col4:
                    st.metric("⏭️ Skipped", result['skipped'])
                
                if result['successful'] > 0:
                    st.success("✅ Processing completed successfully!")
                    st.balloons()
                elif result['failed'] > 0:
                    st.error("❌ Some videos failed to process. Check logs for details.")
                
    else:
        st.warning(f"""
        ⚠️ No videos found in input folder!
        
        **To add videos:**
        1. Navigate to: `{config.VIDEOS_INPUT_PATH}`
        2. Copy your video files there
        3. Refresh this page
        """)


def chat_page():
    """Page for chatting with videos"""
    st.markdown('<div class="main-header">💬 Chat with Your Videos</div>', unsafe_allow_html=True)
    
    # Check if any videos are available
    stats = st.session_state.vector_store.get_collection_stats()
    
    if stats['total_chunks'] == 0:
        st.warning("""
        ⚠️ No videos available for chat yet!
        
        Please process some videos first in the **Process Videos** page.
        """)
        return
    
    # Video filter
    video_names = stats['video_names']
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        filter_option = st.selectbox(
            "Search in:",
            ["All Videos"] + video_names,
            help="Select a specific video or search across all videos"
        )
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.rag_chat.clear_memory()
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
    
    st.markdown("---")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['answer'])
                    
                    # Display sources
                    if message.get('sources'):
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(message['sources'], 1):
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i}</strong><br>
                                    🎥 Video: {source['video_name']}<br>
                                    ⏰ Time: {source['start_time']} - {source['end_time']}<br>
                                    📝 Context: {source['text_preview']}
                                </div>
                                """, unsafe_allow_html=True)
    
    # Chat input
    user_question = st.chat_input("Ask a question about your videos...")
    
    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if filter_option == "All Videos":
                    result = st.session_state.rag_chat.ask(user_question)
                else:
                    result = st.session_state.rag_chat.ask_with_video_filter(
                        user_question,
                        filter_option
                    )
                
                # Display answer
                st.write(result['answer'])
                
                # Display sources
                if result['sources']:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(result['sources'], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {i}</strong><br>
                                🎥 Video: {source['video_name']}<br>
                                ⏰ Time: {source['start_time']} - {source['end_time']}<br>
                                📝 Context: {source['text_preview']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'answer': result['answer'],
                    'sources': result['sources']
                })


def manage_videos_page():
    """Page for managing videos"""
    st.markdown('<div class="main-header">⚙️ Manage Videos</div>', unsafe_allow_html=True)
    
    # Get all videos
    videos = st.session_state.processor.database.get_all_videos()
    
    if not videos:
        st.info("No videos in database yet. Process some videos first!")
        return
    
    st.markdown("### 📹 Processed Videos")
    
    for video in videos:
        with st.expander(f"🎥 {video.video_name}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Duration", f"{video.total_duration/60:.1f} min")
            with col2:
                st.metric("Chunks", video.total_chunks)
            with col3:
                st.metric("Status", video.status)
            
            st.text(f"Processed: {video.processed_date.strftime('%Y-%m-%d %H:%M')}")
            st.text(f"Path: {video.original_path}")
            
            # Delete button
            if st.button(f"🗑️ Delete {video.video_name}", key=f"delete_{video.id}"):
                with st.spinner(f"Deleting {video.video_name}..."):
                    success = st.session_state.processor.delete_video(video.video_name)
                    
                    if success:
                        st.success(f"✅ Deleted {video.video_name}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to delete {video.video_name}")


def main():
    """Main application"""
    
    # Sidebar
    st.sidebar.title("🎥 Video RAG System")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["💬 Chat", "🎬 Process Videos", "⚙️ Manage Videos"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Display statistics
    display_statistics()
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    This system transcribes videos and enables conversational Q&A using RAG.
    
    **Features:**
    - Automatic transcription
    - Semantic search
    - Source attribution
    - Multi-video chat
    """)
    
    # Main content
    if page == "💬 Chat":
        chat_page()
    elif page == "🎬 Process Videos":
        process_videos_page()
    elif page == "⚙️ Manage Videos":
        manage_videos_page()


if __name__ == "__main__":
    # Validate configuration on startup
    try:
        config.validate_create_dirs()
        main()
    except Exception as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please check your .env file and ensure OPENAI_API_KEY is set.")