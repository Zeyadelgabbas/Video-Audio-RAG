# 🎥 Video RAG System

A production-ready system that transcribes videos and enables conversational Q&A about their content using Retrieval Augmented Generation (RAG).

## 📋 Features

- **Automatic Video Processing**: Extract, chunk, and transcribe videos using OpenAI Whisper
- **Semantic Search**: Find relevant content using vector embeddings
- **Source Attribution**: Get answers with exact video names and timestamps
- **Multi-Video Support**: Chat across multiple videos simultaneously
- **Conversation Memory**: Ask follow-up questions with context awareness
- **Streamlit Interface**: User-friendly web app for video ingestion and chat

## 🏗️ Architecture

```
Video Input → Audio Extraction → Transcription (Whisper) 
    ↓
Metadata Storage (SQLite) + Vector Store (ChromaDB)
    ↓
RAG Chat (OpenAI + LangChain)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Zeyadelgabbas/Video-Audio-RAG.git
cd Video-Audio-RAG

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_api_key_here
```

### 3. Run Application

```bash
streamlit run app.py
```

## 📁 Project Structure

```
video-rag-system/
├── Src/
│   ├── config.py              # Configuration management
│   ├── audio_extractor.py     # Video to audio conversion
│   ├── transcriber.py         # Whisper transcription
│   ├── database.py            # SQLite metadata storage
│   ├── vector_store.py        # ChromaDB vector operations
│   ├── rag_chat.py            # RAG chat engine
│   └── video_processor.py     # Complete pipeline
├── data/
│   ├── videos_to_process/     # Input videos here
│   └── finished_videos/       # Processed videos
├── app.py                     # Streamlit web interface
├── requirements.txt
└── .env.example
```

## 💡 Usage

### Processing Videos

1. Place videos in `data/videos_to_process/`
2. Open Streamlit app
3. Click "Process Videos" button
4. Videos are transcribed and moved to `finished_videos/`

### Chatting with Videos

1. Go to "Chat" tab
2. Ask questions about your videos
3. Get answers with source citations (video name + timestamp)

### Example Queries

- "What did the speaker say about machine learning?"
- "Summarize the main points from tutorial.mp4"
- "Compare the approaches discussed in video1 and video2"

## 🛠️ Technologies

- **Transcription**: OpenAI Whisper API
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: GPT-4 / GPT-3.5-turbo
- **Vector DB**: ChromaDB
- **Metadata DB**: SQLite
- **Framework**: LangChain (LCEL)
- **Interface**: Streamlit

## ⚙️ Configuration Options

Edit `.env` file:

```bash
# Video Processing
CHUNK_LENGTH_SECONDS=600        # Audio chunk size (10 min default)

# Models
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4

# RAG Settings
CHUNK_SIZE=1000                 # Text chunk size for retrieval
CHUNK_OVERLAP=200               # Overlap between chunks
TOP_K_RESULTS=5                 # Number of results to retrieve
```

## 📊 System Requirements

- Python 3.8+
- OpenAI API key with credits
- ~2GB disk space per hour of video
- ffmpeg (installed automatically with moviepy)

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📄 License

MIT License

## 👤 Author

**Zeyad Emad**
- GitHub: [@Zeyadelgabbas](https://github.com/Zeyadelgabbas)
- LinkedIn: [Zeyad Elgabas](https://www.linkedin.com/in/zeyad-elgabas-9862082b7)
- Email: Zeyadelgabas@gmail.com

---

⭐ Star this repo if you find it helpful!