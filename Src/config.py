import os 
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config: 

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    LLM_MODEL = os.getenv("LLM_MODEL")

    VIDEOS_INPUT_PATH = Path(os.getenv("VIDEOS_INPUT_PATH"))
    VIDEOS_FINISHED_PATH = Path(os.getenv("VIDEOS_FINISHED_PATH"))

    CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH"))
    METADATA_DB_PATH = Path(os.getenv("METADATA_DB_PATH"))

    CHUNK_LENGTH_SECONDS = int(os.getenv("CHUNK_LENGTH_SECONDS"))
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]

    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))


    @classmethod
    def validate_create_dirs(cls):
        """ 
        Create dirs and check for api key 
        Call once at start
        """

        if not cls.OPENAI_API_KEY:
            raise ValueError("OPEN AI KEY Not found , add it to your .env file")
        

        cls.VIDEOS_FINISHED_PATH.mkdir(parents=True,exist_ok=True)
        cls.VIDEOS_INPUT_PATH.mkdir(parents=True,exist_ok=True)
        cls.CHROMA_DB_PATH.mkdir(parents=True,exist_ok=True)

        cls.METADATA_DB_PATH.parent.mkdir(parents=True,exist_ok=True)



config = Config()

