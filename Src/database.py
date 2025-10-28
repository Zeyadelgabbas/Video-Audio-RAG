from sqlalchemy import create_engine , Column , Integer , Float , DateTime , Text , String , func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List , Dict

from Src.config import config
from Src.logger import get_logger

logging = get_logger(__name__)

Base = declarative_base()

class VideoMetaData(Base):
    """
    Stores metadata about each processed 
    """

    __tablename__ = 'video_metadata'

    id = Column(Integer , primary_key= True , autoincrement= True)
    video_name = Column(String(255),nullable=False, unique= True)
    original_path  = Column(String(512),nullable=False)
    total_chunks = Column(Integer,nullable=False)
    total_duration = Column(Float, nullable=False)
    processed_date = Column(DateTime,default=datetime.utcnow)
    status = Column(String(50),default='comleted') 


    def __repr__(self):
        return f"<Video(name='{self.video_name}', duration = {self.total_duration}s)>"
    

class TranscriptChunk(Base):

    """
    Stores info about each transcripted chunk 
    """
    
    __tablename__ = 'transcript_chunks'


    id = Column(Integer, primary_key=True, autoincrement=True)
    video_name = Column(String(255), nullable=False)       
    chunk_index = Column(Integer, nullable=False)          
    start_time = Column(Float, nullable=False)             
    end_time = Column(Float, nullable=False)               
    start_formatted = Column(String(20), nullable=False)   
    end_formatted = Column(String(20), nullable=False)     
    text = Column(Text, nullable=False)                   
    char_count = Column(Integer, nullable=False)           
    vector_id = Column(String(255))                       
    created_date = Column(DateTime, default=datetime.utcnow)


    def __repr__(self):
        return f"<Chunk(video='{self.video_name}', index={self.chunk_index}, time={self.start_formatted}-{self.end_formatted})>"
    


class Database: 

    """
    Manages all database operations
    """
    def __init__(self , db_path: str = None):

        if not db_path:
            db_path = str(config.METADATA_DB_PATH)

        self.engine = create_engine(f"sqlite:///{db_path}")

        # create tables if not exist
        Base.metadata.create_all(self.engine)

        # Create session factory 
        Session = sessionmaker(bind=self.engine)

        self.session = Session()

        print(f"Database connected  : {db_path}")


    def add_video(self, video_name :str , original_path: str , 
                  total_duration: float , total_chunks: int ) ->int:
        
        """
        Adds a new video to the database 

        returns:
            video_id : the Id "primary key" of the inserted video
        """

        try:
            video = VideoMetaData(
                video_name = video_name,
                original_path = original_path,
                total_chunks = total_chunks ,
                total_duration  = total_duration,
                status = 'completed'
            )

            self.session.add(video)
            self.session.commit()
            logging.info(f"added video : {video_name} with id : {video.id}to the VideoMetaData database")

            return video.id
        except Exception as e: 
            self.session.rollback()
            logging.error(f"Error adding video {str(e)}")
            raise

    def add_transcript_chunk(self,video_name:str , chunk_data:dict , vector_id: str = None) ->int:

        """
        add a transcript chunk with its metadata.
        
        Args:
            chunk_data : dictionary resulted from transcriber with keys :
            (text,start_time,end_time,chunk_index,start_formatted,end_formatted,audio_name)
            
            vector_id ; ID from chromadb 

        Returns: 
            chunk_id : id for the inserted chunk
        """

        try:
            chunk = TranscriptChunk(
                video_name = video_name , 
                vector_id = vector_id , 
                text = chunk_data['text'],
                start_time = chunk_data['start_time'],
                end_time = chunk_data['end_time'],
                start_formatted = chunk_data['start_formatted'],
                end_formatted = chunk_data['end_formatted'],
                char_count = len(chunk_data['text']),
                chunk_index = chunk_data.get('chunk_index',0)

            )
            self.session.add(chunk)
            self.session.commit()
            logging.info(f"chunk added ")
            return chunk.id
        except Exception as e:
            self.session.rollback()
            logging.error(f"Error in adding chunk : {e}")
            raise

    def update_chunk_vector_id(self,chunk_id : int , vector_id : str):

        """
        Update the vector id after adding the chunk to vector database

        Links the database to the vector store
        """


        try:
            chunk = self.session.query(TranscriptChunk).filter_by(id=chunk_id).first()
            if chunk:
                chunk.vector_id = vector_id
                self.session.commit()
        except Exception as e: 
            self.session.rollback()
            logging.error(f"erorr updating vector id : {e}")

    def get_video_by_name(self,video_name):

        return self.session.query(VideoMetaData).filter_by(video_name=video_name).first()
    

    def video_exists(self,video_name: str) ->bool:

        if self.get_video_by_name(video_name):
            return True
        return False
    
    def get_chunks_by_video(self,video_name: str) ->List[TranscriptChunk]:
        """
        get all chunks for a specific video by name ordered by chunk index
        """

        return self.session.query(TranscriptChunk).filter_by(video_name=video_name
        ).order_by(TranscriptChunk.chunk_index).all()

    def get_all_videos(self):

        return self.session.query(VideoMetaData).order_by(VideoMetaData.processed_date.desc()).all()
    

    def get_chunk_by_id(self,chunk_id):

        return self.session.query(TranscriptChunk).filter_by(id = chunk_id).first()
    

    def delete_video(self,video_name: str) ->bool:

        """
        delete video and all its chunks from database

        this doesnt delete it from the vector store
        """


        try:

            deleted_chunks = self.session.query(TranscriptChunk).filter_by(video_name=video_name).delete()
            
            deleted_video = self.session.query(VideoMetaData).filter_by(video_name=video_name).delete()
            self.session.commit()

            logging.info(f"deleted video {video_name} and all its chunks")
            return True
        except Exception as e:
            self.session.rollback()
            logging.error(f"error in deleting video {video_name} : {str(e)}")
            return False
        

    def get_video_statistics(self) -> dict:
        """
        Get overall statistics about processed videos.
        
        Returns:
            Dictionary with stats like total videos, total chunks, etc.
        """
        total_videos = self.session.query(VideoMetaData).count()
        total_chunks = self.session.query(TranscriptChunk).count()
        total_duration = self.session.query(VideoMetaData).with_entities(
            func.sum(VideoMetaData.total_duration)
        ).scalar() or 0
        
        return {
            'total_videos': total_videos,
            'total_chunks': total_chunks,
            'total_duration_seconds': total_duration,
            'total_duration_hours': total_duration / 3600
        }
    
    def close(self):

        self.session.close()
        logging.info("Database connection closed")



if __name__ == "__main__":
    db = Database()
    
    # Test adding a video
    video_id = db.add_video(
        video_name="test_video.mp4",
        original_path="Assets/test_video.mp4",
        total_duration=1500.0,
        total_chunks=3
    )
    
    # Test adding chunks
    chunk_data = {
        'text': 'This is a test transcription...',
        'start_time': 0.0,
        'end_time': 600.0,
        'start_formatted': '00:00:00',
        'end_formatted': '00:10:00',
        'chunk_index': 0
    }
    
    chunk_id = db.add_transcript_chunk("test_video.mp4", chunk_data)
    
    # Test queries
    print("\nAll videos:", db.get_all_videos())
    print("\nChunks for test_video.mp4:", db.get_chunks_by_video("test_video.mp4"))
    
    db.close()
