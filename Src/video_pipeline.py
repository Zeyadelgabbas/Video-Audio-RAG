from pathlib import Path 
from typing import List , Dict
import shutil


from Src.config import config
from Src.transcriber import Transcriber
from Src.audio_extractor import AudioExtractor
from Src.database import Database
from Src.vector_store import VectorStore
from Src.logger import get_logger


logging = get_logger(__name__)

class VideoProcessor:

    """
    Create complete video processing pipeline
    """

    def __init__(self):

        self.audio_extractor = AudioExtractor()
        self.transcriber = Transcriber()
        self.database = Database()
        self.vector_store = VectorStore()


    def process_video(self,video_path: Path) -> bool:

        try:
            video_name = video_path.name
            # Check if video already exists in database
            if self.database.video_exists(video_name=video_name) and video_name in self.vector_store.get_all_video_names():
                logging.warning(f"video : {video_name} already exists in the database")
                return False
            

            # extract audio from video
            audio_path = self.audio_extractor.extract_audio(video_path=video_path)
            
            # split extracted audio into chunks
            audio_splits_chunks = self.audio_extractor.split_audio(audio_path=audio_path)

            if not audio_splits_chunks:
                logging.error(f"No audio chuks created for {video_name}")
                return False
            


            # transcript all chunks : return list of dictionaries each is a chunk
            transcriptions = self.transcriber.transcribe_all_chunks(chunks= audio_splits_chunks)

            if not transcriptions:
                logging.error(f"No transcribtions generated for video {video_name}")
                return False
            
            total_duration = audio_splits_chunks[-1][2] if audio_splits_chunks else 0.0

            logging.info(f"Saving video {video_name} to Database")
            # Save to database
            if not self.database.video_exists(video_name=video_name):
                self.database.add_video(
                    video_name=video_name,
                    original_path=str(video_path),
                    total_duration = total_duration,
                    total_chunks = len(transcriptions)
                )

            # Save chunks to database
            logging.info(f"Adding transcript chunks to the database")
            for transcript in transcriptions:
                self.database.add_transcript_chunk(
                    video_name=video_name,
                    chunk_data=transcript
                )

            logging.info(f"Adding embeddings to the vector database for video {video_name}")

            vector_ids = self.vector_store.add_transcripts(
                video_name=video_name,
                transcripts=transcriptions
            )

            if not vector_ids:
                logging.error(f"No embedding created for video {video_name}")

            finished_path = config.VIDEOS_FINISHED_PATH / video_name

            if finished_path.exists():
                counter = 1
                while finished_path.exists():
                    stem = video_path.stem
                    suffix = video_path.suffix

                    finished_path=config.VIDEOS_FINISHED_PATH/f"{stem}_{counter}{suffix}"
                    counter +=1

            shutil.move(str(video_path),str(finished_path))
            logging.info(f"moved video to: {finished_path}")
            self.audio_extractor.cleanup()
            return True
                        

        except Exception as e :
            logging.error(f"Error processing video : {video_name} : {e}")

            try:
                self.audio_extractor.cleanup()
            except:
                pass
            
            return False
        

    def process_folder(self,folder_path: Path = None) ->Dict:

        """
        Processes all videos in a specific folder 

        returns:    
            dictionary with ( total video , success , failed , skipped , processed_videos names , failed_videos names)
        """


        if folder_path is None:
            folder_path = config.VIDEOS_INPUT_PATH

        logging.info(f"Processing all videos in folder : {folder_path}")

        videos_paths = []
        for extension in config.SUPPORTED_VIDEO_FORMATS:
            videos_paths.extend(list(folder_path.glob(f"*{extension}")))

        if not videos_paths:
            return{
                'total':0,
                'sucess':0,
                'failed':0,
                'skipped':0,
                'processed_videos':[],
                'failed_videos':[]
            }
        logging.info(f"Found {len(videos_paths)}  videos")

        stats = {
                'total':len(videos_paths),
                'sucess':0,
                'failed':0,
                'skipped':0,
                'processed_videos':[],
                'failed_videos':[]
            }
        
        for i, video_path in enumerate(videos_paths):
            
            if self.database.video_exists(video_name=video_path.name) and video_path.name in self.vector_store.get_all_video_names():
                stats['skipped']+=1
                logging.info(f"skipping video : {video_path.name}")
                continue

            sucess = self.process_video(video_path=video_path)
            if sucess:
                stats['sucess'] +=1
                stats['processed_videos'].append(video_path.name)

            else:
                stats['failed'] +=1
                stats['failed_videos'].append(video_path.name)

        logging.info(f"Folder processing completed with stats : {stats}")
        return stats


    def get_statistics(self):

        """
        returns all database statistics
        """


        database_stats = self.database.get_video_statistics()
        vectordb_stats = self.vector_store.get_collection_stats()


        return {
            'database':database_stats,
            'vector_store':vectordb_stats
        }
    
    def delete_video(self,video_name: str) -> bool:

        """
        Delete video from both database and vector store
        """


        if self.database.video_exists(video_name=video_name):
            db_sucess = self.database.delete_video(video_name=video_name)

        vs_db = self.vector_store.delete_by_video_name(video_name=video_name)

        if db_sucess and vs_db:
            return True
        else:
            return False
        

    def close(self):

        self.database.close()
        logging.info(f"Video processor closed ! ")



if __name__ == "__main__":
    print("🎬 Video Processor Test\n")
    
    # Initialize processor
    processor = VideoProcessor()
    
    # Show current statistics
    print("📊 Current Statistics:")
    stats = processor.get_statistics()
    print(f"   Database: {stats['database']}")
    print(f"   Vector Store: {stats['vector_store']}")
    print()
    
    # Check if there are videos to process
    input_path = config.VIDEOS_INPUT_PATH
    video_files = []
    for ext in config.SUPPORTED_VIDEO_FORMATS:
        video_files.extend(list(input_path.glob(f"*{ext}")))
    
    if video_files:
        print(f"Found {len(video_files)} video(s) in input folder.")
        response = input("Do you want to process them? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            # Process all videos in folder
            result = processor.process_folder()
            
            # Show updated statistics
            print("\n📊 Updated Statistics:")
            stats = processor.get_statistics()
            print(f"   Database: {stats['database']}")
            print(f"   Vector Store: {stats['vector_store']}")
        else:
            print("Processing cancelled.")
    else:
        print(f"⚠️  No videos found in: {input_path}")
        print(f"\nTo test the processor:")
        print(f"1. Place video files in: {input_path}")
        print(f"2. Run this script again")
    
    # Close processor
    processor.close()


