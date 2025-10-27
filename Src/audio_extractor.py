from pathlib import Path 
import math
import tempfile
from moviepy.editor import VideoFileClip
from Src.logger import get_logger
from pydub import AudioSegment
from Src.config import config

logging = get_logger(__name__)

class AudioExtractor:
    """ Extracts audio from video and split it into chunks """

    def __init__(self):

        self.temp_dir = Path(tempfile.gettempdir()) / "video_to_audio"
        self.temp_dir.mkdir(exist_ok=True)
        logging.info("Audio temp directory :{self.temp_dir}")


    def extract_audio(self,video_path: Path) ->Path:

        """ 
        Extracts audio from video 
        Args : 
            Path to the video file 
        returns:
             Path to extracted audio file 
        """

        logging.info("Extracting audio from video")

        try:
            video = VideoFileClip(str(video_path))
            audio_path = self.temp_dir/f"{video_path.stem}.wav"

            video.audio.write_audiofile(
                filename=str(audio_path),
                fps = 16000,
                verbose = False , 
                logger = None
            )
            video.close()
            logging.info(f"audio extracted from {video_path} to {audio_path}")

            return audio_path 
        
        except Exception as e :
            logging.error(f"Erorr extracting audio : {str(e)}")
            raise 

    
    def split_audio(self, audio_path: Path) ->list: 
        """
        Splits audio into chunks 

        Args: 
            audio_path : path to the audio file 

        returns: 
            list of tuples: [(chunk_path , start_time , end_time ), ....]
        """

        logging.info(f"splitting audio file: {str(audio_path)}")
        try: 
            audio = AudioSegment.from_file(file = str(audio_path))
            duration_in_seconds = len(audio)/ 1000

            chunk_length = config.CHUNK_LENGTH_SECONDS
            num_chunks = math.ceil(duration_in_seconds/chunk_length)

            chunks = []

            for i in range(num_chunks):
                start_ms = i*chunk_length*1000
                end_ms = min(len(audio), (i+1)*chunk_length*1000)

                chunk = audio[start_ms:end_ms]
                chunk_path = self.temp_dir/f"{audio_path.stem}_{i}.wav"

                start_time = start_ms / 1000
                end_time = end_ms / 1000
                
                chunks.append((chunk_path,start_time,end_time))
                chunk.export(str(chunk_path),format="wav")

            logging.info(f"audio file {str(audio_path)} splitted!")

            return chunks

        except Exception as e : 
            logging.error(f"Error chunking the audio file {str(audio_path)} : {str(e)}")

    def cleanup(self, file_path: Path = None):
        """
        Delete temporary audio files to free disk space.
        
        Args:
            file_path: Specific file to delete, or None to delete all
        """
        try:
            if file_path and file_path.exists():
                file_path.unlink()
            elif not file_path:
                for file in self.temp_dir.glob("*"):
                    file.unlink()
        except Exception as e:
            logging.error(f"error deleting ")


# Example usage (for testing):
if __name__ == "__main__":
    extractor = AudioExtractor()
    
    # Test with a video file
    video = Path("video_test.mp4")
    if video.exists():
        audio = extractor.extract_audio(video)
        chunks = extractor.split_audio(audio)
        print(f"\nCreated {len(chunks)} audio chunks")
        extractor.cleanup()