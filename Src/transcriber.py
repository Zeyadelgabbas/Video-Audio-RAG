from openai import OpenAI
from pathlib import Path 

from Src.config import config 
from Src.logger import get_logger

logging = get_logger(__name__)
class Transcriber:

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        logging.info("openai client created")
    def transcribe_chunk(self, audio_path: Path , start_time: float , end_time: float):
        """
        transcribe a chunk using openai whisper model 
        
        Args: 
            audio_path : Path to the audio chunk 
            start_time : the starting time of the audio 
            end_time : the ending time of the audio 
        """

        try:
            with open(audio_path,'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    file = audio_file,
                    model = 'whisper-1',
                    response_format= 'text'
                )
                logging.info("chunk transcripted sucessefully")
                
            result = {
                'text':transcript , 
                'start_time': start_time,
                'end_time' : end_time,
                'duration' : end_time - start_time,
                'start_formatted': self.format_time(start_time),
                'end_formatted' : self.format_time(end_time)
            }
        
        except Exception as e:
            logging.error(f"Error transcribing a chunk : {str(e)}")
            raise

        return result
    
    def transcribe_all_chunks(self,chunks: list) ->list:

        transcribtions = []
        for i , (audio_path,start_time,end_time,audio_name) in enumerate(chunks):

            result = self.transcribe_chunk(
                audio_path=audio_path ,
                start_time=start_time,
                end_time=end_time
            )
            result['chunk_index'] = i
            result['audio_name'] = str(audio_name)
            transcribtions.append(result)
        logging.info(f"all chunks transcribted sucessfully !")
        return transcribtions
    
    def format_time(self,time):
        """
        convert seconds to formatted time

        args: 
            time: time in seconds

        returns: 
            time : time formatted Hour:minutes:seconds
        """

        hours = int(time // 3600)        
        minutes = int((time % 3600) // 60)  
        secs = int(time % 60)            
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Example usage (for testing):
if __name__ == "__main__":
    transcriber = Transcriber()
    
    # Test transcribing a single file
    test_audio = Path("test_audio.wav")
    if test_audio.exists():
        result = transcriber.transcribe_chunk(test_audio, 0.0, 600.0)
        print("\nResult:", result)
    
    # Test time formatting
    print("\nTime format tests:")
    print(f"0s → {transcriber.format_time(0)}")
    print(f"65s → {transcriber.format_time(65)}")
    print(f"3725s → {transcriber.format_time(3725)}")