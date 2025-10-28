from pathlib import Path
from typing import List , Dict , Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Src.config import config 
from Src.logger import get_logger

logging = get_logger(__name__)

class VectorStore:
    """
    Manages vector embeddings using ChromaDB.
    """

    def __init__(self,collection_name: str = "video_transcripts"):

        """
        Initialize vector store with openai embeddings and ChromaDB
        """

        self.collection_name = collection_name

        self.embeddings = OpenAIEmbeddings(
            model = config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = config.CHUNK_SIZE,
            chunk_overlap = config.CHUNK_OVERLAP,
            length_function = len,
        )

        self.vectorstore = Chroma(
            collection_name= collection_name,
            embedding_function= self.embeddings,
            persist_directory=str(config.CHROMA_DB_PATH)
        )

    
    def add_transcripts(self,transcripts: List[Dict], video_name: str) -> List[str]:

        """
        adds transcripts to the vector database with metadata

        Args:
            transcripts: list of transcript dicts contain ( text , start_time , end_time , etc)

        returns: 
            list of documents ids
        """

        logging.info(f"adding transcripts to vector store for video : {video_name}")

        try:
            all_texts = []
            all_metadatas = []
            
            for i , transcript in enumerate(transcripts):
                text = transcript.get('text','')

                if not text.strip():
                    continue

                sub_chunks = self.text_splitter.split_text(text)

                for j , sub_chunk in enumerate(sub_chunks):
                    all_texts.append(sub_chunk)

                    metadata = {
                        'video_name':video_name,
                        'chunk_index' : i,
                        'sub_chunk_index' : j , 
                        'start_time' : transcript.get('start_time',0.0),
                        'end_time' : transcript.get('end_time',0.0),
                        'start_formatted' : transcript['start_formatted'],
                        'end_formatted' : transcript['end_formatted']
                    }

                    all_metadatas.append(metadata)

            if not all_texts:
                print(f"  ⚠️  No valid text chunks to add for {video_name}")
                return []
            
            ids = self.vectorstore.add_texts(
                texts=all_texts,
                metadatas=all_metadatas
            )

            logging.info(f"added {len(all_texts)} chunks to vector store")
            return ids


        except Exception as e:
            logging.error(f"Error adding transcripts to vector store for video {video_name} : {e}")
            raise



    def similarity_search(self, query: str , k: int = None , 
                          video_name: str = None) -> List[tuple]:
        
        """"
        Applies similarity seacrh between a query and the vector database chunks 

        Args: 
            query: user's quesiton
            k: number of returned results
            video_name : optional (search only for this video) 'metadata query'
        
        returns:
            List of tuples contains the result document and the similarity score
            [(Document,similarity_score) , ....]

        """
        
        if not k :
            k = config.TOP_K_RESULTS
        
        try:
            filter_dict = None 
            if video_name:
                filter_dict = {'video_name':video_name}
            
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter= filter_dict
                )

            else:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                )
            
            return results

        except Exception as e:
            logging.error(f"Error during similarity search for your query : {query} : \n\n Error: {e}")


    def get_retriever(self, search_kwargs : Dict = None):

        """
        Get langchain retriever for RAG.

        returns vectorstore retreiver to be used in langchain
        """
        
        if not search_kwargs:
            search_kwargs = {'k':config.TOP_K_RESULTS}

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    

    def delete_by_video_name(self,video_name : str) ->bool:

        """"
        Delete all chunks belonging to specific video 

        returns:
            True if sucess

        """
        try: 

            results = self.vectorstore.get(
                where = {'video_name':video_name}
            )

            if results and results['ids']:
                self.vectorstore.delete(ids = results['ids'])
                logging.info(f"deleted video {video_name} from vector database sucessfully")
                return True
            else:
                logging.error(f"no chunks found for video {video_name}")
                return False

        except Exception as e:
            logging.error(f"Error deleting video {video_name} , Error: {e}")
            return False
        

    def get_all_video_names(self) -> List[str]:

        try:
            results = self.vectorstore.get()

            if results and results['metadatas']:
                video_names = set()

                for metadata in results['metadatas']:
                    if metadata['video_name']:
                        video_names.add(metadata['video_name'])

                return sorted(list(video_names))
            return []

        except Exception as e:
            logging.error(f"Error getting video names {e}")
            return []
        

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats like total chunks, unique videos, etc.
        """
        try:
            results = self.vectorstore.get()
            
            total_chunks = len(results['ids']) if results and results['ids'] else 0
            video_names = self.get_all_video_names()
            
            return {
                'total_chunks': total_chunks,
                'unique_videos': len(video_names),
                'video_names': video_names
            }
            
        except Exception as e:
            logging.error(f" Error gettin states : {e}")
            return {'total_chunks': 0, 'unique_videos': 0, 'video_names': []}


# Example usage (for testing):
if __name__ == "__main__":
    # Initialize
    vector_store = VectorStore()
    
    # Test adding transcripts
    test_transcripts = [
        {
            'text': 'Machine learning is a subset of artificial intelligence...',
            'start_time': 0.0,
            'end_time': 600.0,
            'start_formatted': '00:00:00',
            'end_formatted': '00:10:00',
            'chunk_index': 0
        },
        {
            'text': 'Deep learning uses neural networks with multiple layers...',
            'start_time': 600.0,
            'end_time': 1200.0,
            'start_formatted': '00:10:00',
            'end_formatted': '00:20:00',
            'chunk_index': 1
        }
    ]
    
    ids = vector_store.add_transcripts(test_transcripts, "AI_Tutorial.mp4")
    print(f"\nAdded {len(ids)} chunks")
    
    # Test search
    results = vector_store.similarity_search("What is deep learning?", k=2)
    print(results)
    
    # Test stats
    stats = vector_store.get_collection_stats()
    
    print(f"\nStats: {stats}")


    delete = vector_store.delete_by_video_name('AI_Tutorial.mp4')
    print(f"Deletion state : True")
    
    stats = vector_store.get_collection_stats()

    print(f"\nStats: {stats}")
