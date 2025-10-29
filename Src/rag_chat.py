from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

from Src.config import config
from Src.vector_store import VectorStore
from Src.logger import get_logger

logging = get_logger(__name__)


class RAGChat:
    """
    Handles conversational question-answering using RAG (Retrieval Augmented Generation).
    
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG chat engine.
        
        Args:
            vector_store: VectorStore instance with loaded transcripts
        """
        self.vector_store = vector_store
        
        # Initialize ChatGPT
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.7,  # 0 = deterministic, 1 = creative
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # Manual conversation history (replaces deprecated ConversationBufferMemory)
        self.chat_history = []
        
        # Get retriever from vector store
        self.retriever = self.vector_store.get_retriever()
        
        # Create the RAG chain using LCEL
        self.chain = self._create_rag_chain()
        
        logging.info("RAG chat engine initialized with LCEL")
    
    
    def _create_rag_chain(self):
        """
        Create RAG chain using  LCEL approach.
 
        """
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on video transcripts.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. If the answer is not in the context, say "I don't have information about that in the videos."
3. ALWAYS cite your sources by mentioning:
   - The video name
   - The timestamp (when that information appears)
4. Be conversational and helpful
5. If multiple videos contain relevant information, mention all of them

Context from videos:
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Create the chain using LCEL
        # Format: retriever | format_docs | prompt | llm | parser
        chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.chat_history
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    
    def _format_docs(self, docs) -> str:
        """
        Format retrieved documents into a string for the prompt.
        
        Args:
            docs: List of Document objects from retriever
            
        Returns:
            Formatted string with all document contents
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            video_name = doc.metadata.get('video_name', 'Unknown')
            start_time = doc.metadata.get('start_formatted', 'N/A')
            end_time = doc.metadata.get('end_formatted', 'N/A')
            
            formatted.append(
                f"[Source {i}]\n"
                f"Video: {video_name}\n"
                f"Time: {start_time} - {end_time}\n"
                f"Content: {doc.page_content}\n"
            )
        
        return "\n".join(formatted)
    
    
    def ask(self, question: str) -> Dict:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and source
            
        """
        logging.info(f"User question: {question}")
        
        try:
            # Retrieve relevant documents manually (to get sources)
            retrieved_docs = self.retriever.invoke(question)
            
            # Generate answer using the chain
            answer = self.chain.invoke(question)
            
            # Update chat history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))
            
            # Format sources with metadata
            sources = []
            for doc in retrieved_docs:
                source = {
                    'video_name': doc.metadata.get('video_name', 'Unknown'),
                    'start_time': doc.metadata.get('start_formatted', 'N/A'),
                    'end_time': doc.metadata.get('end_formatted', 'N/A'),
                    'text_preview': doc.page_content[:150] + '...' if len(doc.page_content) > 150 else doc.page_content
                }
                sources.append(source)
            
            logging.info(f"Generated answer with {len(sources)} sources")
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            logging.error(f"Error during RAG query: {e}")
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': []
            }
    
    
    def ask_with_video_filter(self, question: str, video_name: str) -> Dict:
        """
        Ask a question about a SPECIFIC video only.
        
        Args:
            question: User's question
            video_name: Search only in this video
            
        Returns:
            Same format as ask()
        """
        logging.info(f"User question (filtered by {video_name}): {question}")
        
        try:
            # Create a filtered retriever
            filtered_retriever = self.vector_store.vectorstore.as_retriever(
                search_kwargs={
                    "k": config.TOP_K_RESULTS,
                    "filter": {"video_name": video_name}
                }
            )
            
            # Retrieve relevant documents
            retrieved_docs = filtered_retriever.invoke(question)
            
            # Create temporary chain with filtered retriever
            temp_chain = (
                {
                    "context": filtered_retriever | self._format_docs,
                    "question": RunnablePassthrough(),
                    "chat_history": lambda x: self.chat_history
                }
                | ChatPromptTemplate.from_messages([
                    ("system", f"""You are a helpful AI assistant that answers questions based on video transcripts.

Answer questions based ONLY on the video: {video_name}

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the provided context from this video
2. If the answer is not in the context, say "I don't have information about that in this video."
3. ALWAYS cite the timestamp when that information appears
4. Be conversational and helpful

Context from video:
{{context}}"""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ])
                | self.llm
                | StrOutputParser()
            )
            
            # Generate answer
            answer = temp_chain.invoke(question)
            
            # Update chat history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))
            
            # Format sources
            sources = []
            for doc in retrieved_docs:
                source = {
                    'video_name': doc.metadata.get('video_name', 'Unknown'),
                    'start_time': doc.metadata.get('start_formatted', 'N/A'),
                    'end_time': doc.metadata.get('end_formatted', 'N/A'),
                    'text_preview': doc.page_content[:150] + '...' if len(doc.page_content) > 150 else doc.page_content
                }
                sources.append(source)
            
            logging.info(f"Generated answer from {video_name} with {len(sources)} sources")
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            logging.error(f"Error during filtered RAG query: {e}")
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': []
            }
    
    
    def clear_memory(self):
        """
        Clear conversation history.
        
        Use this when starting a new conversation or switching topics.
        """
        self.chat_history = []
        logging.info("Conversation history cleared")
    
    
    def get_chat_history(self) -> List[Dict]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries
        """
        try:
            history = []
            
            for msg in self.chat_history:
                history.append({
                    'role': 'human' if isinstance(msg, HumanMessage) else 'ai',
                    'content': msg.content
                })
            
            return history
            
        except Exception as e:
            logging.error(f"Error getting chat history: {e}")
            return []
    
    
    def format_answer_with_sources(self, result: Dict) -> str:
        """
        Format the answer with sources in a nice readable way.
        
        Args:
            result: Dictionary from ask() or ask_with_video_filter()
            
        Returns:
            Formatted string for display
        """
        output = f"{result['answer']}\n"
        
        if result['sources']:
            output += "\n" + "="*60 + "\n"
            output += "📚 Sources:\n\n"
            
            for i, source in enumerate(result['sources'], 1):
                output += f"{i}. 🎥 Video: {source['video_name']}\n"
                output += f"   ⏰ Time: {source['start_time']} - {source['end_time']}\n"
                output += f"   📝 Context: {source['text_preview']}\n\n"
        
        return output


# Example usage (for testing):
if __name__ == "__main__":
    print("🤖 Initializing RAG Chat (New LCEL Approach)...\n")
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Check if we have any data
    stats = vector_store.get_collection_stats()
    print(f"📊 Database stats: {stats}\n")
    
    if stats['total_chunks'] == 0:
        print("⚠️  No video transcripts found in database!")
        print("Adding test data for demo...\n")
        
        # Add test data
        test_transcripts = [
            {
                'text': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns and make decisions.',
                'start_time': 0.0,
                'end_time': 600.0,
                'start_formatted': '00:00:00',
                'end_formatted': '00:10:00',
                'chunk_index': 0
            },
            {
                'text': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers. These neural networks can learn complex patterns in large amounts of data.',
                'start_time': 600.0,
                'end_time': 1200.0,
                'start_formatted': '00:10:00',
                'end_formatted': '00:20:00',
                'chunk_index': 1
            },
            {
                'text': 'Python is a popular programming language for machine learning because it has powerful libraries like TensorFlow, PyTorch, and scikit-learn.',
                'start_time': 1200.0,
                'end_time': 1800.0,
                'start_formatted': '00:20:00',
                'end_formatted': '00:30:00',
                'chunk_index': 2
            }
        ]
        
        vector_store.add_transcripts(test_transcripts, "AI_Basics.mp4")
        print("✅ Test data added!\n")
    
    # Initialize RAG chat
    rag = RAGChat(vector_store)
    
    # Test questions
    print("="*60)
    print("🧪 Testing RAG Chat with LCEL\n")
    
    # Question 1
    print("Question 1: What is machine learning?\n")
    result1 = rag.ask("What is machine learning?")
    print(rag.format_answer_with_sources(result1))
    
    # Question 2 (follow-up with memory)
    print("\n" + "="*60)
    print("Question 2: What programming language is used for it?\n")
    result2 = rag.ask("What programming language is used for it?")
    print(rag.format_answer_with_sources(result2))
    
    # Show chat history BEFORE clearing
    print("\n" + "="*60)
    print("💬 Chat History (Before Clearing):")
    history = rag.get_chat_history()
    
    if not history:
        print("  (No messages in history)")
    else:
        for msg in history:
            role = "👤 User" if msg['role'] == 'human' else "🤖 AI"
            content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
            print(f"\n{role}: {content_preview}")
    
    print(f"\n📊 Total messages: {len(history)}")
    
    # Clear memory and ask new question
    print("\n" + "="*60)
    rag.clear_memory()
    print("🗑️  Memory cleared\n")
    
    # Question 3
    print("Question 3: Tell me about deep learning\n")
    result3 = rag.ask("Tell me about deep learning")
    print(rag.format_answer_with_sources(result3))
    
    # Show chat history AFTER clearing
    print("\n" + "="*60)
    print("💬 Chat History (After Clearing & New Question):")
    history_after = rag.get_chat_history()
    
    if not history_after:
        print("  (No messages in history)")
    else:
        for msg in history_after:
            role = "👤 User" if msg['role'] == 'human' else "🤖 AI"
            content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
            print(f"\n{role}: {content_preview}")
    
    print(f"\n📊 Total messages: {len(history_after)}")