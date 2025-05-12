from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from rich.console import Console
import os
import warnings
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Avoid warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Data models
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None

class LLMConfig:
    """Configuration for LLM models"""

    def __init__(self):
        self.console = Console()
        load_dotenv()

    def initialize_models(self):
        """Initialize LLM models"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            
            if not api_key:
                self.console.print("[red]Error: GOOGLE_API_KEY not found in environment variables[/red]")
                raise ValueError("GOOGLE_API_KEY not found in environment variables")

            self.gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
            )
            return {"gemini": self.gemini_llm}

        except Exception as e:
            self.console.print(f"[red]Error initializing LLM models: {str(e)}[/red]")
            raise

class PromptManager:
    """Manages prompt templates for different tasks"""
    @staticmethod
    def get_chat_prompt() -> ChatPromptTemplate:
        """Returns a prompt template for conversational AI"""
        return ChatPromptTemplate.from_template(
            """You are a helpful, friendly, and knowledgeable AI assistant. You provide accurate, 
            thoughtful responses to user questions. You're designed to be helpful, harmless, and honest.
            
            User message: {message}
            
            Provide a clear, concise, and helpful response. If you don't know something, admit it rather than making up information.
            If the question is unclear, ask for clarification. If the question is inappropriate, politely decline to answer.
            """
        )

class ChatService:
    """Service for chat operations"""
    def __init__(self, llm_map: Dict[str, Any]):
        self.llm_map = llm_map
        self.prompt_manager = PromptManager()

    async def chat_with_ai(self, llm_name: str, message: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process a chat message using specified LLM and return the response"""
        if llm_name not in self.llm_map:
            raise ValueError(f"LLM '{llm_name}' not found")
        
        chat_prompt = self.prompt_manager.get_chat_prompt()
        selected_llm = self.llm_map[llm_name]
        
        # Process conversation history if provided
        if conversation_history:
            pass

        response = (chat_prompt | selected_llm).invoke({"message": message})
        
        # Extract content from response
        content = response.content
        
        return {
            "response": content,
            "conversation_id": None  # In a real implementation, you might generate and track conversation IDs
        }

class APIRouter:
    """Manages API routes"""
    def __init__(self, app: FastAPI, llm_map: Dict[str, Any], chat_service: ChatService):
        self.app = app
        self.llm_map = llm_map
        self.chat_service = chat_service
        self.prompt_manager = PromptManager()

    def setup_routes(self):
        """Set up API routes"""
        # Add a unified route for chat
        @self.app.post("/{llm}/chat", response_model=ChatResponse)
        async def chat_with_ai(llm: str, request: ChatRequest):
            try:
                result = await self.chat_service.chat_with_ai(
                    llm, 
                    request.message,
                    request.conversation_history
                )
                return result

            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

class ChatWithAI:
    """Main application class"""
    
    def __init__(self):
        self.llm_config = LLMConfig()
        self.llm_map = self.llm_config.initialize_models()

        # Initialize FastAPI app with additional metadata
        self.app = FastAPI(
            title="Chat with AI API",
            version="1.0",
            description="An API for conversational AI using LLMs",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Setup services and routes
        self.chat_service = ChatService(self.llm_map)
        self.router = APIRouter(self.app, self.llm_map, self.chat_service)
        self.router.setup_routes()

    def get_app(self):
        """Get the FastAPI app instance"""
        return self.app

# Create application instance
application = ChatWithAI()
app = application.get_app()

# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "chat_with_ai_api:app",
        host="localhost", 
        port=8001,  # Using a different port than grammar API
        log_level="info",
        reload=True
    )