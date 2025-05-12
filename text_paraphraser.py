from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from rich.console import Console
import os
import warnings
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel

# Avoid warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Data models
class ParaphraseRequest(BaseModel):
    text: str
    style: Literal["Fluency", "Humanize", "Formal", "Academic", "Simple", "Creative", "Shorten"]

class ParaphraseResponse(BaseModel):
    original_text: str
    paraphrased_text: str
    style: str

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
    def get_paraphrase_prompt() -> ChatPromptTemplate:
        """Returns a prompt template for text paraphrasing with different styles"""
        return ChatPromptTemplate.from_template(
            """You are an expert language paraphraser. Your task is to paraphrase the given text according to the specified style.
            
            Original text: {text}
            Style: {style}
            
            Style Guidelines:
            - Fluency: Make the text flow naturally and smoothly, focusing on readability.
            - Humanize: Make the text sound more conversational, warm, and relatable.
            - Formal: Use professional language, avoid contractions, and maintain a respectful tone.
            - Academic: Use scholarly language, precise terminology, and complex sentence structures.
            - Simple: Use straightforward language, short sentences, and common words.
            - Creative: Use vivid language, metaphors, and unique expressions.
            - Shorten: Condense the text while preserving the key information.
            
            Provide only the paraphrased text without any additional comments or explanations.
            """
        )

class ParaphraseService:
    """Service for text paraphrasing operations"""
    def __init__(self, llm_map: Dict[str, Any]):
        self.llm_map = llm_map
        self.prompt_manager = PromptManager()

    async def paraphrase_text(self, llm_name: str, text: str, style: str) -> Dict[str, Any]:
        """Paraphrase text using specified LLM and style"""
        if llm_name not in self.llm_map:
            raise ValueError(f"LLM '{llm_name}' not found")
        
        paraphrase_prompt = self.prompt_manager.get_paraphrase_prompt()
        selected_llm = self.llm_map[llm_name]
        response = (paraphrase_prompt | selected_llm).invoke({"text": text, "style": style})

        # Extract content from response
        content = response.content
        
        return {
            "original_text": text,
            "paraphrased_text": content,
            "style": style
        }

class APIRouter:
    """Manages API routes"""
    def __init__(self, app: FastAPI, llm_map: Dict[str, Any], paraphrase_service: ParaphraseService):
        self.app = app
        self.llm_map = llm_map
        self.paraphrase_service = paraphrase_service
        self.prompt_manager = PromptManager()

    def setup_routes(self):
        """Set up API routes"""
        # Add a unified route for text paraphrasing
        @self.app.post("/{llm}/paraphrase", response_model=ParaphraseResponse)
        async def paraphrase_text(llm: str, request: ParaphraseRequest):
            try:
                result = await self.paraphrase_service.paraphrase_text(
                    llm, 
                    request.text,
                    request.style
                )
                return result

            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

class TextParaphraser:
    """Main application class"""
    
    def __init__(self):
        self.llm_config = LLMConfig()
        self.llm_map = self.llm_config.initialize_models()

        # Initialize FastAPI app with additional metadata
        self.app = FastAPI(
            title="Text Paraphraser API",
            version="1.0",
            description="An API for paraphrasing text in different styles using LLMs",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Setup services and routes
        self.paraphrase_service = ParaphraseService(self.llm_map)
        self.router = APIRouter(self.app, self.llm_map, self.paraphrase_service)
        self.router.setup_routes()

    def get_app(self):
        """Get the FastAPI app instance"""
        return self.app

# Create application instance
application = TextParaphraser()
app = application.get_app()

# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "text_paraphraser:app",
        host="localhost", 
        port=8002,
        log_level="info",
        reload=True
    )