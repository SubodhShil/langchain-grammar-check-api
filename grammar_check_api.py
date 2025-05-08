from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
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
class TextRequest(BaseModel):
    text: str

class GrammarRule(BaseModel):
    rule_name: str
    description: str
    correct_examples: List[str]
    incorrect_examples: List[str]

class Correction(BaseModel):
    error: str
    suggestion: str
    type: str
    explanation: str
    grammar_rule: Optional[GrammarRule] = None

class TextResponse(BaseModel):
    original_text: str
    corrected_text: str
    corrections: List[Correction]

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
    def get_grammar_prompt() -> ChatPromptTemplate:
        """Returns a prompt template for grammar checking with detailed corrections"""
        return ChatPromptTemplate.from_template(
            """You are a professional editor and an expert grammar tutor. Check the following text for grammar and spelling errors, factual errors, word choice issues, and suggest better word combinations.
            
            Original text: {text}
            
            Provide your response in the following JSON format:

            ```json
            {{
              "corrected_text": "The corrected version of the text with ONLY grammatical errors fixed. DO NOT completely paraphrase the text.",
              "corrections": [
                {{
                  "error": "The original error text (the exact part of the sentence that is incorrect)",
                  "suggestion": "The corrected version of that specific part",
                  "type": "The type of error (e.g., spelling, grammar, punctuation, subject-verb agreement, verb tense, noun form, article usage, factual error, word choice, word combination)",
                  "explanation": "A brief, clear explanation of why this specific part is an error and how the suggestion fixes it.",
                  "grammar_rule": {{
                    "rule_name": "A concise name for the grammar rule that was violated (e.g., 'Subject-Verb Agreement for Singular Nouns', 'Past Tense Verb Form', 'Use of Definite Article').",
                    "description": "A detailed but simple, beginner-friendly explanation of the grammar rule. Explain it as if you are teaching someone learning English. Avoid jargon where possible or explain it clearly.",
                    "correct_examples": [
                      "A clear example sentence demonstrating the correct application of the rule.",
                      "Another clear example sentence, if applicable, showing a slightly different correct usage."
                    ],
                    "incorrect_examples": [
                      "An example sentence showing the common mistake related to this rule (similar to the 'error' found).",
                      "Another incorrect example, if it helps clarify the rule."
                    ]
                  }}
                }}
              ]
            }}
            ```
            
            IMPORTANT INSTRUCTIONS:
            1. For the "corrected_text", maintain the original text structure and only fix actual errors. DO NOT completely rewrite or paraphrase the text.
            2. Only suggest complete paraphrasing when the correction type is specifically "word combination".
            3. For grammatical errors, make minimal changes necessary to fix the specific issue.
            4. Identify actual errors only - don't suggest stylistic changes unless they're grammatically incorrect.
            5. For each correction, the "error" field must contain the exact text from the original that contains the error.
            6. You MUST include a variety of correction types in your analysis, including:
               - Grammar errors (verb tense, subject-verb agreement, etc.)
               - Spelling errors
               - Punctuation errors
               - Factual errors (when statements are objectively incorrect)
               - Word choice issues (when words are used incorrectly)
               - Word combination suggestions (better phrasing options)
            
            Instructions for the "grammar_rule" object:
            - "rule_name": Be specific. For example, instead of just "Verb Tense," use "Past Simple Tense for Completed Actions."
            - "description": Make this the core of the educational part. Explain the 'why' behind the rule.
            - "correct_examples": Ensure these are simple and directly illustrate the rule.
            - "incorrect_examples": These should mirror common mistakes, ideally similar to the error found in the original text.

            For non-grammatical issues like factual errors, word choice, or word combinations:
            - Include these in the "corrections" array with appropriate "type" values
            - For these types, the "grammar_rule" field can be null or contain simplified guidance on better writing practices
            - For "word combination" type, you may suggest alternative phrasing that improves clarity or flow
            - For "factual error" type, explain why the statement is factually incorrect and provide the correct information
            - For "word choice" type, explain why the current word is inappropriate and suggest better alternatives
            
            If there are no errors in the original text, return the original text as "corrected_text" and an empty array for "corrections".
            Ensure the JSON is well-formed.
            """
        )

class GrammarService:
    """Service for grammar checking operations"""
    def __init__(self, llm_map: Dict[str, Any]):
        self.llm_map = llm_map
        self.prompt_manager = PromptManager()

    async def check_grammar(self, llm_name: str, text: str) -> Dict[str, Any]:
        """Check grammar using specified LLM and return detailed corrections"""
        if llm_name not in self.llm_map:
            raise ValueError(f"LLM '{llm_name}' not found")
        
        grammar_prompt = self.prompt_manager.get_grammar_prompt()
        selected_llm = self.llm_map[llm_name]
        response = (grammar_prompt | selected_llm).invoke({"text": text})

        # Extract JSON from response if needed
        content = response.content

        # Handle potential formatting issues in the response
        try:
            import json
            import re

            # Try to extract JSON if it's wrapped in markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            # Parse the JSON response
            result = json.loads(content)

            # Ensure the response has the required fields
            if "corrected_text" not in result:
                result["corrected_text"] = text

            if "corrections" not in result:
                result["corrections"] = []

            return {
                "original_text": text,
                "corrected_text": result["corrected_text"],
                "corrections": result["corrections"]
            }

        except Exception as e:
            # Fallback if JSON parsing fails
            return {
                "original_text": text,
                "corrected_text": content,
                "corrections": []
            }


class APIRouter:

    """Manages API routes"""
    def __init__(self, app: FastAPI, llm_map: Dict[str, Any], grammar_service: GrammarService):
        self.app = app
        self.llm_map = llm_map
        self.grammar_service = grammar_service
        self.prompt_manager = PromptManager()

    def setup_routes(self):
        """Set up API routes"""
        # Add a unified route for grammar checking
        @self.app.post("/{llm}/check_grammar", response_model=TextResponse)
        async def check_grammar(llm: str, request: TextRequest):
            try:
                result = await self.grammar_service.check_grammar(llm, request.text)
                return result

            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

class GrammarCheckAPI:

    """Main application class"""
    
    def __init__(self):
        self.llm_config = LLMConfig()
        self.llm_map = self.llm_config.initialize_models()

        # Initialize FastAPI app with additional metadata
        self.app = FastAPI(
            title="Grammar Checking API",
            version="1.0",
            description="An API for checking and correcting grammar using LLMs with detailed corrections",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Setup services and routes
        self.grammar_service = GrammarService(self.llm_map)
        self.router = APIRouter(self.app, self.llm_map, self.grammar_service)
        self.router.setup_routes()

    def get_app(self):
        """Get the FastAPI app instance"""
        return self.app
# Create application instance

application = GrammarCheckAPI()
app = application.get_app()

# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "grammar_check_api:app",
        host="localhost", 
        port=8000,
        log_level="info",
        reload=True
    )

