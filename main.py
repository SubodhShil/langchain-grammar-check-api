from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from grammar_check_api import GrammarCheckAPI
from chat_with_ai_api import ChatWithAI
from text_paraphraser import TextParaphraser

# Create the application instances
grammar_app_instance = GrammarCheckAPI()
chat_app_instance = ChatWithAI()
paraphraser_app_instance = TextParaphraser()

# Create a new FastAPI app that will combine both APIs
app = FastAPI(
    title="AI Language Services API",
    version="1.0",
    description="All AI based API's",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the grammar API
grammar_app = grammar_app_instance.get_app()
app.mount("/grammar", grammar_app)

# Mount the chat API
chat_app = chat_app_instance.get_app()
app.mount("/chat", chat_app)

# Mount the paraphraser API
paraphraser_app = paraphraser_app_instance.get_app()
app.mount("/paraphraser", paraphraser_app)

# This allows the app to be run with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)