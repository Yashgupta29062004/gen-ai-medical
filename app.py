import os
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# LangChain / AI Imports
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt
from fastapi.staticfiles import StaticFiles
# 1. Initialize App and Environment
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
load_dotenv()


# Setup Templates (Assumes your chat.html is in a 'templates' folder)
templates = Jinja2Templates(directory="template")

# 2. Environment Variables
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["GEMINI_API_KEY"]=os.getenv('GEMINI_API_KEY')

gemini_key = os.getenv("GEMINI_API_KEY")

# 3. Initialize RAG Components
embeddings = download_embedding()
index_name = "medibot"

# Connect to existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key
else:
    print(" Error: GEMINI_API_KEY not found in .env file")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Build the Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Renders the main chat page."""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get")
async def chat(msg: str = Form(...)):
    """
    Handles chat messages via Form Data (to match your Flask logic).
    If your frontend sends JSON, we would use a Pydantic model instead.
    """
    try:
        print(f"User Input: {msg}")
        
        # Invoke the RAG chain
        response = rag_chain.invoke({"input": msg})
        
        answer = response["answer"]
        print(f"Response: {answer}")
        
        return {"answer": str(answer)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    # Run the app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)