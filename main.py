import os
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse, JSONResponse, Response
from twilio.twiml.messaging_response import MessagingResponse
from utils.text_utils import CharacterTextSplitter, TextFileLoader, PDFLoader
from utils.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from utils.vectordatabase import VectorDatabase
from utils.openai_utils.chatmodel import ChatOpenAI
import sqlite3
from dotenv import load_dotenv

load_dotenv()

PLATFORM = os.getenv("MESSAGING_PLATFORM", "twilio").lower()
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

app = FastAPI()

vector_db = None  # global placeholder
PDF_FILE_PATH = "catalog.pdf"
DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_user_history(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM history WHERE user_id = ? ORDER BY timestamp ASC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in rows]

def add_to_history(user_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
    conn.commit()
    conn.close()

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, vector_db_retriever: VectorDatabase):
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    async def arun_pipeline(self, user_query: str, user_id: str):
        # Retrieve catalog context
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)
        context_prompt = "\n".join([context[0] for context in context_list])

        # Load past conversation from DB
        history = get_user_history(user_id)

        # Build the messages list
        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en la administración de la Facultad de Ciencias Físico Matemáticas de la Universidad Autónoma de Nuevo León en Monterrey, México. Utiliza la información provista para responder preguntas a alumnos sobre los procesos de inscripción y materías."
                    "Realiza preguntas para averiguar lo que busca el usuario cuando no cuentes con información suficiente, ya que puede no saber qué es lo que necesita y debes apoyarlo."
                    "Responde siempre en español. Mantén la conversación sencilla y haz solo una pregunta a la vez.\n\n"
                    f"Contexto:\n{context_prompt}"
                )
            }
        ] + history + [{"role": "user", "content": user_query}]

        # Generate response
        response_chunks = []
        async for chunk in self.llm.astream(messages):
            response_chunks.append(chunk)
        final_response = "".join(response_chunks)

        # Store new messages in DB
        add_to_history(user_id, "user", user_query)
        add_to_history(user_id, "assistant", final_response)

        return final_response

async def prepare_vector_db(file_path):
    print(f"Processing file: {file_path}")
    if file_path.lower().endswith('.pdf'):
        loader = PDFLoader(file_path)
    else:
        loader = TextFileLoader(file_path)

    documents = loader.load_documents()
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_texts(documents)

    print(f"Loaded {len(texts)} text chunks")

    vector_db = VectorDatabase()
    await vector_db.abuild_from_list(texts)
    return vector_db

@app.on_event("startup")
async def startup_event():
    global vector_db
    vector_db = await prepare_vector_db(PDF_FILE_PATH)
    print("Vector DB initialized")

@app.get("/webhook")
async def verify(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge")
):
    if PLATFORM == "meta" and mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge)
    return PlainTextResponse(content="Forbidden", status_code=403)

# 2. Unified Webhook (POST)
@app.post("/webhook")
async def unified_webhook(request: Request):
    user_id = None
    incoming_msg = None

    if PLATFORM == "meta":
        # Meta JSON Logic
        data = await request.json()
        try:
            entry = data["entry"][0]["messaging"][0]
            user_id = entry["sender"]["id"]
            incoming_msg = entry["message"].get("text")
        except (KeyError, IndexError):
            return JSONResponse({"status": "ignored"})

    else:
        # Twilio Form Logic
        form = await request.form()
        user_id = form.get("From")
        incoming_msg = form.get("Body", "").strip()

    if not incoming_msg:
        return Response(status_code=200)

    # --- SHARED RAG PIPELINE ---
    async with ChatOpenAI() as chat_openai:
        qa_pipeline = RetrievalAugmentedQAPipeline(
            llm=chat_openai,
            vector_db_retriever=vector_db
        )
        response_text = await qa_pipeline.arun_pipeline(incoming_msg, user_id)

    # --- CONDITIONAL RESPONSE ---
    if PLATFORM == "meta":
        await send_meta_message(user_id, response_text)
        return JSONResponse({"status": "ok"})
    else:
        twiml_resp = MessagingResponse()
        twiml_resp.message(response_text)
        return Response(content=str(twiml_resp), media_type="application/xml")

async def send_meta_message(recipient_id: str, text: str):
    url = f"https://graph.facebook.com/v19.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={
            "recipient": {"id": recipient_id},
            "message": {"text": text}
        })