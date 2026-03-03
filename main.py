import os, math, sqlite3
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse, JSONResponse, Response, HTMLResponse
from fastapi.templating import Jinja2Templates
from twilio.twiml.messaging_response import MessagingResponse
from utils.text_utils import CharacterTextSplitter, TextFileLoader, PDFLoader
from utils.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from utils.vectordatabase import VectorDatabase
from utils.openai_utils.chatmodel import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

PLATFORM = os.getenv("MESSAGING_PLATFORM", "twilio").lower()
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

app = FastAPI()

# Configuration for different segments
FILES_MAP = {
    "applying": "catalog.pdf",
    "enrolled": "catalog.pdf",
    "undergraduate": "catalog.pdf",
    "graduate": "catalog.pdf"
}

# Global dictionary to hold multiple vector databases
vector_dbs = {}

DB_PATH = "chat_history.db"

templates = Jinja2Templates(directory="templates")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Existing history table
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # New table for User State
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            status TEXT, -- applying, enrolled, student
            level TEXT,  -- undergraduate, graduate
            step INTEGER DEFAULT 0
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

def get_user_profile(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT status, level, step FROM user_profiles WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"status": row[0], "level": row[1], "step": row[2]}
    return None

def update_user_profile(user_id: str, status=None, level=None, step=0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_profiles (user_id, status, level, step) 
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET 
            status=COALESCE(?, status), 
            level=COALESCE(?, level), 
            step=?
    """, (user_id, status, level, step, status, level, step))
    conn.commit()
    conn.close()

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, profile: dict):
        self.llm = llm
        self.profile = profile
        # Determine which vector DB to use
        db_key = profile['level'] if profile['status'] == 'student' else profile['status']
        self.vector_db = vector_dbs.get(db_key)

    async def arun_pipeline(self, user_query: str, user_id: str):
        # Retrieve context from the SPECIFIC vector DB
        context_prompt = ""
        if self.vector_db:
            context_list = self.vector_db.search_by_text(user_query, k=4)
            context_prompt = "\n".join([context[0] for context in context_list])

        history = get_user_history(user_id)
        
        # Dynamic System Prompt based on situation
        role_desc = {
            "applying": "asistente de ADMISIONES para nuevos aspirantes",
            "enrolled": "asistente de INSCRIPCIONES para alumnos de reingreso",
            "undergraduate": "asistente académico de LICENCIATURA",
            "graduate": "asistente académico de POSGRADO"
        }
        
        current_role = role_desc.get(self.profile['level'] or self.profile['status'], "asistente administrativo")

        omit_desc = {
            "applying": "No incluyas información acerca de precios",
            "enrolled": "No inlcuyas información promocional ni de costos, el alumno ya está inscrito",
            "undergraduate": "No incluyas información acerca de precios, ni promociones, ni información de posgrado ya que el estudiante ya está inscrito en licenciatura",
            "graduate": "No incluyas información de precios, ni promociones, ni información de licenciatura ya que el estudiante ya está inscrito en posgrado"
        }
        
        current_omissions = omit_desc.get(self.profile['level'] or self.profile['status'], "")

        messages = [
            {
                "role": "system",
                "content": (
                    f"Eres un {current_role} en la Facultad de Ciencias Físico Matemáticas de la Universidad Autónoma de Nuevo León en Monterrey, México. "
                    f"Tu objetivo es ayudar exclusivamente con temas de {self.profile['status']}. "
                    "Realiza preguntas para averiguar lo que busca el usuario cuando no cuentes con información suficiente, ya que puede no saber qué es lo que necesita y debes apoyarlo."
                    f"{current_omissions}."
                    "Responde siempre en español. Sé conciso pero muestra toda la información relevante con la que cuentes. Mantén la conversación sencilla y haz solo una pregunta a la vez.\n\n"
                    f"Contexto relevante:\n{context_prompt}"
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
    global vector_dbs
    for key, path in FILES_MAP.items():
        if os.path.exists(path):
            vector_dbs[key] = await prepare_vector_db(path)
    print("All Vector DBs initialized")

@app.get("/webhook")
async def verify(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge")
):
    if PLATFORM == "meta" and mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge)
    return PlainTextResponse(content="Forbidden", status_code=403)

# Helper to keep the code clean
async def respond_to_platform(user_id, text):
    if PLATFORM == "meta":
        await send_meta_message(user_id, text)
        return JSONResponse({"status": "ok"})
    else:
        twiml_resp = MessagingResponse()
        twiml_resp.message(text)
        return Response(content=str(twiml_resp), media_type="application/xml")

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

    profile = get_user_profile(user_id)

    # STEP 0: New User - Send Main Menu
    if not profile or profile['step'] == 0:
        menu_text = (
            "¡Hola! Para ayudarte mejor, selecciona tu situación actual:\n"
            "1️⃣\tEstoy aplicando (Aspirante)\n"
            "2️⃣\tYa soy alumno y quiero inscribirme\n"
            "3️⃣\tSoy estudiante (Consulta de materias/plan)\n"
            "\nConsulta nuestro aviso de privacidad en https://www.uanl.mx/aviso-de-privacidad/"
        )
        update_user_profile(user_id, step=1)
        return await respond_to_platform(user_id, menu_text)

    # STEP 1: Handle Main Menu Response
    if profile['step'] == 1:
        if "1" in incoming_msg or "aplicando" in incoming_msg.lower():
            update_user_profile(user_id, status="applying", step=3)
            return await respond_to_platform(user_id, "Entendido. ¿Qué dudas tienes sobre el proceso de admisión?")
        
        elif "2" in incoming_msg or "inscribirme" in incoming_msg.lower():
            update_user_profile(user_id, status="enrolled", step=3)
            return await respond_to_platform(user_id, "Perfecto. Dime tus dudas sobre inscripciones y cuotas.")

        elif "3" in incoming_msg or "estudiante" in incoming_msg.lower():
            update_user_profile(user_id, status="student", step=2)
            return await respond_to_platform(user_id, "Excelente. ¿En qué grado estás?\n1️⃣\tLicenciatura\n2️⃣\tPosgrado")
        
        else:
            return await respond_to_platform(user_id, "Por favor, selecciona una opción válida (1 o 2).")

    # STEP 2: Handle Student Level (Undergrad vs Grad)
    if profile['step'] == 2:
        if "1" in incoming_msg or "licenciatura" in incoming_msg.lower():
            update_user_profile(user_id, level="undergraduate", step=3)
            return await respond_to_platform(user_id, "¿En qué puedo ayudarte acerca de licenciatura?")
        elif "2" in incoming_msg or "posgrado" in incoming_msg.lower():
            update_user_profile(user_id, level="graduate", step=3)
            return await respond_to_platform(user_id, "¿Qué dudas tienes respecto al posgrado?")
        else:
            return await respond_to_platform(user_id, "Por favor selecciona 1 para Licenciatura o 2 para Posgrado.")

    # STEP 3: Normal RAG Flow
    async with ChatOpenAI() as chat_openai:
        qa_pipeline = RetrievalAugmentedQAPipeline(llm=chat_openai, profile=profile)
        response_text = await qa_pipeline.arun_pipeline(incoming_msg, user_id)
        return await respond_to_platform(user_id, response_text)

async def send_meta_message(recipient_id: str, text: str):
    url = f"https://graph.facebook.com/v19.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={
            "recipient": {"id": recipient_id},
            "message": {"text": text}
        })

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    # We pass the 'request' object because Jinja2 requires it
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/api/profiles")
async def get_all_profiles(page: int = 1, size: int = 10):
    offset = (page - 1) * size
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get total count for pagination math
    c.execute("SELECT COUNT(*) FROM user_profiles")
    total_records = c.fetchone()[0]
    total_pages = math.ceil(total_records / size)

    # Get paginated data with message counts
    c.execute("""
        SELECT p.user_id, p.status, p.level, p.step, COUNT(h.id) 
        FROM user_profiles p
        LEFT JOIN history h ON p.user_id = h.user_id
        GROUP BY p.user_id
        ORDER BY p.user_id DESC
        LIMIT ? OFFSET ?
    """, (size, offset))
    
    rows = c.fetchall()
    conn.close()

    profiles = [
        {"user_id": r[0], "status": r[1], "level": r[2], "step": r[3], "messages": r[4]}
        for r in rows
    ]
    
    return {
        "profiles": profiles,
        "total_pages": total_pages,
        "current_page": page,
        "total_records": total_records
    }