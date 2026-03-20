import asyncio
import os, math, sqlite3, secrets
from fastapi import FastAPI, Request, Query, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse, JSONResponse, Response, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from flask import request
from twilio.twiml.messaging_response import MessagingResponse
from utils.text_utils import CharacterTextSplitter, TextFileLoader, PDFLoader
from utils.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from utils.vectordatabase import VectorDatabase
from utils.openai_utils.chatmodel import ChatOpenAI
from utils.web_scraper import scrape_program_pages
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

PLATFORM = os.getenv("MESSAGING_PLATFORM", "twilio").lower()
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD")
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

app = FastAPI()

# Configuration for different segments
FILES_MAP = {
    "applying": "faqAspirantes.pdf",
    "enrolled": "faqReingreso.pdf",
    "undergraduate": "faqUndergraduate.pdf",
    "graduate": "faqGraduate.pdf",
    "alumni": "faqEgresados.pdf"
}

# Global dictionary to hold multiple vector databases (one per user segment)
vector_dbs = {}

# Programs vector DB — built at startup from three sources and refreshed daily:
#   1. materias/*.txt   — authoritative semester-by-semester course plans
#   2. downloaded_pdfs/ — plan de estudios / malla curricular PDFs from UANL
#   3. live web pages   — program descriptions, requirements, career fields
# Used as a supplemental source for aspirant and undergraduate queries.
programs_db: VectorDatabase | None = None
PROGRAMS_DB_REFRESH_HOURS = 24
MATERIAS_DIR = "materias"

DB_PATH = "chat_history.db"

templates = Jinja2Templates(directory="templates")

security = HTTPBasic()
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

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

    # NEW: Admin Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            username TEXT PRIMARY KEY,
            hashed_password TEXT NOT NULL
        )
    """)
    # Check if we need a default admin
    c.execute("SELECT COUNT(*) FROM admins")
    if c.fetchone()[0] == 0:
        default_pw = pwd_context.hash(DEFAULT_ADMIN_PASSWORD)
        c.execute("INSERT INTO admins (username, hashed_password) VALUES (?, ?)", ("admin", default_pw))
        print("Default admin created: admin")

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

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT hashed_password FROM admins WHERE username = ?", (credentials.username,))
    row = c.fetchone()
    conn.close()

    if row is None or not verify_password(credentials.password, row[0]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, profile: dict):
        self.llm = llm
        self.profile = profile
        # Determine which FAQ vector DB to use for this user segment
        db_key = profile['level'] if profile['status'] == 'student' else profile['status']
        self.vector_db = vector_dbs.get(db_key)
        # Aspirants and undergraduate students may ask about programs, careers,
        # and course plans; supplement with programs_db (materias + PDFs + web).
        self.use_programs_db = db_key in ("applying", "undergraduate")

    async def arun_pipeline(self, user_query: str, user_id: str):
        # Retrieve context from the segment-specific FAQ vector DB
        context_prompt = ""
        if self.vector_db:
            context_list = self.vector_db.search_by_text(user_query, k=4)
            context_prompt = "\n".join([context[0] for context in context_list])

        # For aspirants and undergrads, also search the programs DB
        # (materias course plans + downloaded PDFs + live web program pages)
        if self.use_programs_db and programs_db:
            prog_results = programs_db.search_by_text(user_query, k=6)
            prog_text = "\n".join([r[0] for r in prog_results])
            if prog_text:
                context_prompt = (context_prompt + "\n" + prog_text) if context_prompt else prog_text

        history = get_user_history(user_id)
        
        # Dynamic System Prompt based on situation
        role_desc = {
            "applying": "asistente de ADMISIONES para nuevos aspirantes",
            "enrolled": "asistente de INSCRIPCIONES para alumnos de reingreso",
            "undergraduate": "asistente académico de LICENCIATURA",
            "graduate": "asistente académico de POSGRADO",
            "alumni": "asistente de TRAMITES Y TITULACION para egresados"
        }

        current_role = role_desc.get(self.profile['level'] or self.profile['status'], "asistente administrativo")

        role_topic = {
            "applying": "admisión y proceso de ingreso para aspirantes",
            "enrolled": "reinscripción e inscripciones para alumnos de reingreso",
            "undergraduate": "licenciatura",
            "graduate": "posgrado",
            "alumni": "trámites de titulación y servicios para egresados"
        }

        current_topic = role_topic.get(self.profile['level'] or self.profile['status'], "atención a la comunidad universitaria")

        omit_desc = {
            "applying": "No incluyas información sobre costos de colegiaturas ni pagos que no correspondan al proceso de admisión",
            "enrolled": "No incluyas información promocional. El alumno ya está inscrito, evita hablar de costos de nueva admisión",
            "undergraduate": "No incluyas información de posgrado ya que el estudiante está inscrito en licenciatura. Evita mencionar costos de nueva admisión o promociones",
            "graduate": "No incluyas información de licenciatura ya que el estudiante está inscrito en posgrado. Evita mencionar costos de nueva admisión o promociones",
            "alumni": "El usuario ya terminó sus estudios, enfócate en trámites de titulación o servicios para ex-alumnos"
        }

        current_omissions = omit_desc.get(self.profile['level'] or self.profile['status'], "")

        messages = [
            {
                "role": "system",
                "content": (
                    f"Eres un {current_role} en la Facultad de Ciencias Físico Matemáticas (FCFM) de la Universidad Autónoma de Nuevo León en Monterrey, México. "
                    f"Tu objetivo es ayudar exclusivamente con temas de {current_topic}. "
                    "IMPORTANTE: Responde ÚNICAMENTE con base en el contexto proporcionado a continuación. "
                    "Si la información solicitada no se encuentra en el contexto, indícalo claramente y sugiere al usuario que visite la página oficial de la UANL o se comunique directamente con la facultad. "
                    "NO inventes ni supongas información que no esté en el contexto. "
                    "NO menciones carreras, programas ni servicios de otras facultades a menos que el contexto lo indique explícitamente. "
                    "El contexto puede incluir tablas de planes de estudio con el formato: 'código nombre_materia requisito créditos'. "
                    "Cuando el contexto contenga ese formato, extrae los nombres de las materias y preséntalos ordenados por semestre, ignorando los códigos numéricos. "
                    "Realiza preguntas para averiguar lo que busca el usuario cuando no cuentes con información suficiente. "
                    f"{current_omissions}. "
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

async def build_programs_db() -> VectorDatabase | None:
    """
    Build the programs VectorDatabase from three independent sources.
    Each source is wrapped in its own try/except so a network failure
    never prevents the local materias files from being indexed.

      1. materias/*.txt  — semester-by-semester course plans (local, always first)
      2. live web pages  — program descriptions, requirements, career fields
      3. downloaded PDFs — plan de estudios / malla curricular from UANL
    """
    # Large chunks keep carreras_fcfm.txt in one piece and preserve whole
    # semester blocks inside each materias file instead of slicing headers.
    splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    all_chunks: list[str] = []

    # ── Source 1: materias/*.txt (local — must never fail) ────────────────
    try:
        if os.path.isdir(MATERIAS_DIR):
            materias_loader = TextFileLoader(MATERIAS_DIR)
            materias_docs = materias_loader.load_documents()
            all_chunks.extend(splitter.split_texts(materias_docs))
            print(f"[ProgramsDB] Loaded {len(materias_docs)} files from {MATERIAS_DIR}/")
        else:
            print(f"[ProgramsDB] Warning: {MATERIAS_DIR}/ not found")
    except Exception as exc:
        print(f"[ProgramsDB] Error loading materias/: {exc}")

    # ── Sources 2 & 3: web scraping + PDF downloads (network — may fail) ──
    try:
        web_docs, pdf_paths = await scrape_program_pages()

        if web_docs:
            all_chunks.extend(splitter.split_texts(web_docs))

        for pdf_path in pdf_paths:
            try:
                pdf_loader = PDFLoader(pdf_path)
                pdf_docs = pdf_loader.load_documents()
                all_chunks.extend(splitter.split_texts(pdf_docs))
            except Exception as exc:
                print(f"[ProgramsDB] Could not load PDF {pdf_path}: {exc}")

    except Exception as exc:
        print(f"[ProgramsDB] Web scraping failed (materias data still loaded): {exc}")

    if not all_chunks:
        print("[ProgramsDB] No content collected — programs_db not updated")
        return None

    db = VectorDatabase()
    await db.abuild_from_list(all_chunks)
    print(f"[ProgramsDB] Built with {len(all_chunks)} chunks total")
    return db


async def _programs_db_refresh_loop():
    """Background task: rebuild programs_db every PROGRAMS_DB_REFRESH_HOURS hours."""
    global programs_db
    while True:
        await asyncio.sleep(PROGRAMS_DB_REFRESH_HOURS * 3600)
        print("[ProgramsDB] Starting scheduled refresh...")
        fresh = await build_programs_db()
        if fresh:
            programs_db = fresh


@app.on_event("startup")
async def startup_event():
    global vector_dbs, programs_db
    # Build FAQ vector DBs from local PDFs
    for key, path in FILES_MAP.items():
        if os.path.exists(path):
            vector_dbs[key] = await prepare_vector_db(path)
    print("All FAQ Vector DBs initialized")
    # Build programs DB (materias + downloaded PDFs + live web)
    programs_db = await build_programs_db()
    # Schedule daily refresh in the background
    asyncio.create_task(_programs_db_refresh_loop())

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
            "2️⃣\tYa soy alumno, quiero consultar acerca del reingreso\n"
            "3️⃣\tSoy estudiante (Consulta de materias/plan)\n"
            "4️⃣\tYa egresé (Trámites de titulación)"
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
            return await respond_to_platform(user_id, "Perfecto. Dime tus dudas sobre el reingreso.")

        elif "3" in incoming_msg or "estudiante" in incoming_msg.lower():
            update_user_profile(user_id, status="student", step=2)
            return await respond_to_platform(user_id, "Excelente. ¿En qué grado estás?\n1️⃣\tLicenciatura\n2️⃣\tPosgrado")
        
        elif "4" in incoming_msg or "egresado" in incoming_msg.lower() or "egresé" in incoming_msg.lower():
            update_user_profile(user_id, status="alumni", step=3)
            return await respond_to_platform(user_id, "¡Felicidades por egresar! ¿En qué trámite o consulta te puedo ayudar hoy?")
        
        else:
            return await respond_to_platform(user_id, "Por favor, selecciona una opción válida (1, 2, 3 o 4).")

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
async def admin_dashboard(request: Request, username: str = Depends(get_current_username)):
    # We pass the 'request' object because Jinja2 requires it
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/api/profiles")
async def get_all_profiles(
    page: int = 1, 
    size: int = 10, 
    username: str = Depends(get_current_username)
):
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