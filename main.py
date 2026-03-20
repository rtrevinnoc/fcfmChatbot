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

# Global dictionary to hold multiple vector databases
vector_dbs = {}

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

# Static course plans sourced directly from faqUndergraduate.pdf (most recent plan per career).
# Keyed by lowercase keywords that identify each career in a user query.
FCFM_COURSE_PLANS = {
    "computacionales": {
        "nombre": "Licenciado en Ciencias Computacionales",
        "plan": "Plan 440",
        "semestres": {
            1: ["Álgebra", "Cálculo diferencial", "Geometría analítica", "Metodología de la programación", "Liderazgo emprendimiento e innovación", "Responsabilidad social y desarrollo"],
            2: ["Cálculo integral", "Mecánica traslacional y rotación", "Programación básica", "Tópicos de álgebra", "Igualdad de género diversidad y sexualidad", "Ética transparencia y cultura de la legalidad", "Cultura de paz y derechos humanos"],
            3: ["Matemáticas discretas", "Álgebra lineal", "Fundamentos de sistemas operativos", "Programación estructurada", "Laboratorio de programación estructurada", "Física para computación", "Laboratorio de física para computación"],
            4: ["Estructura de datos", "Teoría de autómatas", "Circuitos digitales", "Laboratorio de circuitos digitales", "Teoría de la información", "Fundamentos de redes", "Laboratorio de fundamentos de redes"],
            5: ["Bases de datos", "Laboratorio de bases de datos", "Teoría de la información aplicada", "Algoritmia y optimización", "Programación orientada a objetos", "Laboratorio de programación orientada a objetos", "Análisis numérico para programación"],
            6: ["Programación lineal", "Arquitectura computacional", "Inglés para tecnologías", "Inteligencia artificial"],
            7: ["Investigación de operaciones", "Cómputo en la nube", "Fundamentos de seguridad informática", "Compiladores", "Minería de datos"],
            8: ["Análisis de sistemas", "Investigación y desarrollo"],
            9: ["Cómputo de alto rendimiento", "Ingeniería de software", "Servicio social"],
            10: ["Taller para examen de egreso", "Seminario para el desempeño profesional", "Administración de proyectos tecnológicos", "Modelo de negocios", "Analítica de datos e inteligencia", "Transformación digital"],
            11: ["Estancias de investigación", "Certificación en tecnologías de información", "Prácticas profesionales"],
        }
    },
    "actuaría": {
        "nombre": "Licenciado en Actuaría",
        "plan": "Plan 430",
        "semestres": {
            1: ["Álgebra", "Cálculo diferencial", "Geometría analítica", "Metodología de la programación", "Responsabilidad social y desarrollo", "Cultura de paz"],
            2: ["Tópicos de álgebra", "Cálculo integral", "Mecánica traslacional y rotación", "Programación básica", "Ética y cultura de la legalidad", "Liderazgo emprendimiento e innovación", "Cultura de género"],
            3: ["Cálculo de varias variables", "Álgebra lineal", "Seguro de vida", "Matemáticas financieras", "Análisis de datos", "Probabilidad básica", "Contexto económico geopolítico"],
            4: ["Probabilidad avanzada", "Ecuaciones diferenciales", "Economía", "Seguro de daños", "Programación lineal", "Matemáticas financieras avanzada", "Contabilidad"],
            5: ["Investigación de operaciones", "Inferencia estadística", "Contabilidad de seguros", "Portafolio de inversión", "Cálculo actuarial", "Mercadotecnia", "Administración"],
            6: ["Simulación", "Procesos estocásticos", "Métodos estadísticos", "Planeación estratégica", "Matemáticas actuariales del seguro", "Comportamiento organizacional", "Legislación de seguros", "Modelado matemático", "Muestreo"],
            7: ["Estadística aplicada", "Demografía", "Productos financieros derivados", "Matemáticas actuariales del seguro", "Pensiones", "Econometría financiera", "Estadística multivariada", "Modelo de negocios", "Minería de datos"],
            8: ["Finanzas corporativas", "Administración del riesgo empresarial", "Teoría del riesgo", "Servicio social"],
            9: ["Administración actuarial", "Administración del riesgo empresarial", "Minería de datos", "Teoría del comportamiento", "Teoría de juegos", "Auditoría actuarial"],
            11: ["Seminario de investigación", "Prácticas profesionales"],
        }
    },
    "física": {
        "nombre": "Licenciado en Física",
        "plan": "Plan 430",
        "semestres": {
            1: ["Álgebra", "Cálculo diferencial", "Geometría analítica", "Metodología de la programación", "Responsabilidad social y desarrollo", "Cultura de paz"],
            2: ["Tópicos de álgebra", "Cálculo integral", "Programación básica", "Ética y cultura de la legalidad", "Mecánica traslacional y rotación", "Liderazgo emprendimiento e innovación", "Cultura de género"],
            3: ["Cálculo de varias variables", "Álgebra lineal", "Métodos diferenciales", "Gravitación fluidos y calor", "Laboratorio de gravitación fluidos y calor", "Probabilidad y estadística", "Lenguajes de programación"],
            4: ["Cálculo vectorial", "Variable compleja", "Electricidad", "Métodos numéricos", "Mecánica teórica", "Diseño experimental", "Didáctica de la física"],
            5: ["Métodos de la física teórica", "Termodinámica", "Ondas y magnetismo", "Laboratorio de ondas y magnetismo", "Mecánica de sistemas con restricciones", "Física moderna"],
            6: ["Cálculo variacional y tensorial", "Física estadística", "Teoría electrostática", "Introducción a la mecánica cuántica", "Circuitos eléctricos", "Física computacional"],
            7: ["Astrofísica de galaxias", "Práctica docente", "Relatividad", "Óptica y aplicaciones", "Ciencia de materiales", "Aplicaciones de física estadística", "Teoría electrodinámica", "Mecánica cuántica", "Electrónica", "Física experimental"],
            8: ["Servicio social", "Óptica", "Física instrumental"],
            9: ["Seminario para el desempeño profesional"],
            11: ["Seminario de investigación", "Prácticas profesionales"],
        }
    },
    "matemáticas": {
        "nombre": "Licenciado en Matemáticas",
        "plan": "Plan 430",
        "semestres": {
            1: ["Álgebra", "Cálculo diferencial", "Geometría analítica", "Metodología de la programación", "Responsabilidad social y desarrollo", "Cultura de paz"],
            2: ["Tópicos de álgebra", "Cálculo integral", "Programación básica", "Mecánica traslacional y rotación", "Ética y cultura de la legalidad", "Liderazgo emprendimiento e innovación", "Cultura de género"],
            3: ["Cálculo de varias variables", "Álgebra lineal", "Programación estructurada", "Laboratorio de programación estructurada", "Matemáticas discretas", "Geometría moderna"],
            4: ["Ecuaciones diferenciales", "Variable compleja", "Matemáticas computacionales", "Tópicos de álgebra lineal", "Cálculo vectorial", "Probabilidad"],
            5: ["Estadística", "Teoría de grupos", "Tópicos de ecuaciones diferenciales", "Historia de las matemáticas", "Tópicos de variable compleja"],
            6: ["Análisis matemático", "Minería de datos", "Programación lineal", "Teoría de anillos y campos", "Matemática educativa"],
            7: ["Tópicos de análisis matemático", "Topología", "Enseñanza de las ciencias físico matemáticas", "Investigación de operaciones", "Programación entera", "Estructura de datos", "Análisis numérico", "Teoría de grafos", "Muestreo", "Teoría de juegos", "Lógica y conjuntos", "Didáctica de las matemáticas"],
            8: ["Teoría de la medida", "Servicio social", "Simulación", "Tópicos de álgebra abstracta", "Diseño de experimentos", "Didáctica de las matemáticas"],
            9: ["Geometría diferencial", "Métodos de optimización", "Análisis funcional", "Investigación educativa", "Tópicos de topología", "Optimización de aplicaciones industriales", "Análisis de algoritmos"],
            11: ["Seminario de investigación", "Taller para examen de egreso", "Seminario para el desempeño profesional"],
        }
    },
    "multimedia": {
        "nombre": "Licenciado en Multimedia y Animación Digital",
        "plan": "Plan 440",
        "semestres": {
            1: ["Liderazgo emprendimiento e innovación", "Responsabilidad social y desarrollo", "Metodología de la programación", "Álgebra", "Cálculo diferencial", "Geometría analítica"],
            2: ["Cultura de paz y derechos humanos", "Ética transparencia y cultura de la legalidad", "Igualdad de género diversidad y sexualidad", "Tópicos de álgebra", "Cálculo integral", "Mecánica traslacional y rotación", "Programación básica"],
            3: ["Programación estructurada", "Laboratorio de programación estructurada", "Relaciones espaciales para video", "Producción multimedia", "Fundamentos del dibujo artístico", "Metodologías ágiles de trabajo", "Proyección de negocios tecnológicos", "Modelado arquitectónico"],
            4: ["Programación avanzada", "Transformaciones gráficas para videojuegos", "Fundamentos de los videojuegos", "Tecnologías multimedia", "Fotografía digital", "Modelo de administración de datos", "Laboratorio de modelo de administración de datos", "Fundamentos de la animación"],
            5: ["Interfaz y experiencia de usuario", "Lógica digital", "Producción de guiones", "Iluminación y audio", "Programación orientada a objetos", "Laboratorio de programación orientada a objetos"],
            6: ["Diseño de hápticos", "Redes computacionales", "Laboratorio de redes computacionales", "Fundamentos y producción cinematográfica"],
            7: ["Procesamiento de imágenes", "Programación de sitios web", "Laboratorio de programación de sitios web", "Introducción a efectos visuales", "Administración de proyectos multimedia", "Gráficas computacionales", "Optimización de videojuegos"],
            8: ["Interfaces y aplicaciones web", "Laboratorio de interfaces y aplicaciones web", "Gestión y producción de efectos visuales", "Propiedad intelectual", "Escenarios digitales", "Realidad virtual y aumentada", "Diseño de videojuegos en línea"],
            9: ["Inglés para tecnologías", "Mercadotecnia"],
            10: ["Postproducción para entornos virtuales", "Postproducción para series animadas", "Inteligencia artificial y ciencia de datos", "Seminario para el desempeño profesional", "Servicio social"],
        }
    },
    "seguridad": {
        "nombre": "Licenciado en Seguridad en Tecnologías de Información",
        "plan": "Plan 430",
        "semestres": {
            1: ["Cultura de paz", "Responsabilidad social y desarrollo", "Metodología de la programación", "Álgebra", "Cálculo diferencial", "Geometría analítica"],
            2: ["Liderazgo emprendimiento e innovación", "Ética y cultura de la legalidad", "Cultura de género", "Tópicos de álgebra", "Cálculo integral", "Mecánica traslacional y rotación", "Programación básica"],
            3: ["Álgebra lineal", "Matemáticas discretas", "Señales de transmisión", "Fundamentos de la seguridad informática", "Fundamentos de sistemas operativos", "Programación para ciberseguridad"],
            4: ["Teoría de autómatas", "Criptografía", "Teoría de la información", "Programa de seguridad", "Fundamentos de redes", "Laboratorio de fundamentos de redes", "Normatividad y regulaciones de datos"],
            5: ["Seguridad en base de datos", "Laboratorio de seguridad en base de datos", "Conmutación de redes locales", "Laboratorio de conmutación de redes locales", "Seguridad en aplicaciones", "Laboratorio de seguridad en aplicaciones", "Teoría de la información aplicada", "Administración de riesgos de seguridad"],
            6: ["Derecho informático", "Diseño de políticas de seguridad", "Inglés para tecnologías", "Interconexión de redes locales", "Laboratorio de interconexión de redes"],
            7: ["Control de accesos", "Laboratorio de control de accesos", "Diseño de arquitecturas de seguridad", "Operación de la seguridad", "Gobierno riesgo y cumplimiento"],
            8: ["Pruebas de vulnerabilidades", "Laboratorio de pruebas de vulnerabilidades", "Continuidad de negocio y recuperación ante desastres"],
            9: ["Cómputo forense", "Gestión de incidentes de seguridad", "Servicio social"],
            10: ["Modelo de negocios", "Amenazas operativas y dependencias empresariales", "Programa de amenazas internas", "Transformación digital"],
            11: ["Estancias de investigación", "Certificación en tecnologías de información", "Prácticas profesionales"],
        }
    },
}

def _find_career_plan(query: str) -> str | None:
    """Return a formatted course plan if the query mentions a specific FCFM career."""
    q = query.lower()
    for keyword, plan in FCFM_COURSE_PLANS.items():
        if keyword in q:
            lines = [f"Plan de estudios: {plan['nombre']} ({plan['plan']})"]
            for sem, materias in sorted(plan["semestres"].items()):
                lines.append(f"  Semestre {sem}: {', '.join(materias)}")
            return "\n".join(lines)
    return None


class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, profile: dict):
        self.llm = llm
        self.profile = profile
        # Determine which vector DB to use
        db_key = profile['level'] if profile['status'] == 'student' else profile['status']
        self.vector_db = vector_dbs.get(db_key)
        # Aspirants may ask about available careers; supplement with the undergraduate DB
        self.supplemental_db = vector_dbs.get("undergraduate") if db_key == "applying" else None

    async def arun_pipeline(self, user_query: str, user_id: str):
        # Retrieve context from the SPECIFIC vector DB
        context_prompt = ""
        if self.vector_db:
            context_list = self.vector_db.search_by_text(user_query, k=4)
            context_prompt = "\n".join([context[0] for context in context_list])
        # For aspirants: if the query mentions a specific career, inject the exact
        # structured course plan (avoids cross-career contamination from vector search).
        # For other cross-cutting questions, fall back to supplemental vector search.
        if self.supplemental_db:
            career_plan = _find_career_plan(user_query)
            if career_plan:
                context_prompt = (context_prompt + "\n" + career_plan) if context_prompt else career_plan
            else:
                supplemental_list = self.supplemental_db.search_by_text(user_query, k=4)
                supplemental_text = "\n".join([context[0] for context in supplemental_list])
                if supplemental_text:
                    context_prompt = (context_prompt + "\n" + supplemental_text) if context_prompt else supplemental_text

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

        # Static facts guaranteed to be correct for FCFM — injected so the model
        # never has to guess them from a potentially-missing context chunk.
        FCFM_CAREERS = (
            "Las carreras que ofrece la FCFM son:\n"
            "• Licenciado en Actuaría\n"
            "• Licenciado en Ciencias Computacionales\n"
            "• Licenciado en Física\n"
            "• Licenciado en Matemáticas\n"
            "• Licenciado en Multimedia y Animación Digital\n"
            "• Licenciado en Seguridad en Tecnologías de Información"
        )

        static_facts = FCFM_CAREERS if (self.profile['level'] or self.profile['status']) in ("applying", "undergraduate") else ""

        messages = [
            {
                "role": "system",
                "content": (
                    f"Eres un {current_role} en la Facultad de Ciencias Físico Matemáticas (FCFM) de la Universidad Autónoma de Nuevo León en Monterrey, México. "
                    f"Tu objetivo es ayudar exclusivamente con temas de {current_topic}. "
                    "IMPORTANTE: Responde ÚNICAMENTE con base en el contexto y los hechos conocidos proporcionados a continuación. "
                    "Si la información solicitada no se encuentra en el contexto ni en los hechos conocidos, indícalo claramente y sugiere al usuario que visite la página oficial de la UANL o se comunique directamente con la facultad. "
                    "NO inventes ni supongas información que no esté en el contexto o los hechos conocidos. "
                    "NO menciones carreras, programas ni servicios de otras facultades a menos que el contexto lo indique explícitamente. "
                    "Realiza preguntas para averiguar lo que busca el usuario cuando no cuentes con información suficiente. "
                    f"{current_omissions}. "
                    "Responde siempre en español. Sé conciso pero muestra toda la información relevante con la que cuentes. Mantén la conversación sencilla y haz solo una pregunta a la vez.\n\n"
                    + (f"Hechos conocidos de la FCFM:\n{static_facts}\n\n" if static_facts else "")
                    + f"Contexto relevante:\n{context_prompt}"
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