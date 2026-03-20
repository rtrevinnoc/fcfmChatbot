"""
web_scraper.py — fetches FCFM/UANL program pages, extracts and downloads
linked PDFs, and returns clean text documents for embedding.

Architecture
============
The UANL program-listing index pages are rendered by JavaScript, so a plain
httpx GET returns an empty shell.  We therefore keep a curated list of known
individual program page URLs (stable; content changes, not URLs) and fetch
their HTML directly.

From each program page we also extract any linked PDF files (plan de
estudios, malla curricular, unidades optativas) and download them to a local
cache directory so they can be indexed alongside the materias/*.txt files.
"""

import asyncio
import html as html_lib
import os
import re
from typing import List, Tuple

import httpx

# ── Known FCFM program page URLs ─────────────────────────────────────────────
# Index pages are JS-rendered and unreachable by httpx; we maintain this list
# directly.  The *content* of each page is still fetched live on every refresh.
KNOWN_PROGRAM_URLS: List[str] = [
    # ── Undergraduate ──────────────────────────────────────────────────────
    "https://www.uanl.mx/oferta/licenciatura-en-actuaria/",
    "https://www.uanl.mx/oferta/licenciado-en-ciencias-computacionales/",
    "https://www.uanl.mx/oferta/licenciatura-en-ciencias-computacionales/",  # no-escolarizada variant
    "https://www.uanl.mx/oferta/licenciado-en-fisica/",
    "https://www.uanl.mx/oferta/licenciado-en-matematicas/",
    "https://www.uanl.mx/oferta/licenciatura-en-multimedia-y-animacion-digital/",
    "https://www.uanl.mx/oferta/licenciado-en-seguridad-en-tecnologias-de-informacion/",
    # ── Master's ───────────────────────────────────────────────────────────
    "https://posgrado.uanl.mx/ofertas_educativas/maestria-en-ciencias-con-orientacion-en-matematicas/",
    "https://posgrado.uanl.mx/ofertas_educativas/maestria-en-ciencia-de-datos/",
    "https://posgrado.uanl.mx/ofertas_educativas/maestria-en-ingenieria-en-seguridad-de-la-informacion/",
    "https://posgrado.uanl.mx/ofertas_educativas/maestria-en-ingenieria-fisica-industrial/",
    "https://posgrado.uanl.mx/ofertas_educativas/maestria-en-astrofisica-planetaria-y-tecnologias-afines/",
    # ── Doctorates ────────────────────────────────────────────────────────
    "https://posgrado.uanl.mx/ofertas_educativas/doctorado-en-ingenieria-fisica/",
    "https://posgrado.uanl.mx/ofertas_educativas/doctorado-en-ciencias-con-orientacion-en-matematicas/",
]

# Regex patterns for PDF links embedded in program pages
_PDF_URL_PATTERNS: List[str] = [
    r'https://www\.uanl\.mx/wp-content/uploads/[^\s"\'<>]+\.pdf',
    r'https://posgrado\.uanl\.mx/wp-content/uploads/[^\s"\'<>]+\.pdf',
]

_HTTP_HEADERS = {"User-Agent": "FCFM-Chatbot/1.0 (+https://www.fcfm.uanl.mx)"}
_TIMEOUT = 20.0
PDF_CACHE_DIR = "downloaded_pdfs"


# ── HTML → plain text ─────────────────────────────────────────────────────────

def html_to_text(raw_html: str) -> str:
    """Convert raw HTML to clean, readable plain text."""
    text = re.sub(r"<!--.*?-->", "", raw_html, flags=re.DOTALL)
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>", "", text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<(br|p|div|section|article|h[1-6]|li|tr|td|th)[^>]*>",
        "\n", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── PDF URL extraction ────────────────────────────────────────────────────────

def extract_pdf_urls(raw_html: str) -> List[str]:
    """Return unique PDF URLs found in the raw HTML of a program page."""
    found: set[str] = set()
    for pattern in _PDF_URL_PATTERNS:
        found.update(re.findall(pattern, raw_html))
    return list(found)


# ── Async fetching ────────────────────────────────────────────────────────────

async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    """GET *url* and return raw HTML as text, or '' on any error."""
    try:
        resp = await client.get(url, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        print(f"[WebScraper] Could not fetch {url}: {exc}")
        return ""


async def _download_pdf(client: httpx.AsyncClient, url: str) -> str | None:
    """
    Download a PDF from *url* into PDF_CACHE_DIR.
    Returns the local file path on success, None on failure.
    Skips download if a cached copy already exists.
    """
    os.makedirs(PDF_CACHE_DIR, exist_ok=True)
    # Derive a safe filename from the URL
    filename = re.sub(r"[^a-zA-Z0-9_\-.]", "_", url.split("/")[-1])
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    dest = os.path.join(PDF_CACHE_DIR, filename)

    if os.path.exists(dest):
        return dest  # already cached

    try:
        resp = await client.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        print(f"[WebScraper] Downloaded {url} → {dest}")
        return dest
    except Exception as exc:
        print(f"[WebScraper] Could not download PDF {url}: {exc}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

async def scrape_program_pages() -> Tuple[List[str], List[str]]:
    """
    Fetch all known FCFM program pages, extract and download their linked PDFs.

    Returns
    -------
    documents : list[str]
        Plain-text documents (one per program page), labeled with their source
        URL.  Ready to be chunked and embedded.
    pdf_paths : list[str]
        Local filesystem paths of successfully downloaded PDFs.
    """
    documents: List[str] = []
    pdf_paths: List[str] = []
    all_pdf_urls: set[str] = set()

    async with httpx.AsyncClient(headers=_HTTP_HEADERS) as client:
        # ── Step 1: fetch each known program page ────────────────────────
        raw_pages = await asyncio.gather(
            *[_fetch_text(client, url) for url in KNOWN_PROGRAM_URLS]
        )

        for url, raw in zip(KNOWN_PROGRAM_URLS, raw_pages):
            if not raw:
                continue
            # Collect text document
            text = html_to_text(raw)
            if text:
                documents.append(f"[Fuente: {url}]\n{text}")
            # Collect PDF URLs found on this page
            all_pdf_urls.update(extract_pdf_urls(raw))

        # ── Step 2: download all discovered PDFs ─────────────────────────
        if all_pdf_urls:
            results = await asyncio.gather(
                *[_download_pdf(client, u) for u in all_pdf_urls]
            )
            pdf_paths = [p for p in results if p]

    print(
        f"[WebScraper] Fetched {len(documents)} program page documents, "
        f"downloaded {len(pdf_paths)}/{len(all_pdf_urls)} PDFs"
    )
    return documents, pdf_paths
