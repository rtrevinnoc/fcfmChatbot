"""
web_scraper.py — fetches FCFM/UANL program pages and returns clean text
documents suitable for embedding into the web_db VectorDatabase.

Discovery flow:
  1. Fetch the three index pages (undergrad list, maestría list, doctorado list).
  2. Extract individual program page URLs via regex.
  3. Fetch every individual program page in parallel.
  4. Convert HTML → plain text and return one labeled document per page.
"""

import asyncio
import html as html_lib
import re
from typing import List

import httpx

# ── Index pages that list all FCFM programs ─────────────────────────────────
PROGRAM_INDEX_URLS: List[str] = [
    # Undergraduate
    (
        "https://www.uanl.mx/oferta/"
        "?search_esc_facu=Facultad+de+Ciencias+F%C3%ADsico+Matem%C3%A1ticas"
        "&search_esc_area=&search="
    ),
    # Master's
    (
        "https://posgrado.uanl.mx/oferta-educativa/"
        "?keyword=&select_facultades=facultad-de-ciencias-fisico-matematicas"
        "&select_niveles=maestria&select_modalidades=0"
        "&select_caracteristicas=0&search="
    ),
    # Doctorate
    (
        "https://posgrado.uanl.mx/oferta-educativa/"
        "?keyword=&select_facultades=facultad-de-ciencias-fisico-matematicas"
        "&select_niveles=doctorado&select_modalidades=0"
        "&select_caracteristicas=0&search="
    ),
]

# ── Regex patterns for individual program pages ──────────────────────────────
_PROGRAM_URL_PATTERNS: List[str] = [
    r"https://www\.uanl\.mx/oferta/[a-z0-9][a-z0-9\-]+/",
    r"https://posgrado\.uanl\.mx/ofertas_educativas/[a-z0-9][a-z0-9\-]+/",
]

_HTTP_HEADERS = {"User-Agent": "FCFM-Chatbot/1.0 (+https://www.fcfm.uanl.mx)"}
_TIMEOUT = 20.0


# ── HTML → plain text ────────────────────────────────────────────────────────

def html_to_text(raw_html: str) -> str:
    """Convert raw HTML to clean, readable plain text."""
    # Strip HTML comments
    text = re.sub(r"<!--.*?-->", "", raw_html, flags=re.DOTALL)
    # Remove <script> and <style> blocks entirely
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>", "", text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Turn block-level tags into newlines so paragraphs are preserved
    text = re.sub(
        r"<(br|p|div|section|article|h[1-6]|li|tr|td|th)[^>]*>",
        "\n", text, flags=re.IGNORECASE,
    )
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities (&amp; → &, &nbsp; → space, etc.)
    text = html_lib.unescape(text)
    # Collapse runs of whitespace; preserve paragraph breaks (≤2 newlines)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── URL discovery ─────────────────────────────────────────────────────────────

def _extract_program_urls(raw_html: str) -> List[str]:
    """Return all unique individual-program URLs found in an index page."""
    found: set[str] = set()
    for pattern in _PROGRAM_URL_PATTERNS:
        found.update(re.findall(pattern, raw_html))
    return list(found)


# ── Async fetching ────────────────────────────────────────────────────────────

async def _fetch(client: httpx.AsyncClient, url: str) -> str:
    """GET *url* and return raw HTML, or '' on any error."""
    try:
        resp = await client.get(url, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        print(f"[WebScraper] Could not fetch {url}: {exc}")
        return ""


# ── Public API ────────────────────────────────────────────────────────────────

async def scrape_program_pages() -> List[str]:
    """
    Discover and fetch all FCFM program pages.

    Returns a list of plain-text documents (one per page) ready to be
    chunked and embedded.  Each document is prefixed with its source URL
    so the model can cite it if needed.
    """
    documents: List[str] = []

    async with httpx.AsyncClient(headers=_HTTP_HEADERS) as client:
        # ── Step 1: fetch index pages & collect individual program URLs ──────
        program_urls: set[str] = set()
        for index_url in PROGRAM_INDEX_URLS:
            raw = await _fetch(client, index_url)
            if not raw:
                continue
            # Keep the index page itself as a document (contains program list)
            text = html_to_text(raw)
            if text:
                documents.append(f"[Fuente: {index_url}]\n{text}")
            program_urls.update(_extract_program_urls(raw))

        # ── Step 2: fetch all individual program pages in parallel ───────────
        urls = list(program_urls)
        raw_pages = await asyncio.gather(*[_fetch(client, u) for u in urls])

        for url, raw in zip(urls, raw_pages):
            if not raw:
                continue
            text = html_to_text(raw)
            if text:
                documents.append(f"[Fuente: {url}]\n{text}")

    print(
        f"[WebScraper] Built {len(documents)} documents "
        f"({len(program_urls)} program pages + {len(PROGRAM_INDEX_URLS)} index pages)"
    )
    return documents
