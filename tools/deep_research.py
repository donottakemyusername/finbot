"""tools/deep_research.py
=========================
Deep research mode: pulls 10-K and 10-Q filings from the SEC EDGAR API,
extracts key sections (MD&A, Risk Factors, Financial Highlights), and
returns structured summaries for Claude to reason over.

EDGAR is free — set EDGAR_USER_AGENT in your .env for polite crawling.

Two approaches
--------------
1. EDGAR full-text search API  (primary)  — fast, no parsing needed
2. Direct CIK lookup + filing index       — fallback for older filings
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "StockAnalyst contact@example.com")
EDGAR_BASE       = "https://data.sec.gov"
EFTS_BASE        = "https://efts.sec.gov"   # full-text search

HEADERS = {
    "User-Agent": EDGAR_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

# Polite crawl — SEC asks for max 10 req/s; we stay well under
_SLEEP = 0.15


def _get(url: str, params: dict | None = None) -> Any:
    time.sleep(_SLEEP)
    resp = requests.get(url, headers=HEADERS, params=params or {}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _get_text(url: str) -> str:
    time.sleep(_SLEEP)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text


# ─────────────────────────────────────────────────────────────────────────────
# CIK resolution
# ─────────────────────────────────────────────────────────────────────────────

def get_cik(ticker: str) -> str | None:
    """Resolve a ticker to a zero-padded 10-digit CIK."""
    try:
        data = _get(f"{EDGAR_BASE}/submissions/", params={"action": "getcompany",
                                                           "company": ticker,
                                                           "output": "json"})
    except Exception:
        pass

    # Direct ticker → CIK lookup via EDGAR company search
    try:
        resp = requests.get(
            "https://www.sec.gov/cgi-bin/browse-edgar",
            headers=HEADERS,
            params={"action": "getcompany", "company": ticker,
                    "type": "", "dateb": "", "owner": "include",
                    "count": "1", "search_text": "", "output": "atom"},
            timeout=15,
        )
        # Extract CIK from response
        match = re.search(r"CIK=(\d+)", resp.text)
        if match:
            return match.group(1).zfill(10)
    except Exception:
        pass

    # Try the company_tickers.json endpoint (most reliable)
    try:
        tickers_json = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=HEADERS, timeout=15,
        ).json()
        for entry in tickers_json.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass

    return None


def get_recent_filings(
    ticker: str,
    form_type: str = "10-K",
    limit: int = 3,
) -> list[dict]:
    """
    Return recent filings of a given type for a ticker.

    Each item contains:
        accession_number, filing_date, form_type, primary_document_url
    """
    cik = get_cik(ticker)
    if not cik:
        return []

    cik_stripped = cik.lstrip("0")
    data = _get(f"{EDGAR_BASE}/submissions/CIK{cik}.json")

    recent = data.get("filings", {}).get("recent", {})
    forms   = recent.get("form", [])
    dates   = recent.get("filingDate", [])
    accnums = recent.get("accessionNumber", [])
    docs    = recent.get("primaryDocument", [])

    results = []
    for form, date, acc, doc in zip(forms, dates, accnums, docs):
        if form == form_type:
            acc_clean = acc.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{acc_clean}/{doc}"
            results.append({
                "form_type":        form,
                "filing_date":      date,
                "accession_number": acc,
                "document_url":     url,
            })
            if len(results) >= limit:
                break

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

# Regex patterns for common 10-K sections (HTML/text filings)
_SECTION_PATTERNS = {
    "mda": re.compile(
        r"(?:management.{0,30}discussion|item\s*7\.?\s*management)",
        re.IGNORECASE,
    ),
    "risk_factors": re.compile(
        r"item\s*1a\.?\s*risk\s*factors",
        re.IGNORECASE,
    ),
    "business": re.compile(
        r"item\s*1\.?\s*business",
        re.IGNORECASE,
    ),
    "financial_statements": re.compile(
        r"item\s*8\.?\s*financial\s*statements",
        re.IGNORECASE,
    ),
}

_MAX_SECTION_CHARS = 8_000   # truncate long sections for LLM context


def _strip_html(text: str) -> str:
    """Very light HTML → plain text."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;",  "&", text)
    text = re.sub(r"&lt;",   "<", text)
    text = re.sub(r"&gt;",   ">", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _extract_section(full_text: str, section: str) -> str:
    """Extract a named section from a filing (best-effort)."""
    pat = _SECTION_PATTERNS.get(section)
    if not pat:
        return ""

    match = pat.search(full_text)
    if not match:
        return ""

    start = match.start()
    # Find the next major section header (Item X.)
    next_item = re.search(r"item\s+\d+[a-z]?\.?\s+[A-Z]", full_text[start + 100:], re.IGNORECASE)
    end = start + 100 + next_item.start() if next_item else start + 20_000

    snippet = full_text[start:end]
    snippet = _strip_html(snippet)
    return snippet[:_MAX_SECTION_CHARS]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_deep_research(
    ticker: str,
    form_type: str = "10-K",
    sections: list[str] | None = None,
    limit: int = 1,
) -> dict:
    """
    Pull the latest 10-K (or 10-Q) and return extracted sections.

    Parameters
    ----------
    ticker    : stock ticker
    form_type : "10-K" or "10-Q"
    sections  : list of keys from _SECTION_PATTERNS; defaults to all four
    limit     : number of filings to return (most recent first)

    Returns
    -------
    {
        "ticker": "AAPL",
        "filings": [
            {
                "form_type": "10-K",
                "filing_date": "2024-11-01",
                "sections": {
                    "business":     "...",
                    "risk_factors": "...",
                    "mda":          "...",
                    "financial_statements": "...",
                }
            }
        ]
    }
    """
    sections_to_pull = sections or list(_SECTION_PATTERNS.keys())
    filings_meta = get_recent_filings(ticker, form_type=form_type, limit=limit)

    if not filings_meta:
        return {"ticker": ticker, "error": f"No {form_type} filings found for {ticker}"}

    results = []
    for meta in filings_meta:
        try:
            raw = _get_text(meta["document_url"])
        except Exception as exc:
            results.append({**meta, "error": str(exc)})
            continue

        extracted: dict[str, str] = {}
        for sec in sections_to_pull:
            content = _extract_section(raw, sec)
            if content:
                extracted[sec] = content

        results.append({
            "form_type":   meta["form_type"],
            "filing_date": meta["filing_date"],
            "source_url":  meta["document_url"],
            "sections":    extracted,
        })

    return {
        "ticker":    ticker,
        "form_type": form_type,
        "filings":   results,
    }


def get_filing_summary(ticker: str, form_type: str = "10-K") -> dict:
    """
    Return a lightweight summary: list of recent filings with dates and URLs.
    No content fetching — useful for showing what's available.
    """
    filings = get_recent_filings(ticker, form_type=form_type, limit=5)
    return {
        "ticker":    ticker,
        "form_type": form_type,
        "available": [
            {"date": f["filing_date"], "url": f["document_url"]}
            for f in filings
        ],
    }