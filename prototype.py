import json
import streamlit as st
from google import genai
import os


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Literature Search", layout="wide")

# -----------------------------
# Config
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "example-bib.json")
MODEL_NAME = "gemini-2.5-flash"
client = genai.Client()  # picks up GEMINI_API_KEY from env


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_papers(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["references"]


def norm_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [str(v)]


def norm_authors(p):
    return [str(x) for x in norm_list(p.get("authors")) if str(x).strip()]


def norm_keywords(p):
    return [str(x) for x in norm_list(p.get("keywords")) if str(x).strip()]


def paper_url(p):
    url = p.get("url") or p.get("link") or p.get("pdf")
    if url:
        return str(url).strip()
    doi = (p.get("doi") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    return ""


def searchable_text(p):
    authors = " ".join(norm_authors(p))
    keywords = " ".join(norm_keywords(p))
    return " ".join(
        [
            str(p.get("title", "")),
            str(p.get("abstract", "")),
            authors,
            str(p.get("journal", "")),
            str(p.get("venue", "")),
            str(p.get("doi", "")),
            keywords,
        ]
    ).lower()


def brief_for_ai(p, abstract_chars=420):
    abstract = (p.get("abstract") or "").replace("\n", " ").strip()
    if len(abstract) > abstract_chars:
        abstract = abstract[:abstract_chars] + "…"
    return {
        "id": p.get("id"),
        "title": p.get("title", ""),
        "year": p.get("year"),
        "authors": norm_authors(p)[:10],
        "journal": p.get("journal", "") or p.get("venue", ""),
        "keywords": norm_keywords(p)[:15],
        "doi": p.get("doi", ""),
        "url": paper_url(p),
        "abstract": abstract,
    }


def parse_json_lenient(text: str):
    if not text:
        return None
    t = text.strip()

    if t.startswith("```"):
        t = t.strip("`").strip()
        lines = t.splitlines()
        if lines and lines[0].strip().lower() in ("json", "javascript"):
            t = "\n".join(lines[1:]).strip()

    try:
        return json.loads(t)
    except Exception:
        pass

    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start : end + 1])
        except Exception:
            return None
    return None


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
<style>
div.stButton > button{
    border-radius: 999px;
    padding: 0.18rem 0.55rem;
}
.meta-chip {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    margin: 0.1rem 0.25rem 0.1rem 0;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.12);
    font-size: 0.85rem;
    opacity: 0.95;
}
.meta-kv {
    margin: 0.15rem 0 0.35rem 0;
    font-size: 0.95rem;
}
.meta-k {
    opacity: 0.7;
    display: inline-block;
    min-width: 90px;
}
.meta-v {
    opacity: 0.98;
}
</style>
""",
    unsafe_allow_html=True,
)


def toggle_save(pid, saved_ids):
    if pid is None:
        return
    if pid in saved_ids:
        saved_ids.discard(pid)
    else:
        saved_ids.add(pid)


def render_metadata_pretty_no_columns(p, key_prefix=""):
    """
    IMPORTANT: No st.columns() here (avoids Streamlit column nesting error).
    """
    pid = p.get("id")
    year = p.get("year")
    venue = p.get("journal", "") or p.get("venue", "")
    authors = norm_authors(p)
    doi = (p.get("doi") or "").strip()
    url = paper_url(p)
    keywords = norm_keywords(p)

    def kv(k, v):
        vv = v if (v is not None and str(v).strip()) else "—"
        st.markdown(
            f'<div class="meta-kv"><span class="meta-k">{k}</span>'
            f'<span class="meta-v">{vv}</span></div>',
            unsafe_allow_html=True,
        )

    kv("Year", year)
    kv("Venue", venue)
    kv("DOI", doi)
    kv("URL", url)
    kv("ID", pid if pid is not None else "")

    st.markdown('<div class="meta-kv"><span class="meta-k">Authors</span></div>', unsafe_allow_html=True)
    if authors:
        if len(authors) <= 12:
            st.write(", ".join(authors))
        else:
            st.write(", ".join(authors[:12]) + f" … (+{len(authors)-12} more)")
    else:
        st.write("—")

    st.markdown('<div class="meta-kv"><span class="meta-k">Keywords</span></div>', unsafe_allow_html=True)
    if keywords:
        chips = " ".join([f'<span class="meta-chip">{k}</span>' for k in keywords[:30]])
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.write("—")

    show_raw = st.toggle("Show raw JSON", value=False, key=f"{key_prefix}raw_{pid}")
    if show_raw:
        st.json(p)


def render_paper_card(p, saved_ids, key_prefix=""):
    """
    No nested expanders. Tabs are OK.
    """
    pid = p.get("id")
    title = p.get("title", "(no title)")
    year = p.get("year", "")
    venue = p.get("journal", "") or p.get("venue", "")
    authors = ", ".join(norm_authors(p))
    url = paper_url(p)
    doi = (p.get("doi") or "").strip()
    kw = norm_keywords(p)

    saved = pid in saved_ids
    star = "★" if saved else "☆"
    #save_text = "Saved" if saved else "Save"

    cols = st.columns([0.12, 0.88], vertical_alignment="top")
    with cols[0]:
        btn_key = f"{key_prefix}save_{pid}_{hash(title)}"
        if st.button(f"{star}", key=btn_key, use_container_width=True):
            toggle_save(pid, saved_ids)
            st.rerun()

    with cols[1]:
        st.markdown(f"**{title}**")
        if year or venue:
            st.caption(f"{year} • {venue}")
        if authors:
            st.write(authors)

        if url:
            st.write(url)
        if doi:
            st.write(f"**DOI:** {doi}")
        if kw:
            preview = ", ".join(kw[:12])
            more = f" … (+{len(kw)-12})" if len(kw) > 12 else ""
            st.caption("Tags: " + preview + more)

        tab_meta, tab_abs = st.tabs(["Metadata", "Abstract"])
        with tab_meta:
            render_metadata_pretty_no_columns(p, key_prefix=key_prefix)
        with tab_abs:
            if p.get("abstract"):
                st.write(p["abstract"])
            else:
                st.caption("No abstract.")

    st.divider()


# -----------------------------
# Load data & session state
# -----------------------------
papers = load_papers()
papers_by_id = {p.get("id"): p for p in papers if p.get("id") is not None}

if "saved_ids" not in st.session_state:
    st.session_state.saved_ids = set()
if "query" not in st.session_state:
    st.session_state.query = ""
if "ai_selected_ids" not in st.session_state:
    st.session_state.ai_selected_ids = []
if "ai_note" not in st.session_state:
    st.session_state.ai_note = ""
if "kw_ran" not in st.session_state:
    st.session_state.kw_ran = False
if "kw_results" not in st.session_state:
    st.session_state.kw_results = []


# -----------------------------
# Top bar
# -----------------------------
st.markdown("## Literature Search")

top = st.container()
with top:
    c1, c2, c3, c4 = st.columns([0.55, 0.15, 0.15, 0.15], vertical_alignment="bottom")
    with c1:
        st.text_input(
            "Search query",
            key="query",
            placeholder="Type keywords or a short phrase (used for both panels by default)",
        )
    with c2:
        max_kw = st.slider("Keyword results", 5, 100, 20)
    with c3:
        top_k_ai = st.slider("AI results", 5, 100, 20)
    with c4:
        use_same_query = st.checkbox("AI uses same query", value=True)

    with st.expander("Keyword filters", expanded=False):
        years = [p.get("year") for p in papers if isinstance(p.get("year"), int)]
        if years:
            y_min, y_max = min(years), max(years)
            year_range = st.slider("Year range", y_min, y_max, (y_min, y_max))
        else:
            year_range = None
        only_with_doi = st.checkbox("Only with DOI", value=False)


# -----------------------------
# Two-pane results
# -----------------------------
left, right = st.columns(2, gap="large")

# Left: keyword search
with left:
    st.markdown("### Keyword matching")

    b1, b2 = st.columns([0.5, 0.5])
    with b1:
        run_kw = st.button("Run keyword search", type="primary", use_container_width=True)
    with b2:
        clear_kw = st.button("Clear keyword results", use_container_width=True)

    if clear_kw:
        st.session_state.kw_ran = False
        st.session_state.kw_results = []
        st.rerun()

    if run_kw:
        q = (st.session_state.query or "").strip()
        if not q:
            st.warning("Please enter a query first.")
        else:
            ql = q.lower()
            results = []
            for p in papers:
                if year_range and isinstance(p.get("year"), int):
                    if not (year_range[0] <= p["year"] <= year_range[1]):
                        continue
                if only_with_doi and not p.get("doi"):
                    continue
                if ql in searchable_text(p):
                    results.append(p)

            st.session_state.kw_results = results
            st.session_state.kw_ran = True
            st.rerun()

    if not st.session_state.kw_ran:
        st.caption("Click “Run keyword search” to view matches.")
    else:
        results = st.session_state.kw_results
        st.caption(f"Found {len(results)} matches. Showing up to {max_kw}.")
        with st.container(height=650):
            for i, p in enumerate(results[:max_kw]):
                render_paper_card(p, st.session_state.saved_ids, key_prefix=f"kw_{i}_")


# Right: AI search
with right:
    st.markdown("### AI search (Gemini selects papers)")
    q = (st.session_state.query or "").strip()

    if use_same_query:
        ai_intent = q
        #st.caption("AI intent is the same as the main query.")
    else:
        ai_intent = st.text_input(
            "AI intent (optional override)",
            value=q,
            placeholder="Describe what you want (e.g., causal probing for interpretability)",
            key="ai_intent_override",
        )

    b1, b2 = st.columns([0.5, 0.5])
    with b1:
        run_ai = st.button("Run AI search", type="primary", use_container_width=True)
    with b2:
        clear_ai = st.button("Clear AI results", use_container_width=True)

    if clear_ai:
        st.session_state.ai_selected_ids = []
        st.session_state.ai_note = ""
        st.rerun()

    if run_ai:
        if not ai_intent.strip():
            st.warning("Please enter a query first.")
        else:
            library = [brief_for_ai(p) for p in papers]
            prompt = f"""
You are an AI retrieval tool over a local paper library.

Return ONLY valid JSON with exactly this schema:
{{
  "paper_ids": ["... up to {top_k_ai} ids from the library ..."],
  "note": "1-2 sentence rationale"
}}

User intent: {ai_intent}

Library (JSON list):
{json.dumps(library, ensure_ascii=False)}
"""
            with st.spinner("Selecting papers…"):
                resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
                parsed = parse_json_lenient(resp.text or "") or {}

            ids = parsed.get("paper_ids", [])
            if not isinstance(ids, list):
                ids = []

            ids = [pid for pid in ids if pid in papers_by_id]
            st.session_state.ai_selected_ids = ids
            st.session_state.ai_note = str(parsed.get("note", "")).strip()
            st.rerun()

    if st.session_state.ai_note:
        st.caption(st.session_state.ai_note)

    total_papers = len(papers)
    found_papers = len(st.session_state.ai_selected_ids)

    if not st.session_state.ai_selected_ids:
        st.caption(f"Found 0 of {total_papers} papers. Click “Run AI search” to select papers.")
    else:
        st.caption(f"Found {found_papers} of {total_papers} papers.")
        with st.container(height=650):
            for i, pid in enumerate(st.session_state.ai_selected_ids):
                render_paper_card(papers_by_id[pid], st.session_state.saved_ids, key_prefix=f"ai_{i}_")


# -----------------------------
# Saved section
# -----------------------------
st.markdown("---")
st.markdown(f"### Saved papers ({len(st.session_state.saved_ids)})")

saved = [papers_by_id[pid] for pid in st.session_state.saved_ids if pid in papers_by_id]
saved.sort(key=lambda p: (p.get("year") is None, p.get("year", 0)), reverse=True)

if not saved:
    st.caption("No saved papers yet. Click ☆ Save on any paper card to save it.")
else:
    a1, a2, a3 = st.columns([0.34, 0.33, 0.33], vertical_alignment="center")
    with a1:
        export_obj = {"references": saved}
        st.download_button(
            "Download saved papers (JSON)",
            data=json.dumps(export_obj, ensure_ascii=False, indent=2),
            file_name="saved-papers.json",
            mime="application/json",
            key="download_saved",
            use_container_width=True,
        )
    with a2:
        if st.button("Clear all saved", use_container_width=True):
            st.session_state.saved_ids = set()
            st.rerun()
    with a3:
        st.caption("Saved items persist during this session.")

    for i, p in enumerate(saved):
        render_paper_card(p, st.session_state.saved_ids, key_prefix=f"saved_{i}_")

