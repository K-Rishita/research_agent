"""
Research Assistant Agent
------------------------
A LangChain tool-calling agent that takes a research topic, searches arXiv
for relevant papers, optionally searches the web for extra context, can pull
a specific section out of a paper's actual PDF, check how influential a
paper is via citation counts, structure a comparison between two papers, and
save the final summary to a file.

The model decides for itself which tools to call and how many times --
e.g. it might search arXiv, look at a few abstracts, check citation counts
to see which paper matters most, then do a web search to clarify something,
before writing the final summary. That decision-making loop across six
different tools is what makes this an "agent" rather than a single fixed
API call.

Setup:
    pip install langchain langchain-openai requests arxiv ddgs pypdf

Run:
    python research_agent.py
"""

import os
import re
import io
import requests
import arxiv
from ddgs import DDGS  # duckduckgo-search package, no API key needed
from pypdf import PdfReader
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# 1. Configure the model to run through OpenRouter (OpenAI-compatible API)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "PASTE_YOUR_KEY_HERE")

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",          # any OpenRouter model id works here
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# ---------------------------------------------------------------------------
# 2. Define tools -- these are real functions the agent can choose to call
# ---------------------------------------------------------------------------

@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for papers on a given topic. Returns titles, authors,
    publication dates, and abstracts for the top 5 matches."""
    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for paper in search.results():
        results.append(
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(a.name for a in paper.authors[:3])}"
            f"{' et al.' if len(paper.authors) > 3 else ''}\n"
            f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
            f"arXiv ID: {paper.get_short_id()}\n"
            f"URL: {paper.entry_id}\n"
            f"Abstract: {paper.summary[:500]}...\n"
        )
    if not results:
        return "No papers found on arXiv for this query."
    return "\n---\n".join(results)


@tool
def search_web(query: str) -> str:
    """Search the general web for background context, news, or non-arXiv
    sources related to a topic. Returns top 5 results with titles, snippets,
    and URLs."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No web results found for this query."
    lines = []
    for r in results:
        lines.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}")
    return "\n---\n".join(lines)



# Common section header patterns found in academic papers. Order matters --
# used to find where a given section starts and where the *next* section
# begins (so we know where to stop extracting).
_SECTION_PATTERNS = {
    "abstract": r"abstract",
    "introduction": r"(?:1\.?\s*)?introduction",
    "related work": r"(?:\d\.?\s*)?related work",
    "methodology": r"(?:\d\.?\s*)?(?:methodology|method|approach|technical approach)",
    "results": r"(?:\d\.?\s*)?(?:results|experiments|evaluation)",
    "conclusion": r"(?:\d\.?\s*)?(?:conclusion|conclusions|discussion)",
    "references": r"references",
}


def _extract_section(full_text: str, section: str) -> str:
    """Best-effort extraction of one named section from raw PDF text by
    locating its header and the next header that follows it."""
    section = section.lower().strip()
    if section not in _SECTION_PATTERNS:
        return ""

    headers_in_order = list(_SECTION_PATTERNS.items())
    start_idx = [i for i, (name, _) in enumerate(headers_in_order) if name == section][0]

    start_pattern = re.compile(_SECTION_PATTERNS[section], re.IGNORECASE)
    start_match = start_pattern.search(full_text)
    if not start_match:
        return ""

    start_pos = start_match.end()

    # Find the earliest match among all *later* section headers, to know
    # where this section ends.
    end_pos = len(full_text)
    for name, pattern in headers_in_order[start_idx + 1:]:
        m = re.search(pattern, full_text[start_pos:], re.IGNORECASE)
        if m:
            candidate_end = start_pos + m.start()
            if candidate_end < end_pos:
                end_pos = candidate_end

    return full_text[start_pos:end_pos].strip()


@tool
def get_paper_section(arxiv_id_or_url: str, section: str = "") -> str:
    """Download a paper's PDF (given an arXiv ID like '2405.15628' or a full
    arXiv URL) and extract one specific section of it. Valid section names:
    'abstract', 'introduction', 'related work', 'methodology', 'results',
    'conclusion', 'references'. If section is left blank or not found, returns
    the first ~2000 characters of the paper instead (usually abstract/intro)."""
    arxiv_id = arxiv_id_or_url.strip().split("/")[-1].replace(".pdf", "")
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    try:
        resp = requests.get(pdf_url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        return f"Error downloading PDF for {arxiv_id}: {e}"

    try:
        reader = PdfReader(io.BytesIO(resp.content))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"Error parsing PDF for {arxiv_id}: {e}"

    if not full_text.strip():
        return f"Could not extract readable text from {arxiv_id} (possibly a scanned/image PDF)."

    if section:
        extracted = _extract_section(full_text, section)
        if extracted:
            # Cap length to keep token usage reasonable even for long sections.
            return extracted[:4000]
        return (f"Could not locate a '{section}' section in {arxiv_id}. "
                f"Here are the first 2000 characters instead:\n\n{full_text[:2000]}")

    return full_text[:2000]


@tool
def compare_papers(paper_a_summary: str, paper_b_summary: str) -> str:
    """Prepare a structured comparison prompt for two papers, given a short
    summary or abstract of each. Returns a formatted prompt highlighting what
    to compare (problem, approach, findings) -- the calling agent should use
    this structure to reason through the actual comparison in its response."""
    return (
        "Compare the following two papers along these dimensions: "
        "(1) the problem each addresses, (2) their technical approach, "
        "(3) their key findings or results, and (4) how they relate to or "
        "differ from each other.\n\n"
        f"Paper A:\n{paper_a_summary}\n\n"
        f"Paper B:\n{paper_b_summary}\n\n"
        "Now reason through points (1)-(4) explicitly before giving a final "
        "verdict on how these two papers relate."
    )


@tool
def get_citation_count(arxiv_id: str) -> str:
    """Look up citation metrics for a paper via the Semantic Scholar API,
    given its arXiv ID (e.g. '2405.15628'). Returns total citation count and
    'influential citation count' (citations where this paper had a
    substantial impact on the citing work, per Semantic Scholar's own
    classification) -- use this to judge whether a paper is foundational and
    widely built upon, versus a new, lightly-cited preprint."""
    arxiv_id = arxiv_id.strip().split("/")[-1].replace(".pdf", "")
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    params = {"fields": "title,citationCount,influentialCitationCount,year"}

    try:
        resp = requests.get(url, params=params, timeout=10)
    except Exception as e:
        return f"Error contacting Semantic Scholar for {arxiv_id}: {e}"

    if resp.status_code == 404:
        return f"No Semantic Scholar record found for arXiv:{arxiv_id} (may be too new or not indexed yet)."
    if resp.status_code != 200:
        return f"Error fetching citation data for {arxiv_id}: HTTP {resp.status_code}"

    data = resp.json()
    return (
        f"Title: {data.get('title')}\n"
        f"Year: {data.get('year')}\n"
        f"Total citations: {data.get('citationCount', 0)}\n"
        f"Influential citations: {data.get('influentialCitationCount', 0)}"
    )


@tool
def save_summary_to_file(filename: str, content: str) -> str:
    """Save the final research summary (or any text content) to a local
    markdown file. filename should be a short name without a path, e.g.
    'llm_agents_summary' -- '.md' is added automatically if missing. Use
    this at the end of a research task if the user wants a saved record
    rather than just a printed answer."""
    safe_name = "".join(c for c in filename if c.isalnum() or c in ("_", "-")).strip()
    if not safe_name:
        safe_name = "research_summary"
    if not safe_name.endswith(".md"):
        safe_name += ".md"

    try:
        with open(safe_name, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return f"Error saving file: {e}"

    return f"Saved summary to {safe_name} ({len(content)} characters)."


tools = [
    search_arxiv,
    search_web,
    get_paper_section,
    compare_papers,
    get_citation_count,
    save_summary_to_file,
]

# ---------------------------------------------------------------------------
# 3. Build the agent
# ---------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant. Given a topic, use the available tools "
     "to find relevant papers and background context. Prefer arXiv for "
     "academic depth, and use web search for context, recency, or when arXiv "
     "results are thin. If the user wants to go deeper on one specific paper "
     "(e.g. its methodology or results), use get_paper_section to pull that "
     "section directly from the PDF rather than relying on the abstract alone. "
     "Use get_citation_count to check how influential or foundational a paper "
     "is before treating it as authoritative, especially when multiple papers "
     "could be cited and you need to judge which ones matter most. "
     "If the user wants two papers compared, use compare_papers to structure "
     "the comparison, then write out the actual comparison yourself following "
     "that structure. If the user asks to save the summary, use "
     "save_summary_to_file at the end. Write a clear summary of the topic's "
     "current state, and cite every paper you reference by title and arXiv "
     "ID or URL. Do not invent papers, sections, citation counts, or "
     "citations -- only use what the tools actually returned."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------------------------------------------------------------------
# 4. Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    topic = input("Enter a research topic to investigate: ").strip()
    result = agent_executor.invoke({
        "input": f"Research the topic '{topic}'. Find relevant recent papers "
                  f"on arXiv, use web search if useful for extra context, "
                  f"and write a short summary with citations."
    })
    print("\n=== RESEARCH SUMMARY ===")
    print(result["output"])
