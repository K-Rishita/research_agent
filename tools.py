"""Tool definitions for the research agent.

Each function decorated with @tool becomes something the LLM can choose to
call. The docstrings are not just documentation -- LangChain sends them to
the model as the tool descriptions, which is how the model decides when to
use each one. Keep them precise.
"""

import difflib
import io
import os
import random
import re
import time

import arxiv
import requests
from ddgs import DDGS  # duckduckgo-search, renamed; no API key needed
from langchain_core.tools import tool
from pypdf import PdfReader

# One shared client, reused across calls. The arxiv package handles its own
# rate limiting internally per client, so a module-level client is both
# faster and politer to arXiv than constructing one per search.
#
# delay_seconds is arXiv's own requested pacing between requests; num_retries
# makes the library retry transient failures before it ever raises to us.
_arxiv_client = arxiv.Client(delay_seconds=3.0, num_retries=3)

# Semantic Scholar's unauthenticated pool is shared across every anonymous
# caller on the internet, which is why it 429s so readily. A free key (request
# one at https://www.semanticscholar.org/product/api) moves you to your own
# much higher quota and is the single most effective fix for rate limiting.
_S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# Status codes worth retrying: rate limiting and transient server errors.
# Anything else (404, 400, ...) is a real answer -- retrying just wastes time.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _get_with_retry(url, *, params=None, headers=None, timeout=20,
                    max_attempts=4):
  """GET with exponential backoff and jitter on transient failures.

  Returns (response, None) on success, or (None, error_message) if every
  attempt failed. Most 429s are short-lived bursts, so backing off and
  retrying resolves them without the caller ever seeing an error.

  The jitter matters: without it, concurrent retries all fire at the same
  instant and re-trigger the same rate limit in lockstep.
  """
  delay = 1.0
  last_error = None

  for attempt in range(1, max_attempts + 1):
    try:
      resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as e:
      last_error = f"network error: {e}"
    else:
      if resp.status_code not in _RETRYABLE_STATUS:
        return resp, None
      last_error = f"HTTP {resp.status_code}"
      # Prefer the server's own instruction over our guess when it sends one.
      retry_after = resp.headers.get("Retry-After")
      if retry_after:
        try:
          delay = max(delay, float(retry_after))
        except ValueError:
          pass

    if attempt < max_attempts:
      time.sleep(delay + random.uniform(0, 0.3 * delay))
      delay *= 2

  return None, last_error


# Within a single research run the agent often revisits the same paper --
# e.g. pulling 'methodology' and then 'results', or re-checking a citation
# count it already looked up. Caching makes the repeat request cost zero
# instead of risking a fresh rate limit. Only successful lookups are cached;
# caching a transient failure would make it permanent for the whole run.
_citation_cache: dict[str, str] = {}
_pdf_text_cache: dict[str, str] = {}


def _format_paper(paper) -> str:
  """Render one arXiv result in the compact form the model reads."""
  return (
      f"Title: {paper.title}\n"
      f"Authors: {', '.join(a.name for a in paper.authors[:3])}"
      f"{' et al.' if len(paper.authors) > 3 else ''}\n"
      f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
      f"arXiv ID: {paper.get_short_id()}\n"
      f"URL: {paper.entry_id}\n"
      f"Abstract: {paper.summary[:500]}...\n"
  )


@tool
def search_arxiv(query: str) -> str:
  """Search arXiv for papers on a given topic. Returns titles, authors,
  publication dates, and abstracts for the top 5 matches. Works for both
  general topics and specific named systems or benchmarks."""
  # arXiv parses a bare query as loose keywords, so a distinctive name gets
  # swamped by common words in it -- searching "LMRL Gym" returns Isaac Gym,
  # SWE-Gym and friends, but never LMRL Gym itself. Running the quoted phrase
  # first surfaces exact matches; the loose query then fills in related work.
  cleaned = query.strip().strip('"')
  attempts = [f'all:"{cleaned}"', cleaned] if cleaned else [cleaned]

  seen, papers, last_error = set(), [], None
  for attempt in attempts:
    if len(papers) >= 5:
      break
    try:
      found = _arxiv_client.results(
          arxiv.Search(query=attempt, max_results=5,
                       sort_by=arxiv.SortCriterion.Relevance))
      for paper in found:
        sid = paper.get_short_id()
        if sid not in seen:
          seen.add(sid)
          papers.append(paper)
        if len(papers) >= 5:
          break
    except Exception as e:
      last_error = e

  if not papers:
    if last_error is not None:
      return f"Error searching arXiv: {last_error}"
    return "No papers found on arXiv for this query."

  return "\n---\n".join(_format_paper(p) for p in papers)


@tool
def get_paper_by_id(arxiv_id: str) -> str:
  """Look up one specific arXiv paper by its ID (e.g. '2311.18232') and
  return its authoritative title, authors, date and abstract. Use this
  whenever you learn an arXiv ID from a web search result, so that you cite
  the real paper metadata rather than trusting a search snippet."""
  clean_id = arxiv_id.strip().split("/")[-1].replace(".pdf", "")
  try:
    found = list(_arxiv_client.results(arxiv.Search(id_list=[clean_id])))
  except Exception as e:
    return f"Error looking up arXiv:{clean_id}: {e}"

  if not found:
    return f"No arXiv paper found with ID {clean_id}."
  return _format_paper(found[0])


@tool
def search_web(query: str) -> str:
  """Search the general web for background context, news, or non-arXiv
  sources related to a topic. Returns top 5 results with titles, snippets,
  and URLs."""
  # backend="auto" already fans out across several search providers, so a
  # single provider being down is handled internally. The retry loop covers
  # the case where the whole call fails transiently.
  results, last_error = [], None
  delay = 1.0
  for attempt in range(3):
    try:
      with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5, backend="auto"))
      break
    except Exception as e:
      last_error = e
      if attempt < 2:
        time.sleep(delay + random.uniform(0, 0.3 * delay))
        delay *= 2

  if last_error is not None and not results:
    return (f"Web search failed after retries ({last_error}). Rely on "
            f"search_arxiv instead of guessing at web sources.")
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


# A numbered heading on its own line -- "3 Model Architecture", "4.1 Training".
# This is how essentially every arXiv paper marks its sections after PDF text
# extraction, and it's far more reliable than guessing from capitalisation.
_NUMBERED_HEADER_RE = re.compile(
    r"^[ \t]*(\d{1,2}(?:\.\d{1,2})*)\.?[ \t]+([A-Z][^\n]{2,60}?)[ \t]*$",
    re.MULTILINE)

# Unnumbered sections that are near-universal and worth catching too.
_BARE_HEADER_RE = re.compile(
    r"^[ \t]*(abstract|references|acknowledgements?|appendix[^\n]{0,30})[ \t]*$",
    re.IGNORECASE | re.MULTILINE)


def _list_headers(full_text: str) -> list[tuple[int, int, str]]:
  """Find every section heading in the paper.

  Returns (body_start, header_start, title) triples in document order, so a
  section's text is everything from its body_start up to the next
  header_start. Headings whose body is too short are dropped -- those are
  table-of-contents entries, not real sections.
  """
  found = []
  for m in _NUMBERED_HEADER_RE.finditer(full_text):
    found.append((m.start(), m.end(), f"{m.group(1)} {m.group(2).strip()}"))
  for m in _BARE_HEADER_RE.finditer(full_text):
    found.append((m.start(), m.end(), m.group(1).strip()))
  found.sort()

  # Drop the table of contents: its entries sit back-to-back with no body.
  kept = []
  for i, (h_start, h_end, title) in enumerate(found):
    next_start = found[i + 1][0] if i + 1 < len(found) else len(full_text)
    if next_start - h_end >= _MIN_SECTION_CHARS:
      kept.append((h_end, h_start, title))
  return kept


def _normalise_header(title: str) -> str:
  """Strip section numbering and punctuation so 'Model Architecture' and
  '3 Model Architecture' compare equal."""
  return re.sub(r"[^a-z ]", " ", re.sub(r"^\d[\d.]*", "", title.lower())).strip()


def _extract_by_header(full_text: str, wanted: str) -> str:
  """Extract the section whose heading matches `wanted`, by exact name or
  close match. Lets the caller ask for a heading it saw listed, e.g.
  'Model Architecture', rather than one of our canonical names."""
  headers = _list_headers(full_text)
  if not headers:
    return ""

  target = _normalise_header(wanted)
  names = [_normalise_header(t) for _, _, t in headers]

  idx = None
  if target in names:
    idx = names.index(target)
  else:
    # Substring either way: "training" should match "Training and Setup".
    for i, name in enumerate(names):
      if target and (target in name or name in target):
        idx = i
        break
  if idx is None:
    close = difflib.get_close_matches(target, names, n=1, cutoff=0.8)
    if close:
      idx = names.index(close[0])
  if idx is None:
    return ""

  body_start = headers[idx][0]
  end = headers[idx + 1][1] if idx + 1 < len(headers) else len(full_text)
  return full_text[body_start:end].strip()


def _as_header(pattern: str) -> str:
  """Anchor a section pattern so it only matches a real section header --
  i.e. at the start of a line, as a whole word. Without this, searching for
  'methodology' matches the table of contents, or a passing mention in the
  introduction, long before the actual section."""
  return rf"^\s*{pattern}\b"


# A table-of-contents entry looks exactly like a real section header -- same
# words, also at the start of a line -- so anchoring alone doesn't exclude it.
# What distinguishes them is what follows: a ToC entry is immediately followed
# by the next ToC entry, leaving almost no text between them, whereas a real
# section has actual content. Anything shorter than this is treated as a ToC
# hit and we keep looking further down the document.
_MIN_SECTION_CHARS = 200


def _extract_section(full_text: str, section: str) -> str:
  """Best-effort extraction of one named section from raw PDF text by
  locating its header and the next header that follows it.

  Considers every candidate header in turn and returns the first one with
  substantive content, which skips table-of-contents entries.
  """
  section = section.lower().strip()
  if section not in _SECTION_PATTERNS:
    # Not one of our canonical names -- the caller is asking for a heading it
    # saw in this specific paper, e.g. "Model Architecture".
    return _extract_by_header(full_text, section)

  headers_in_order = list(_SECTION_PATTERNS.items())
  start_idx = [i for i, (name, _) in enumerate(headers_in_order) if name == section][0]

  start_re = re.compile(_as_header(_SECTION_PATTERNS[section]),
                        re.IGNORECASE | re.MULTILINE)
  later_res = [re.compile(_as_header(pattern), re.IGNORECASE | re.MULTILINE)
               for _, pattern in headers_in_order[start_idx + 1:]]

  best = ""
  for start_match in start_re.finditer(full_text):
    start_pos = start_match.end()

    # Find the earliest match among all *later* section headers, to know
    # where this section ends. Searching with a pos (rather than slicing)
    # keeps '^' anchored to genuine line starts.
    end_pos = len(full_text)
    for pattern in later_res:
      m = pattern.search(full_text, start_pos)
      if m and m.start() < end_pos:
        end_pos = m.start()

    body = full_text[start_pos:end_pos].strip()
    if len(body) >= _MIN_SECTION_CHARS:
      return body
    # Keep the longest near-miss in case no candidate clears the bar (e.g. a
    # genuinely short section, or a paper with unusual formatting).
    if len(body) > len(best):
      best = body

  if best:
    return best

  # The canonical name isn't in this paper. Papers name the same thing
  # differently -- "Model Architecture" instead of "Methodology" -- so fall
  # back to matching against the headings this paper actually uses.
  return _extract_by_header(full_text, section)


@tool
def get_paper_section(arxiv_id_or_url: str, section: str = "") -> str:
  """Download a paper's PDF (given an arXiv ID like '2405.15628' or a full
  arXiv URL) and extract one specific section of it.

  `section` accepts either a generic name -- 'abstract', 'introduction',
  'related work', 'methodology', 'results', 'conclusion', 'references' -- or
  the exact heading used by that paper, e.g. 'Model Architecture'.

  Papers name the same content differently, so if a generic name isn't found
  this returns the list of headings that paper actually uses. When that
  happens, pick the heading that best matches what you wanted and call again
  with it: a paper with no 'Methodology' section may describe its method
  under 'Model Architecture' or 'Approach'. If section is left blank, returns
  the first ~2000 characters (usually abstract and intro)."""
  arxiv_id = arxiv_id_or_url.strip().split("/")[-1].replace(".pdf", "")

  # Papers are often queried twice in one run (methodology, then results).
  # Reusing the parsed text avoids re-downloading a multi-megabyte PDF.
  full_text = _pdf_text_cache.get(arxiv_id)

  if full_text is None:
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    resp, error = _get_with_retry(pdf_url, timeout=30)
    if resp is None:
      return (f"Could not download the PDF for {arxiv_id} after several "
              f"attempts ({error}). Use the abstract from search_arxiv "
              f"instead of guessing at the section contents.")
    if resp.status_code != 200:
      return f"Error downloading PDF for {arxiv_id}: HTTP {resp.status_code}"

    try:
      reader = PdfReader(io.BytesIO(resp.content))
      full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
      return f"Error parsing PDF for {arxiv_id}: {e}"

    if not full_text.strip():
      return f"Could not extract readable text from {arxiv_id} (possibly a scanned/image PDF)."

    _pdf_text_cache[arxiv_id] = full_text

  if section:
    extracted = _extract_section(full_text, section)
    if extracted:
      # Cap length to keep token usage reasonable even for long sections.
      return extracted[:4000]
    # Show the model what this paper actually calls its sections, so it can
    # map what it wanted onto the paper's own naming and ask again.
    headings = [title for _, _, title in _list_headers(full_text)]
    if headings:
      listed = "\n".join(f"  - {h}" for h in headings)
      return (f"This paper has no section named '{section}'. The headings it "
              f"actually uses are:\n\n{listed}\n\n"
              f"Call get_paper_section again with whichever of these best "
              f"matches what you wanted (for example, a paper's methodology "
              f"is often under 'Model Architecture', 'Approach' or 'Method').")

    return (f"Could not locate a '{section}' section in {arxiv_id}, and could "
            f"not detect its headings. Here are the first 2000 characters "
            f"instead:\n\n{full_text[:2000]}")

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
  if arxiv_id in _citation_cache:
    return _citation_cache[arxiv_id]

  url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
  params = {"fields": "title,citationCount,influentialCitationCount,year"}
  headers = {"x-api-key": _S2_API_KEY} if _S2_API_KEY else None

  resp, error = _get_with_retry(url, params=params, headers=headers, timeout=15)

  if resp is None:
    # Deliberately not cached -- a later attempt in the same run may succeed.
    hint = ("" if _S2_API_KEY else
            " Setting SEMANTIC_SCHOLAR_API_KEY in .env raises this limit a lot.")
    return (f"Could not reach Semantic Scholar for {arxiv_id} after several "
            f"retries ({error}). Treat the citation count as unknown rather "
            f"than guessing it.{hint}")

  if resp.status_code == 404:
    result = (f"No Semantic Scholar record found for arXiv:{arxiv_id} "
              f"(may be too new or not indexed yet).")
    _citation_cache[arxiv_id] = result  # A definitive answer; safe to cache.
    return result

  if resp.status_code != 200:
    return f"Error fetching citation data for {arxiv_id}: HTTP {resp.status_code}"

  try:
    data = resp.json()
  except ValueError:
    return f"Semantic Scholar returned a non-JSON response for {arxiv_id}."

  result = (
    f"Title: {data.get('title')}\n"
    f"Year: {data.get('year')}\n"
    f"Total citations: {data.get('citationCount', 0)}\n"
    f"Influential citations: {data.get('influentialCitationCount', 0)}"
  )
  _citation_cache[arxiv_id] = result
  return result


@tool
def save_summary_to_file(filename: str, content: str) -> str:
  """Save the final research summary (or any text content) to a local
  markdown file. filename should be a short name without a path, e.g.
  'llm_agents_summary' -- '.md' is added automatically if missing. Use
  this at the end of a research task if the user wants a saved record
  rather than just a printed answer."""
  # Strip any path components and a trailing extension first, then whitelist
  # characters. Doing it in this order matters: the whitelist removes '.', so
  # checking for a '.md' suffix afterwards would never match.
  stem = os.path.basename(filename.strip())
  if stem.lower().endswith(".md"):
    stem = stem[:-3]
  safe_name = "".join(c for c in stem if c.isalnum() or c in ("_", "-")).strip()
  if not safe_name:
    safe_name = "research_summary"
  safe_name += ".md"

  try:
    with open(safe_name, "w", encoding="utf-8") as f:
      f.write(content)
  except Exception as e:
    return f"Error saving file: {e}"

  return f"Saved summary to {safe_name} ({len(content)} characters)."


tools = [
  search_arxiv,
  get_paper_by_id,
  search_web,
  get_paper_section,
  compare_papers,
  get_citation_count,
  save_summary_to_file,
]
