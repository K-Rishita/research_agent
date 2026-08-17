# Research Assistant Agent

A tool-calling AI agent built with LangChain that researches any topic —
it searches arXiv for academic papers, optionally searches the web for
broader context, can pull a specific section (methodology, results, etc.)
out of one paper's actual PDF, checks citation counts to judge how
influential a paper actually is, can structure a comparison between two
papers, and can save the final summary to a file. The model decides for
itself which tool(s) to use and how many times, before writing the final
answer.

## Setup

1. Create a virtualenv and install dependencies:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```

2. Put your OpenRouter API key in a `.env` file next to the script:
   ```bash
   OPENROUTER_API_KEY=sk-or-your-key-here
   ```
   `.env` is gitignored, so the key never gets committed. Optionally add a
   `MODEL=` line to swap models — any tool-calling model on OpenRouter works,
   and `.env.example` lists per-run costs for the common options:
   ```bash
   MODEL=qwen/qwen3.7-flash
   ```

3. Run it (with the virtualenv active):
   ```bash
   python research_agent.py
   ```

It will prompt for a topic. Some examples that exercise different tools:

| Topic to enter | Tools it exercises |
| --- | --- |
| `long-context LLM evaluation` | arXiv + web search |
| `the methodology of arXiv 2311.18232` | PDF section extraction |
| `most cited papers on transformer attention` | citation lookup |
| `compare LMRL Gym and SWE-Gym` | comparison scaffold |
| `LMRL Gym, and save the summary to a file` | file output |

## Testing

Most of the logic is testable without an API key and without network access:

```bash
.venv/bin/python -m pytest test_tools.py -v
```

That covers section extraction (including the table-of-contents trap),
filename sanitising, retry/backoff behaviour, caching, and the rule that
tools return errors as strings rather than raising.

To also hit the real arXiv, Semantic Scholar, and PDF endpoints — slower and
occasionally flaky, since it depends on those services being up:

```bash
.venv/bin/python -m pytest test_tools.py -m network -v
```

## What makes this an "agent" and not just an API call

- The LLM is given seven tools and decides for itself which to call, how many
  times, and in what order, based on the topic — this is LangChain 1.x's
  agent pattern (`langchain.agents.create_agent`, which compiles to a
  LangGraph state machine running the reason → act → observe loop).
- `search_arxiv` hits arXiv's free public API directly (no key needed) and
  returns real titles, authors, dates, and abstracts. It runs the query as a
  quoted phrase first, then as loose keywords: arXiv's default parsing splits
  a distinctive name into common words, so searching `LMRL Gym` without this
  returns Isaac Gym, SWE-Gym and friends but never LMRL Gym itself.
- `get_paper_by_id` fetches authoritative metadata for one arXiv ID, so a
  paper found via web search gets cited from real arXiv data rather than from
  a search snippet.
- `search_web` uses DuckDuckGo's search (via the `ddgs` package, also no key
  needed) for context arXiv alone might miss.
- `get_paper_section` downloads a paper's actual PDF and extracts one named
  section, rather than handing the whole paper to the model — this keeps
  token usage sane and lets the agent "zoom in" on the part that matters.
  Section detection is a two-stage fallback, because papers name the same
  content differently: it first matches generic names (methodology, results,
  …), and if the paper has no such heading it returns the list of headings
  that paper *does* use, so the agent can map what it wanted onto the
  paper's own naming and ask again. "Attention Is All You Need" has no
  "Methodology" section — its method lives under "Model Architecture" — and
  this is what lets the agent find it anyway.
- `compare_papers` doesn't call an external API -- it's a lightweight tool
  that structures a comparison (problem / approach / findings / relationship)
  for two given paper summaries, which the agent's own reasoning then fills
  in. This is intentionally simple: no nested LLM call, just a scaffold the
  main agent uses to reason more systematically.
- `get_citation_count` hits the free Semantic Scholar API (no key needed) to
  fetch total and "influential" citation counts for a paper -- this lets the
  agent judge whether a paper is foundational and widely built upon, versus
  a lightly-cited new preprint, rather than treating every search result as
  equally authoritative.
- `save_summary_to_file` writes the final output to a local markdown file --
  a genuinely different *kind* of tool from the others, since it takes an
  action/produces an artifact rather than just retrieving information.
- The model reasons over the combined tool outputs and writes a final
  summary, explicitly instructed not to invent citations, sections, or
  citation counts -- it can only use what the tools actually returned, and to
  discard search results that merely share a word with the query. That
  reason -> act -> observe -> reason loop, across seven tools spanning
  search, deep-dive analysis, and action/output, is the core of agentic
  behavior, versus a single fixed prompt-response call.

## Reliability

External APIs fail constantly, and in an agent a raised exception kills the
whole run and loses all prior work. So every tool returns its errors as a
*string* the model can reason about, rather than raising. On top of that:

- **Caching.** Within a run the agent often revisits the same paper — pulling
  `methodology` then `results`, or re-checking a citation count. Parsed PDF
  text and citation lookups are cached, so the repeat costs nothing. A
  request never sent can't be rate-limited. Transient failures are
  deliberately *not* cached, so a momentary 429 doesn't become permanent.
- **Retry with exponential backoff and jitter**, honouring `Retry-After`.
  Only genuinely transient statuses (429, 5xx) are retried — a 404 is an
  answer, not a failure.
- **Optional API key** for Semantic Scholar, which moves you off the shared
  anonymous quota that causes most of the rate limiting.

## Design notes

- **Framework:** LangChain 1.x (`langchain.agents.create_agent`)
- **Model:** provider-agnostic via OpenRouter — defaults to
  `google/gemini-2.5-flash-lite`, swap to any tool-calling model by setting
  `MODEL=` in `.env` with no code changes
- **Tools:** 7 custom tools spanning three categories — search/discovery
  (arXiv phrase+keyword search, arXiv ID lookup, web search via
  DuckDuckGo/`ddgs`), deep-dive analysis (PDF section extraction via `pypdf`
  with a two-stage heading fallback, citation impact lookup via the Semantic
  Scholar API), and reasoning/output (a paper-comparison scaffold, and
  saving the final summary to a markdown file)
- **Cost profile:** dominated by *input* tokens, since every tool result is
  resent on each iteration of the agent loop — which is why sections are
  extracted rather than whole papers being loaded, and why a low prompt price
  matters more than a low completion price when choosing a model
- **Tests:** 40 tests that run with no API key and no network, covering the
  parts where the bugs actually were — section extraction, filename
  sanitising, retry/backoff, and caching

## If something breaks

- `arxiv` package API can be slow/rate-limited on heavy use — if a search
  times out, just retry.
- Semantic Scholar rate-limits aggressively without an API key. The tool
  retries with exponential backoff, which clears most 429s on its own. To
  cut them further, get a free key at
  https://www.semanticscholar.org/product/api and set
  `SEMANTIC_SCHOLAR_API_KEY` in `.env` — that moves you off the shared
  anonymous quota onto your own. If a lookup still fails, the tool tells the
  agent to treat the count as unknown rather than inventing one.
- `ddgs` (formerly `duckduckgo-search`) occasionally rate-limits without an
  API key; if web search fails, the agent still works fine using arXiv alone.
- To drop web search entirely, remove `search_web` from the `tools` list at
  the bottom of `tools.py` — everything else keeps working.
- Section extraction is best-effort: it relies on text extraction from the
  PDF, so a scanned or image-only paper will report that no readable text
  could be found rather than returning garbage.
