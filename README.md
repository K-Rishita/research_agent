# Research Assistant Agent

A tool-calling AI agent built with LangChain that researches any topic —
it searches arXiv for academic papers, optionally searches the web for
broader context, can pull a specific section (methodology, results, etc.)
out of one paper's actual PDF, checks citation counts to judge how
influential a paper actually is, can structure a comparison between two
papers, and can save the final summary to a file. The model decides for
itself which tool(s) to use and how many times, before writing the final
answer.

## Setup (takes ~5-10 minutes)

1. Install dependencies:
   ```bash
   pip install langchain langchain-openai requests arxiv ddgs pypdf
   ```

2. Set your OpenRouter API key:
   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```
   (or paste it directly into the `OPENROUTER_API_KEY` line in the script)

3. Run it:
   ```bash
   python research_agent.py
   ```

4. Try a topic like `"long-context multi-agent LLM evaluation"` or
   `"backward reachability analysis for neural network alignment"` —
   both close to your own research area, so you'll be able to judge the
   quality of what it finds.

## What makes this an "agent" and not just an API call

- The LLM is given two tools (`search_arxiv`, `search_web`) and decides for
  itself which to call, how many times, and in what order, based on the
  topic — this is LangChain's tool-calling agent pattern
  (`create_tool_calling_agent` + `AgentExecutor`).
- `search_arxiv` hits arXiv's free public API directly (no key needed) and
  returns real titles, authors, dates, and abstracts.
- `search_web` uses DuckDuckGo's search (via the `ddgs` package, also no key
  needed) for context arXiv alone might miss.
- `get_paper_section` downloads a paper's actual PDF from arXiv and extracts
  one named section (abstract, introduction, related work, methodology,
  results, conclusion, references) via header pattern-matching, rather than
  handing the whole paper to the model -- keeps token usage sane and lets the
  agent "zoom in" on exactly the part relevant to the user's question.
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
  citation counts -- it can only use what the tools actually returned. That
  reason -> act -> observe -> reason loop, across six different tools
  spanning search, deep-dive analysis, and action/output, is the core of
  agentic behavior, versus a single fixed prompt-response call.

## Notes for describing this on an application

- **Framework:** LangChain (`langchain.agents.create_tool_calling_agent`)
- **Model:** any OpenRouter-hosted model (defaults to `gpt-4o-mini`, swap the
  `model=` string for any other OpenRouter model id)
- **Tools:** 6 custom tools spanning three categories -- search/discovery
  (arXiv search via the `arxiv` package, web search via DuckDuckGo/`ddgs`),
  deep-dive analysis (PDF section extraction via `pypdf` with regex-based
  section detection, citation impact lookup via the Semantic Scholar API),
  and reasoning/output (a paper-comparison scaffold, and saving the final
  summary to a markdown file)
- **What it does:** given a research topic, autonomously searches academic
  and web sources, can drill into a specific section of a specific paper's
  PDF, can judge how influential a paper actually is via citation data, can
  structure a comparison between two papers, and can persist the final
  summary to disk -- directly relevant to research work like benchmarking
  LLMs or evaluating alignment methods.

## If something breaks

- `arxiv` package API can be slow/rate-limited on heavy use — if a search
  times out, just retry.
- `ddgs` (formerly `duckduckgo-search`) occasionally rate-limits without an
  API key; if web search fails, the agent will still work fine using arXiv
  alone.
- If you'd rather not deal with web search fragility at all, you can strip
  the `search_web` tool out entirely and keep just `search_arxiv` -- still a
  fully valid, working agent for the questionnaire.
