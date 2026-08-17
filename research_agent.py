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
    pip install langchain langchain-openai python-dotenv requests arxiv ddgs pypdf

    Then put your key in a .env file next to this script:
        OPENROUTER_API_KEY=sk-or-...

Run:
    python research_agent.py
"""

import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import tools

# ---------------------------------------------------------------------------
# 1. Configure the model to run through OpenRouter (OpenAI-compatible API)
# ---------------------------------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    sys.exit(
        "OPENROUTER_API_KEY is not set.\n"
        "Create a .env file next to this script containing:\n"
        "    OPENROUTER_API_KEY=sk-or-...\n"
        "or export it in your shell."
    )

# Any OpenRouter model id works here; override with MODEL in .env to swap.
#
# The model must support tool calling, or the agent can't do anything. Cost
# here is dominated by *input* tokens, since every tool result (abstracts,
# PDF sections) is resent on each iteration of the loop -- so prefer a low
# prompt price and a large context window over a low completion price.
# See .env.example for cheaper and stronger alternatives.
MODEL = os.getenv("MODEL", "google/gemini-2.5-flash-lite")

llm = ChatOpenAI(
    model=MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# ---------------------------------------------------------------------------
# 2. Build the agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
     "You are a research assistant. Given a topic, use the available tools "
     "to find relevant papers and background context. Prefer arXiv for "
     "academic depth, and use web search for context, recency, or when arXiv "
     "results are thin. If the user wants to go deeper on one specific paper "
     "(e.g. its methodology or results), use get_paper_section to pull that "
     "section directly from the PDF rather than relying on the abstract alone. "
     "Papers name sections differently, so if get_paper_section reports that "
     "the section does not exist and lists the paper's actual headings, pick "
     "the heading that best matches what you were after and call it again "
     "with that name -- a paper's methodology may live under 'Model "
     "Architecture', 'Approach' or 'Method'. Do not give up after one miss. "
     "Use get_citation_count to check how influential or foundational a paper "
     "is before treating it as authoritative, especially when multiple papers "
     "could be cited and you need to judge which ones matter most. "
     "If the user wants two papers compared, use compare_papers to structure "
     "the comparison, then write out the actual comparison yourself following "
     "that structure. If the user asks to save the summary, use "
     "save_summary_to_file at the end. Write a clear summary of the topic's "
     "current state, and cite every paper you reference by title and arXiv "
     "ID or URL. Do not invent papers, sections, citation counts, or "
     "citations -- only use what the tools actually returned.\n\n"
     "Search results are often noisy: a keyword search for a specific named "
     "system will return unrelated papers that merely share a word in the "
     "name. Discard those. A short summary covering only genuinely relevant "
     "papers is much better than a padded one that lists everything the "
     "search returned -- never include a paper just because it appeared in "
     "the results. If a web result mentions an arXiv ID, call get_paper_by_id "
     "to confirm the real title and authors before citing it, rather than "
     "relying on the snippet. If arXiv search does not surface the paper you "
     "expect, try a more specific query before concluding it does not exist."
)

# create_agent returns a compiled graph that runs the full reason -> call
# tool -> observe -> reason loop internally, until the model stops asking
# for tools and produces a final answer.
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# 3. Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    topic = input("Enter a research topic to investigate: ").strip()
    if not topic:
        sys.exit("No topic entered, nothing to research.")

    request = (
        f"Research the topic '{topic}'. Find relevant recent papers on arXiv, "
        f"use web search if useful for extra context, and write a short "
        f"summary with citations."
    )

    # stream_mode="values" lets us print each step as it happens, which is
    # the 1.x equivalent of AgentExecutor's verbose=True.
    final_state = None
    for state in agent.stream({"messages": [("user", request)]},
                              stream_mode="values"):
        final_state = state
        state["messages"][-1].pretty_print()

    print("\n=== RESEARCH SUMMARY ===")
    print(final_state["messages"][-1].content)
