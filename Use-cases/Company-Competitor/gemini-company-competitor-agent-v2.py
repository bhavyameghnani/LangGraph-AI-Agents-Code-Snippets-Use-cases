from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import os
import requests
from dotenv import load_dotenv

# --- For embeddings + similarity ---
from sentence_transformers import SentenceTransformer, util

# ----- Load env -----
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Add to .env

# ----- LLM -----
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ----- Search Tools -----
search_tool = DuckDuckGoSearchResults(max_results=20)
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# ----- Embedding Model (local) -----
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----- State -----
class CompetitorState(TypedDict):
    company_name: str
    search_query: str
    search_results: str
    raw_competitor_info: str
    filtered_competitors: list
    enriched_info: str
    funding_info: str
    competitors_json: str
    qa_result: Literal["approved", "needs_fix"]
    feedback: str
    iteration: int
    max_iterations: int


# ----- Agents -----
def form_query(state: CompetitorState):
    search_query = f"Top 10 competitors of {state['company_name']} startup"
    return {"search_query": search_query}


def run_search(state: CompetitorState):
    results = search_tool.run(state["search_query"])
    return {"search_results": str(results)}


def extract_competitors(state: CompetitorState):
    messages = [
        SystemMessage(content="You extract competitor names only."),
        HumanMessage(content=f"""
From the following search results, identify **possible competitors** of {state['company_name']}.
Return only company names (one per line, up to 15 max).
Search Results:
{state['search_results']}
""")
    ]
    raw_info = llm.invoke(messages).content
    return {"raw_competitor_info": raw_info}


def semantic_filter(state: CompetitorState):
    """
    Use embeddings to select top 5 most semantically similar competitors
    compared to the input company name.
    """
    company = state["company_name"]
    competitors = [c.strip() for c in state["raw_competitor_info"].split("\n") if c.strip()]

    if not competitors:
        return {"filtered_competitors": []}

    # Encode
    query_emb = embedder.encode(company, convert_to_tensor=True)
    comp_embs = embedder.encode(competitors, convert_to_tensor=True)

    # Compute cosine similarity
    scores = util.cos_sim(query_emb, comp_embs)[0]

    # Rank top 5
    scored = list(zip(competitors, scores.tolist()))
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)[:5]
    filtered = [name for name, _ in ranked]

    return {"filtered_competitors": filtered}


def enrich_wikipedia(state: CompetitorState):
    enriched = []
    for comp in state["filtered_competitors"]:
        try:
            wiki_data = wiki.run(comp)
            enriched.append(f"{comp}: {wiki_data}")
        except Exception as e:
            enriched.append(f"{comp}: No Wikipedia data found ({e})")
    return {"enriched_info": "\n".join(enriched)}


def enrich_newsapi(state: CompetitorState):
    enriched_news = {}

    for comp in state["filtered_competitors"]:
        url = f"https://newsapi.org/v2/everything?q={comp}&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
        resp = requests.get(url).json()

        if "articles" in resp and resp["articles"]:
            snippets = [
                {
                    "title": a["title"],
                    "url": a["url"]
                }
                for a in resp["articles"]
            ]
            enriched_news[comp] = snippets
        else:
            enriched_news[comp] = [{"title": "No recent news found", "url": ""}]

    return {"funding_info": enriched_news}


def structure_json(state: CompetitorState):
    messages = [
        SystemMessage(content="You format competitor research into clean JSON."),
        HumanMessage(content=f"""
Create a JSON array of **exactly 5 competitors** with these fields:
- name
- theme/domain
- location
- founding_year
- founders (list)
- ceo
- funding_stage
- investors (list)
- usp
- recent_news (list of objects: title, url)

Competitor Wikipedia Info:
{state['enriched_info']}

Funding & News Info:
{state['funding_info']}
""")
    ]
    competitors_json = llm.invoke(messages).content
    return {"competitors_json": competitors_json}


def qa_json(state: CompetitorState):
    messages = [
        SystemMessage(content="You are a QA validator for structured JSON."),
        HumanMessage(content=f"""
Validate the competitor JSON below. Ensure:
- It has **exactly 5 items**  
- All fields filled with real data (not 'N/A' unless absolutely unavailable)  
- JSON is valid  

JSON:
{state['competitors_json']}
""")
    ]
    resp = llm.invoke(messages).content
    if "valid" in resp.lower() or "approved" in resp.lower():
        return {"qa_result": "approved", "feedback": resp}
    else:
        return {"qa_result": "needs_fix", "feedback": resp}


def optimize_json(state: CompetitorState):
    messages = [
        SystemMessage(content="You refine JSON competitor data."),
        HumanMessage(content=f"Fix the competitor JSON based on feedback:\n\nFeedback: {state['feedback']}\n\nJSON:\n{state['competitors_json']}")
    ]
    new_json = llm.invoke(messages).content
    return {"competitors_json": new_json, "iteration": state["iteration"] + 1}


# ----- Routing -----
def route_qa(state: CompetitorState):
    if state["qa_result"] == "approved" or state["iteration"] >= state["max_iterations"]:
        return "done"
    return "fix"


# ----- Graph -----
graph = StateGraph(CompetitorState)
graph.add_node("form_query", form_query)
graph.add_node("search", run_search)
graph.add_node("extract", extract_competitors)
graph.add_node("semantic_filter", semantic_filter)
graph.add_node("wiki", enrich_wikipedia)
graph.add_node("newsapi", enrich_newsapi)
graph.add_node("structure", structure_json)
graph.add_node("qa", qa_json)
graph.add_node("optimize", optimize_json)

graph.add_edge(START, "form_query")
graph.add_edge("form_query", "search")
graph.add_edge("search", "extract")
graph.add_edge("extract", "semantic_filter")
graph.add_edge("semantic_filter", "wiki")
graph.add_edge("wiki", "newsapi")
graph.add_edge("newsapi", "structure")
graph.add_edge("structure", "qa")
graph.add_conditional_edges("qa", route_qa, {"done": END, "fix": "optimize"})
graph.add_edge("optimize", "qa")

workflow = graph.compile()

# ----- Usage -----
initial_state = {
    "company_name": "OpenAI",  # Input company name
    "iteration": 1,
    "max_iterations": 2
}

result = workflow.invoke(initial_state)
print("Final JSON Competitors:\n", result["competitors_json"])
