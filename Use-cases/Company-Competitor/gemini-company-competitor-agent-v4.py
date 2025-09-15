from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchResults
import numpy as np
import os
import requests
from dotenv import load_dotenv
from langchain_community.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

# ----- Load env -----
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

search_wrapper = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_CSE_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID"),
    k=10
)

# ----- LLM -----
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ----- Search Tools -----
# Create the tool with wrapper
search_tool = Tool(
    name="google_search",
    func=search_wrapper.run,
    description="Use this tool to search Google"
)
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# ----- Embedding Model (local HuggingFace) -----
embeddings = HuggingFaceEmbeddings(model_name="./models/sentence-transformers/all-MiniLM-L6-v2")

# ----- State -----
class MarketState(TypedDict):
    company_name: str
    search_query: str
    search_results: str
    raw_market_info: str
    themes: list
    enriched_info: str
    funding_info: str
    market_json: str
    qa_result: Literal["approved", "needs_fix"]
    feedback: str
    iteration: int
    max_iterations: int


# ----- Agents -----
def form_query(state: MarketState):
    search_query = f"Market domain, industry sector, and theme of {state['company_name']} startup"
    return {"search_query": search_query}


def run_search(state: MarketState):
    results = search_tool.run(state["search_query"])
    return {"search_results": str(results)}


def extract_market(state: MarketState):
    messages = [
        SystemMessage(content="You extract **market themes/domains** only."),
        HumanMessage(content=f"""
From the following search results, identify the **market theme/domain/industry sector** of {state['company_name']}.
Return only possible themes/domains (one per line, up to 10 max).
Search Results:
{state['search_results']}
""")
    ]
    raw_info = llm.invoke(messages).content
    return {"raw_market_info": raw_info}


def semantic_filter(state: MarketState):
    """
    Use HuggingFaceEmbeddings to select top 3 most semantically relevant themes
    compared to the input company name.
    """
    company = state["company_name"]
    themes = [c.strip() for c in state["raw_market_info"].split("\n") if c.strip()]

    if not themes:
        return {"themes": []}

    # Encode
    query_emb = embeddings.embed_query(company)
    theme_embs = [embeddings.embed_query(c) for c in themes]

    # Compute cosine similarity
    scores = []
    for theme, emb in zip(themes, theme_embs):
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        scores.append((theme, sim))

    # Rank top 3
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    filtered = [name for name, _ in ranked]

    return {"themes": filtered}


def enrich_wikipedia(state: MarketState):
    enriched = []
    for theme in state["themes"]:
        try:
            wiki_data = wiki.run(theme)
            enriched.append(f"{theme}: {wiki_data}")
        except Exception as e:
            enriched.append(f"{theme}: No Wikipedia data found ({e})")
    return {"enriched_info": "\n".join(enriched)}


def enrich_newsapi(state: MarketState):
    enriched_news = {}

    for theme in state["themes"]:
        url = f"https://newsapi.org/v2/everything?q={theme}&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
        resp = requests.get(url).json()

        if "articles" in resp and resp["articles"]:
            snippets = [
                {"title": a["title"], "url": a["url"]}
                for a in resp["articles"]
            ]
            enriched_news[theme] = snippets
        else:
            enriched_news[theme] = [{"title": "No recent news found", "url": ""}]

    return {"funding_info": enriched_news}


def structure_json(state: MarketState):
    messages = [
        SystemMessage(content="You format market research into clean JSON."),
        HumanMessage(content=f"""
Create a JSON array of **exactly 3 themes/domains** with these fields:
- theme
- industry
- sub_industries (list)
- major_players (list)
- market_size
- growth_rate
- use_cases
- recent_news (list of objects: title, url)

Wikipedia Info:
{state['enriched_info']}

Funding & News Info:
{state['funding_info']}
""")
    ]
    market_json = llm.invoke(messages).content
    return {"market_json": market_json}


def qa_json(state: MarketState):
    messages = [
        SystemMessage(content="You are a QA validator for structured JSON."),
        HumanMessage(content=f"""
Validate the JSON below. Ensure:
- It has **exactly 3 items**  
- All fields filled with real data (not 'N/A' unless absolutely unavailable)  
- JSON is valid  

JSON:
{state['market_json']}
""")
    ]
    resp = llm.invoke(messages).content
    if "valid" in resp.lower() or "approved" in resp.lower():
        return {"qa_result": "approved", "feedback": resp}
    else:
        return {"qa_result": "needs_fix", "feedback": resp}


def optimize_json(state: MarketState):
    messages = [
        SystemMessage(content="You refine JSON market data."),
        HumanMessage(content=f"Fix the JSON based on feedback:\n\nFeedback: {state['feedback']}\n\nJSON:\n{state['market_json']}")
    ]
    new_json = llm.invoke(messages).content
    return {"market_json": new_json, "iteration": state["iteration"] + 1}


# ----- Routing -----
def route_qa(state: MarketState):
    if state["qa_result"] == "approved" or state["iteration"] >= state["max_iterations"]:
        return "done"
    return "fix"


# ----- Graph -----
graph = StateGraph(MarketState)
graph.add_node("form_query", form_query)
graph.add_node("search", run_search)
graph.add_node("extract", extract_market)
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
    "company_name": "ChatterBox",  # Input company name
    "iteration": 1,
    "max_iterations": 2
}

result = workflow.invoke(initial_state)
print("Final JSON Market:\n", result["market_json"])
