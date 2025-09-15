from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults
import os
from dotenv import load_dotenv

# ----- Load env -----
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ----- LLM -----
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ----- Search Tool -----
search_tool = DuckDuckGoSearchResults(max_results=20)

# ----- State -----
class CompetitorState(TypedDict):
    company_name: str
    search_query: str
    search_results: str
    raw_competitor_info: str
    competitors_json: str
    qa_result: Literal["approved", "needs_fix"]
    feedback: str
    iteration: int
    max_iterations: int

# ----- Agents -----
def form_query(state: CompetitorState):
    """Agent 1: Form search query"""
    search_query = f"Top 10 competitors of {state['company_name']} startup"
    return {"search_query": search_query}

def run_search(state: CompetitorState):
    """Agent 2: Run web search"""
    results = search_tool.run(state["search_query"])
    return {"search_results": str(results)}

def extract_competitors(state: CompetitorState):
    """Agent 3: Extract competitor details from search"""
    messages = [
        SystemMessage(content="You extract structured competitor information from raw search results."),
        HumanMessage(content=f"""
From the search results below, identify the top 10 competitors of {state['company_name']}.
For each competitor, include:
- name
- theme/domain
- HQ or main location
- funding stage or size (if available)
- USP/differentiator
- founding_year (if available)
- founders (array of names if available)
- CEO (if available)
- investors (array/list of investors/funders if available)

Search Results:
{state['search_results']}
""")
    ]
    raw_info = llm.invoke(messages).content
    return {"raw_competitor_info": raw_info}

def structure_json(state: CompetitorState):
    """Agent 4: Convert to JSON"""
    messages = [
        SystemMessage(content="You format competitor research into clean JSON."),
        HumanMessage(content=f"""
Convert the following competitor research into a JSON array with fields:
- name
- theme
- location
- funding_stage
- usp
- founding_year
- founders (array of names)
- ceo
- investors (array of names)

Competitor Info:
{state['raw_competitor_info']}
""")
    ]
    competitors_json = llm.invoke(messages).content
    return {"competitors_json": competitors_json}

def qa_json(state: CompetitorState):
    """Agent 5: Validate JSON"""
    messages = [
        SystemMessage(content="You are a QA validator for structured data."),
        HumanMessage(content=f"""
Validate the competitor JSON below. 
Rules:
- Must be valid JSON array.
- Must contain exactly 10 items.
- Each item must have all 9 fields filled:
  - name, theme, location, funding_stage, usp, founding_year, founders, ceo, investors
- founders and investors must be arrays (can be empty arrays if not found).
- All values must be strings or arrays of strings.

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
    """Agent 6: Refine JSON if QA fails"""
    messages = [
        SystemMessage(content="You refine JSON competitor data."),
        HumanMessage(content=f"""
Fix and improve the competitor JSON below based on feedback:

Feedback: {state['feedback']}

JSON:
{state['competitors_json']}
""")
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
graph.add_node("structure", structure_json)
graph.add_node("qa", qa_json)
graph.add_node("optimize", optimize_json)

graph.add_edge(START, "form_query")
graph.add_edge("form_query", "search")
graph.add_edge("search", "extract")
graph.add_edge("extract", "structure")
graph.add_edge("structure", "qa")
graph.add_conditional_edges("qa", route_qa, {"done": END, "fix": "optimize"})
graph.add_edge("optimize", "qa")

workflow = graph.compile()

# ----- Usage -----
initial_state = {
    "company_name": "Sarvam AI",   # <-- Input is company name now
    "iteration": 1,
    "max_iterations": 2
}

result = workflow.invoke(initial_state)
print("Final JSON Competitors:\n", result["competitors_json"])
