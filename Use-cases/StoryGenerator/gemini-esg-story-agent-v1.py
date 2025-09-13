from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.tools import Tool
import os
import requests
from dotenv import load_dotenv

# ===============================
# Load environment variables
# ===============================
load_dotenv()

# Required API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Google Search API Wrapper
search_wrapper = GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_CSE_KEY,
    google_cse_id=GOOGLE_CSE_ID,
    k=5
)

# Search tool abstraction for LangChain
search_tool = Tool(
    name="google_search",
    func=search_wrapper.run,
    description="Use this tool to search Google"
)

# LLM: Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ===============================
# State definition
# ===============================
class StoryState(TypedDict):
    topic: str
    queries: list
    search_results: str
    insights: str
    story_draft: str
    story_final: str
    qa_result: Literal["approved", "needs_fix"]
    feedback: str
    iteration: int
    max_iterations: int

# ===============================
# Agent Functions
# ===============================

def form_queries(state: StoryState):
    queries = [
        f"{state['topic']} sustainability best practices 2024",
        f"{state['topic']} ESG challenges and failures",
        f"{state['topic']} mindset shift sustainability",
        f"Recent innovations in {state['topic']} ESG"
    ]
    return {"queries": queries}


def run_research(state: StoryState):
    all_results = []
    for q in state["queries"]:
        google_res = search_tool.run(q)
        news_res = requests.get(
            f"https://newsapi.org/v2/everything?q={q}&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
        ).json()
        snippets = [a["title"] + " - " + a["url"] for a in news_res.get("articles", [])]
        all_results.append(f"Query: {q}\nGoogle: {google_res}\nNews: {snippets}")
    return {"search_results": "\n\n".join(all_results)}


def extract_insights(state: StoryState):
    messages = [
        SystemMessage(content="You analyze sustainability/ESG search results."),
        HumanMessage(content=f"""
From the research below, identify:
- Key problems and failures
- Mindset or cultural barriers
- Best practices that worked
- Inspiring recent cases

Research:
{state['search_results']}
""")
    ]
    insights = llm.invoke(messages).content
    return {"insights": insights}


def draft_story(state: StoryState):
    messages = [
        SystemMessage(content="You are a creative storyteller specializing in ESG and sustainability."),
        HumanMessage(content=f"""
Using these insights:
{state['insights']}

Write an engaging short story (~600-800 words, ~5 minutes read). 
Story structure:
- Hook (why it matters now)
- Problem (real-world failures / challenges)
- Mindset barrier (what prevents change)
- Solutions (best practices + recent examples)
- Ending (hopeful, inspiring, call to action)

Keep it human, narrative-driven, and impactful.
""")
    ]
    draft = llm.invoke(messages).content
    return {"story_draft": draft}


def qa_story(state: StoryState):
    messages = [
        SystemMessage(content="You are a QA editor."),
        HumanMessage(content=f"""
Check the story below. Requirements:
- 600-800 words (5 minutes read)
- Has real-world examples from research
- Narrative flow (hook → problem → mindset → solutions → ending)
- Clear, engaging, factually sound

Story:
{state['story_draft']}
""")
    ]
    resp = llm.invoke(messages).content
    if "approved" in resp.lower():
        return {"qa_result": "approved", "feedback": resp, "story_final": state["story_draft"]}
    else:
        return {
            "qa_result": "needs_fix",
            "feedback": resp,
            "story_final": state["story_draft"]  # <-- keep last draft anyway
        }



def refine_story(state: StoryState):
    messages = [
        SystemMessage(content="You refine stories based on editor feedback."),
        HumanMessage(content=f"Fix this story based on feedback:\n{state['feedback']}\n\nStory:\n{state['story_draft']}")
    ]
    new_story = llm.invoke(messages).content
    return {"story_draft": new_story, "iteration": state["iteration"] + 1}

# ===============================
# Routing Logic
# ===============================

def route_qa(state: StoryState):
    if state["qa_result"] == "approved" or state["iteration"] >= state["max_iterations"]:
        return "done"
    return "fix"

# ===============================
# Build the Workflow Graph
# ===============================

graph = StateGraph(StoryState)
graph.add_node("form_queries", form_queries)
graph.add_node("research", run_research)
graph.add_node("insights", extract_insights)
graph.add_node("draft", draft_story)
graph.add_node("qa", qa_story)
graph.add_node("refine", refine_story)

graph.add_edge(START, "form_queries")
graph.add_edge("form_queries", "research")
graph.add_edge("research", "insights")
graph.add_edge("insights", "draft")
graph.add_edge("draft", "qa")
graph.add_conditional_edges("qa", route_qa, {"done": END, "fix": "refine"})
graph.add_edge("refine", "qa")

workflow = graph.compile()

# ===============================
# Usage
# ===============================
if __name__ == "__main__":
    initial_state = {
        "topic": "Plastic Waste in Oceans",  # change this to any ESG topic
        "iteration": 1,
        "max_iterations": 2
    }

    result = workflow.invoke(initial_state)
    print("\nFinal ESG Story (5-min read):\n")
    print(result["story_final"])
