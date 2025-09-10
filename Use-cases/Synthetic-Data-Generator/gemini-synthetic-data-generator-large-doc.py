from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
import operator
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ----- LLM -----
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = HuggingFaceEmbeddings(model_name="./models/sentence-transformers/all-MiniLM-L6-v2")

# ----- Pre-processing: Chunk + Vector Store -----
def build_vectorstore(raw_text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)
    vectordb = FAISS.from_texts(chunks, embeddings)
    return vectordb

# ----- State -----
class SyntheticState(TypedDict):
    raw_text: str
    vectordb: FAISS
    outline: str
    schema: str
    synthetic_data: str
    qa_result: Literal["approved", "needs_fix"]
    feedback: str
    iteration: int
    max_iterations: int

# ----- Agents -----
def ingest_data(state: SyntheticState):
    retriever = state["vectordb"].as_retriever(search_kwargs={"k": 5})
    context_chunks = retriever.get_relevant_documents("Summarize the key topics and sections")
    context_text = "\n\n".join([doc.page_content for doc in context_chunks])
    messages = [
        SystemMessage(content="You are a text analysis expert."),
        HumanMessage(content=f"Create an outline of topics and sections based on the following content:\n\n{context_text}")
    ]
    outline = llm.invoke(messages).content
    return {"outline": outline}

def analyze_patterns(state: SyntheticState):
    messages = [
        SystemMessage(content="You analyze text for patterns and schemas."),
        HumanMessage(content=f"From the outline below, infer key data fields, entities, relationships, and patterns for synthetic generation:\n\n{state['outline']}")
    ]
    schema = llm.invoke(messages).content
    return {"schema": schema}

def design_synthetic(state: SyntheticState):
    messages = [
        SystemMessage(content="You create synthetic data that mirrors structure but not real values."),
        HumanMessage(content=f"Using the schema below, generate synthetic data preserving relationships, types, and variety. Produce at least 2 versions:\n\n{state['schema']}")
    ]
    synthetic_data = llm.invoke(messages).content
    return {"synthetic_data": synthetic_data}

def qa_synthetic(state: SyntheticState):
    messages = [
        SystemMessage(content="You are a QA agent verifying synthetic data quality."),
        HumanMessage(content=f"Review the synthetic data below for realism, consistency, and no leakage of real sensitive data. Return 'approved' or 'needs_fix' with feedback.\n\n{state['synthetic_data']}")
    ]
    resp = llm.invoke(messages).content
    if "approved" in resp.lower():
        return {"qa_result": "approved", "feedback": resp}
    else:
        return {"qa_result": "needs_fix", "feedback": resp}

def optimize_synthetic(state: SyntheticState):
    messages = [
        SystemMessage(content="You refine synthetic data based on QA feedback."),
        HumanMessage(content=f"Improve the synthetic data below based on the feedback:\n\nFeedback: {state['feedback']}\n\nData:\n{state['synthetic_data']}")
    ]
    new_data = llm.invoke(messages).content
    return {"synthetic_data": new_data, "iteration": state["iteration"] + 1}

# ----- Routing -----
def route_qa(state: SyntheticState):
    if state["qa_result"] == "approved" or state["iteration"] >= state["max_iterations"]:
        return "done"
    return "fix"

# ----- Graph -----
graph = StateGraph(SyntheticState)
graph.add_node("ingest", ingest_data)
graph.add_node("analyze", analyze_patterns)
graph.add_node("design", design_synthetic)
graph.add_node("qa", qa_synthetic)
graph.add_node("optimize", optimize_synthetic)

graph.add_edge(START, "ingest")
graph.add_edge("ingest", "analyze")
graph.add_edge("analyze", "design")
graph.add_edge("design", "qa")
graph.add_conditional_edges("qa", route_qa, {"done": END, "fix": "optimize"})
graph.add_edge("optimize", "qa")

workflow = graph.compile()

# ----- Usage -----
raw_doc = open("big_doc.txt").read()
vectordb = build_vectorstore(raw_doc)

initial_state = {
    "raw_text": raw_doc,
    "vectordb": vectordb,
    "iteration": 1,
    "max_iterations": 3
}

result = workflow.invoke(initial_state)
print(result["synthetic_data"])
