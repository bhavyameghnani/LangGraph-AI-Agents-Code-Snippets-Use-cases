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
    return vectordb, chunks

# ----- State -----
class SyntheticState(TypedDict):
    raw_text: str
    vectordb: FAISS
    chunks: list[str]
    synthetic_chunks: Annotated[list[str], operator.add]
    synthetic_data: str
    qa_result: Literal["approved", "needs_fix"]
    feedback: str
    iteration: int
    max_iterations: int

# ----- Agents -----
def chunkwise_synthetic(state: SyntheticState):
    synthetic_chunks = []
    for idx, chunk in enumerate(state["chunks"]):
        messages = [
            SystemMessage(content="You are an expert at creating synthetic but realistic SEC filings."),
            HumanMessage(content=f"""
Rewrite the following section into a synthetic version:
- Maintain overall structure, tone, and section purpose
- Replace all sensitive values (names, amounts, dates, identifiers) with realistic but fake alternatives
- Ensure consistency with typical SEC filing language
- Do not shorten or summarize — produce roughly same length

Section {idx+1}:
{chunk}
""")
        ]
        synthetic_chunks.append(llm.invoke(messages).content)
    return {"synthetic_chunks": synthetic_chunks}

def assemble_synthetic(state: SyntheticState):
    synthetic_data = "\n\n".join(state["synthetic_chunks"])
    return {"synthetic_data": synthetic_data}

def qa_synthetic(state: SyntheticState):
    messages = [
        SystemMessage(content="You are a QA agent verifying synthetic SEC filings."),
        HumanMessage(content=f"""
Review the following synthetic SEC filing:
{state['synthetic_data']}

Check for:
- Real data leakage (e.g., actual company names or identifiers)
- Structural consistency with real SEC filings
- Tone, language, and formatting
Return 'approved' or 'needs_fix' with reasoning.
""")
    ]
    resp = llm.invoke(messages).content
    if "approved" in resp.lower():
        return {"qa_result": "approved", "feedback": resp}
    else:
        return {"qa_result": "needs_fix", "feedback": resp}

def optimize_synthetic(state: SyntheticState):
    messages = [
        SystemMessage(content="You refine synthetic SEC filings."),
        HumanMessage(content=f"""
Improve the synthetic filing below based on the feedback:
Feedback: {state['feedback']}

Filing:
{state['synthetic_data']}
""")
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
graph.add_node("chunkwise_synthetic", chunkwise_synthetic)
graph.add_node("assemble_synthetic", assemble_synthetic)
graph.add_node("qa", qa_synthetic)
graph.add_node("optimize", optimize_synthetic)

graph.add_edge(START, "chunkwise_synthetic")
graph.add_edge("chunkwise_synthetic", "assemble_synthetic")
graph.add_edge("assemble_synthetic", "qa")
graph.add_conditional_edges("qa", route_qa, {"done": END, "fix": "optimize"})
graph.add_edge("optimize", "qa")

workflow = graph.compile()

# ----- Usage -----
raw_doc = open("input/big_doc.txt").read()
vectordb, chunks = build_vectorstore(raw_doc)

initial_state = {
    "raw_text": raw_doc,
    "vectordb": vectordb,
    "chunks": chunks,
    "iteration": 1,
    "max_iterations": 3
}

result = workflow.invoke(initial_state)
with open("synthetic_big_doc.txt", "w") as f:
    f.write(result["synthetic_data"])

print("Synthetic SEC filing generated:", "output/synthetic_big_doc.txt")
