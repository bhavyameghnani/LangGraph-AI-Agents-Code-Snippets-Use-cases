import os
import json
import time
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ----- Setup -----
load_dotenv()
llm = ChatOllama(model="phi3:latest")

# ----- State -----
class ReviewState(TypedDict):
    page_number: int
    value: str
    corrected_value: str
    qa_result: Literal["approved", "needs_fix"]
    feedback: str
    iteration: int
    max_iterations: int

# ----- Agents -----
def correct_text(state: ReviewState):
    """Agent 1: Fix spelling/grammar mistakes"""
    messages = [
        SystemMessage(content="You are an expert text corrector. Fix grammar, spelling, and clarity."),
        HumanMessage(content=f"Correct the following text:\n\n{state['value']}\n\nReturn only the corrected version.")
    ]
    corrected = llm.invoke(messages).content.strip()
    return {"corrected_value": corrected}

def qa_review(state: ReviewState):
    """Agent 2: Review the correction"""
    messages = [
        SystemMessage(content="You are a QA reviewer for text corrections."),
        HumanMessage(content=f"""
Review the following corrected text for grammar and spelling quality:

{state['corrected_value']}

Respond with:
- 'approved' if it's good
- 'needs_fix' if improvements are needed
Include reasoning.
""")
    ]
    resp = llm.invoke(messages).content
    if "approved" in resp.lower():
        return {"qa_result": "approved", "feedback": resp}
    else:
        return {"qa_result": "needs_fix", "feedback": resp}

def refine_text(state: ReviewState):
    """Improve based on QA feedback"""
    messages = [
        SystemMessage(content="You refine text corrections based on QA feedback."),
        HumanMessage(content=f"""
Fix the text below based on feedback.

Feedback: {state['feedback']}

Text:
{state['corrected_value']}
""")
    ]
    refined = llm.invoke(messages).content.strip()
    return {"corrected_value": refined, "iteration": state["iteration"] + 1}

# ----- Routing -----
def route_qa(state: ReviewState):
    if state["qa_result"] == "approved" or state["iteration"] >= state["max_iterations"]:
        return "done"
    return "fix"

# ----- Graph -----
graph = StateGraph(ReviewState)
graph.add_node("correct", correct_text)
graph.add_node("qa", qa_review)
graph.add_node("refine", refine_text)

graph.add_edge(START, "correct")
graph.add_edge("correct", "qa")
graph.add_conditional_edges("qa", route_qa, {"done": END, "fix": "refine"})
graph.add_edge("refine", "qa")

workflow = graph.compile()

# ----- OCR + Processing -----
def process_pdf(pdf_path, output_path="output_corrected.json"):
    pages = convert_from_path(pdf_path)
    results = {}

    for i, page in enumerate(pages, start=1):
        text = pytesseract.image_to_string(page)

        initial_state = {
            "page_number": i,
            "value": text,
            "iteration": 1,
            "max_iterations": 2
        }
        result = workflow.invoke(initial_state)
        results[f"Page_{i}"] = result["corrected_value"]

        print(f"✅ Processed Page {i}")
        time.sleep(3)  # to avoid API overload

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Corrected output saved to {output_path}")

# Example usage
process_pdf("input/scan-sample.pdf")
