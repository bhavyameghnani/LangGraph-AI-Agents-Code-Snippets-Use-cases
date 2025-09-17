import os
import pandas as pd
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
import operator
import time # Added for the sleep function

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# ----- Setup -----
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
from langchain_ollama import ChatOllama      
from typing import TypedDict

llm = ChatOllama(model="phi3:latest") 

# Global mask dictionary
mask_map = {}

# ----- State -----
class MaskState(TypedDict):
    value: str
    masked_value: str
    qa_result: Literal["approved", "needs_fix"]
    feedback: str
    iteration: int
    max_iterations: int


# ----- Agents -----
def mask_value(state: MaskState):
    """Generate synthetic masked value"""
    if state["value"] in mask_map:
        return {"masked_value": mask_map[state["value"]]}  # already replaced

    messages = [
        SystemMessage(content="You are an expert at anonymizing wealth management data."),
        HumanMessage(content=f"""
Replace the following value with a synthetic but realistic alternative:
- Maintain type (name → fake name, date → fake date, amount → fake amount, ID → fake ID).
- Ensure it's realistic and anonymized.
- Only return the replacement value.

Value: {state['value']}
""")
    ]
    synthetic = llm.invoke(messages).content.strip()
    mask_map[state["value"]] = synthetic
    return {"masked_value": synthetic}


def qa_mask(state: MaskState):
    """Check if masking is done properly"""
    messages = [
        SystemMessage(content="You are a QA agent verifying anonymized values."),
        HumanMessage(content=f"""
Review the following masked value:
{state['masked_value']}

Check for:
- No real data leakage
- Correct type (name, date, amount, etc.)
Return 'approved' or 'needs_fix' with reasoning.
""")
    ]
    resp = llm.invoke(messages).content
    if "approved" in resp.lower():
        return {"qa_result": "approved", "feedback": resp}
    else:
        return {"qa_result": "needs_fix", "feedback": resp}


def optimize_mask(state: MaskState):
    """Improve based on QA feedback"""
    messages = [
        SystemMessage(content="You refine masked values."),
        HumanMessage(content=f"""
Fix the value below based on the feedback:
Feedback: {state['feedback']}

Value:
{state['masked_value']}
""")
    ]
    new_data = llm.invoke(messages).content.strip()
    mask_map[state["value"]] = new_data
    return {"masked_value": new_data, "iteration": state["iteration"] + 1}


# ----- Routing -----
def route_qa(state: MaskState):
    if state["qa_result"] == "approved" or state["iteration"] >= state["max_iterations"]:
        return "done"
    return "fix"


# ----- Graph -----
graph = StateGraph(MaskState)
graph.add_node("mask", mask_value)
graph.add_node("qa", qa_mask)
graph.add_node("optimize", optimize_mask)

graph.add_edge(START, "mask")
graph.add_edge("mask", "qa")
graph.add_conditional_edges("qa", route_qa, {"done": END, "fix": "optimize"})
graph.add_edge("optimize", "qa")

workflow = graph.compile()


# ----- Processing Multiple CSVs -----
input_folder = "input"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Step 1: Collect all values
all_values = set()
for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(input_folder, file))
        for val in df.values.flatten():
            if pd.notna(val) and str(val).strip():
                all_values.add(str(val))

# Step 2: Build consistent mask_map
for val in all_values:
    initial_state = {
        "value": val,
        "iteration": 1,
        "max_iterations": 2
    }
    result = workflow.invoke(initial_state)
    mask_map[val] = result["masked_value"]
    
    # Add a delay to prevent API rate-limiting
    print("Pausing for 4 seconds to respect API rate limits...")
    time.sleep(4)

# Save mapping
with open("output/mask_map.json", "w") as f:
    json.dump(mask_map, f, indent=2)

# Step 3: Apply mapping to each CSV
for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, f"masked_{file}")

        df = pd.read_csv(input_path)
        masked_df = df.applymap(lambda x: mask_map.get(str(x), x) if pd.notna(x) else x)

        masked_df.to_csv(output_path, index=False)

        print(f"✅ Masked CSV saved at {output_path}")