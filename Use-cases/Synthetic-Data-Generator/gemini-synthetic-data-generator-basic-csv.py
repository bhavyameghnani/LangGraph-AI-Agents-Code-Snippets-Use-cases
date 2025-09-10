from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import operator
import os
from dotenv import load_dotenv

# Load env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ----- LLMs -----
generator_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
optimizer_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)

# ----- Structured evaluator -----
class SyntheticDataEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="Feedback on how realistic, diverse, and coherent the synthetic data is.")

structured_evaluator_llm = evaluator_llm.with_structured_output(SyntheticDataEvaluation)

# ----- STATE -----
class SyntheticDataState(TypedDict):
    raw_text: str
    outline: str
    synthetic_sample: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
    synthetic_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]

# ----- NODES -----
def extract_outline(state: SyntheticDataState):
    """Understand the structure of the original data."""
    messages = [
        SystemMessage(content="You are a data analyst. You extract structure and patterns from raw text data."),
        HumanMessage(content=f"""
Analyze the following raw dataset (100+ lines of text) and describe its structure, key fields, and relationships:

{state['raw_text']}
""")
    ]
    response = generator_llm.invoke(messages).content
    return {'outline': response}

def generate_synthetic_data(state: SyntheticDataState):
    """Generate synthetic data based on the outline."""
    messages = [
        SystemMessage(content="You are a synthetic data generator."),
        HumanMessage(content=f"""
Using this outline:

{state['outline']}

Generate a realistic synthetic data sample (one record set). Keep it logically consistent, creative, and useful for testing.
""")
    ]
    response = generator_llm.invoke(messages).content
    return {'synthetic_sample': response, 'synthetic_history': [response]}

def evaluate_synthetic_data(state: SyntheticDataState):
    """Evaluate if the synthetic data is good enough."""
    messages = [
        SystemMessage(content="You are a strict data quality evaluator."),
        HumanMessage(content=f"""
Evaluate the following synthetic data for realism, diversity, and consistency:

{state['synthetic_sample']}

Return:
- evaluation: "approved" or "needs_improvement"
- feedback: one paragraph describing strengths and weaknesses
""")
    ]
    response = structured_evaluator_llm.invoke(messages)
    return {'evaluation': response.evaluation, 'feedback': response.feedback, 'feedback_history': [response.feedback]}

def optimize_synthetic_data(state: SyntheticDataState):
    """Improve the synthetic data based on feedback."""
    messages = [
        SystemMessage(content="You improve synthetic data based on feedback."),
        HumanMessage(content=f"""
Improve this synthetic data based on feedback:

Feedback: {state['feedback']}

Original Synthetic Data:
{state['synthetic_sample']}

Return a new, improved synthetic data sample.
""")
    ]
    response = optimizer_llm.invoke(messages).content
    iteration = state['iteration'] + 1
    return {'synthetic_sample': response, 'iteration': iteration, 'synthetic_history': [response]}

def route_evaluation(state: SyntheticDataState):
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'

# ----- BUILD GRAPH -----
graph = StateGraph(SyntheticDataState)

graph.add_node('extract_outline', extract_outline)
graph.add_node('generate_synthetic_data', generate_synthetic_data)
graph.add_node('evaluate_synthetic_data', evaluate_synthetic_data)
graph.add_node('optimize_synthetic_data', optimize_synthetic_data)

graph.add_edge(START, 'extract_outline')
graph.add_edge('extract_outline', 'generate_synthetic_data')
graph.add_edge('generate_synthetic_data', 'evaluate_synthetic_data')

graph.add_conditional_edges(
    'evaluate_synthetic_data',
    route_evaluation,
    {'approved': END, 'needs_improvement': 'optimize_synthetic_data'}
)

graph.add_edge('optimize_synthetic_data', 'evaluate_synthetic_data')

workflow = graph.compile()

# ----- TEST -----
initial_state = {
    "raw_text": """User ID, Name, Age, Purchase Amount, Location
1, Alice, 29, 250.75, New York
2, Bob, 34, 99.99, Los Angeles
3, Charlie, 22, 120.50, Chicago
... (imagine 100+ similar lines) ...
""",
    "iteration": 1,
    "max_iteration": 3
}

result = workflow.invoke(initial_state)

print("\n=== Synthetic Data Generation Result ===\n")
print("Outline:\n", result['outline'])
print("\nFinal Synthetic Data:\n", result['synthetic_sample'])
print("\nFeedback History:\n", result['feedback_history'])
