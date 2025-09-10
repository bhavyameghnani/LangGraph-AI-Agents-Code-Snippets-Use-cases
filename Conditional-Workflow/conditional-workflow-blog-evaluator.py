from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)

# ----- STATE -----
class BlogState(TypedDict):
    blog: str
    classification: str
    result: str

# ----- NODES -----
def classify_blog(state: BlogState):
    """Classify the blog quality"""
    prompt = f"""You are a blog evaluator. Classify the following blog as one of:
    - Excellent
    - Needs Improvement
    - Poor
    Only respond with one of these three words.
    
    Blog:
    {state['blog']}"""
    resp = model.invoke(prompt)
    return {'classification': resp.content.strip()}

def handle_excellent(state: BlogState):
    prompt = f"Summarize the following excellent blog in 3 sentences:\n\n{state['blog']}"
    resp = model.invoke(prompt)
    return {'result': f"Blog is Excellent.\nSummary:\n{resp.content.strip()}"}

def handle_needs_improvement(state: BlogState):
    prompt = f"Suggest improvements to make the following blog better:\n\n{state['blog']}"
    resp = model.invoke(prompt)
    return {'result': f"Blog Needs Improvement.\nSuggestions:\n{resp.content.strip()}"}

def handle_poor(state: BlogState):
    prompt = f"The following blog is poor. Suggest a complete rewrite idea in 3 bullet points:\n\n{state['blog']}"
    resp = model.invoke(prompt)
    return {'result': f"Blog is Poor.\nRewrite Plan:\n{resp.content.strip()}"}

# ----- CONDITIONAL -----
def check_quality(state: BlogState) -> Literal["handle_excellent", "handle_needs_improvement", "handle_poor"]:
    c = state['classification'].lower()
    if "excellent" in c:
        return "handle_excellent"
    elif "needs" in c:
        return "handle_needs_improvement"
    else:
        return "handle_poor"

# ----- BUILD GRAPH -----
graph = StateGraph(BlogState)

graph.add_node('classify_blog', classify_blog)
graph.add_node('handle_excellent', handle_excellent)
graph.add_node('handle_needs_improvement', handle_needs_improvement)
graph.add_node('handle_poor', handle_poor)

graph.add_edge(START, 'classify_blog')
graph.add_conditional_edges('classify_blog', check_quality)
graph.add_edge('handle_excellent', END)
graph.add_edge('handle_needs_improvement', END)
graph.add_edge('handle_poor', END)

workflow = graph.compile()

# ----- TEST -----
initial_state = {
    'blog': """AI is changing the future of creativity. While some fear it will replace human imagination,
    others believe it will amplify it. In this article, we explore how AI and humans can collaborate
    to create innovative works that were never before possible."""
}

result = workflow.invoke(initial_state)

print("=== Blog Workflow Result ===\n")
print(result['result'])
