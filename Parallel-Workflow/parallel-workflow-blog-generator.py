from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
)

# Define the blog state
class BlogState(TypedDict):
    topic: str
    title: str
    short_story: str
    plot_twist: str
    conclusion: str

# Generate a title
def generate_title(state: BlogState):
    prompt = f"Generate an engaging blog title about: {state['topic']}."
    response = model.invoke([("user", prompt)])
    return {'title': response.content}

# Generate a short story (around 5 minutes read ~ 600-700 words)
def generate_short_story(state: BlogState):
    prompt = f"Write a short story (~600-700 words, around a 5-minute read) based on the topic: {state['topic']}."
    response = model.invoke([("user", prompt)])
    return {'short_story': response.content}

# Generate a plot twist
def generate_plot_twist(state: BlogState):
    prompt = f"Create a surprising plot twist for a blog about: {state['topic']}. Make it engaging and unexpected."
    response = model.invoke([("user", prompt)])
    return {'plot_twist': response.content}

# Generate a conclusion
def generate_conclusion(state: BlogState):
    prompt = f"Write a compelling conclusion (~150-200 words) for a blog about: {state['topic']}."
    response = model.invoke([("user", prompt)])
    return {'conclusion': response.content}

# Build the graph
graph = StateGraph(BlogState)

# Add nodes
graph.add_node('generate_title', generate_title)
graph.add_node('generate_short_story', generate_short_story)
graph.add_node('generate_plot_twist', generate_plot_twist)
graph.add_node('generate_conclusion', generate_conclusion)

# Parallel edges (all start from START)
graph.add_edge(START, 'generate_title')
graph.add_edge(START, 'generate_short_story')
graph.add_edge(START, 'generate_plot_twist')
graph.add_edge(START, 'generate_conclusion')

# All converge to END
graph.add_edge('generate_title', END)
graph.add_edge('generate_short_story', END)
graph.add_edge('generate_plot_twist', END)
graph.add_edge('generate_conclusion', END)

# Compile workflow
workflow = graph.compile()

# Example usage
initial_state = {
    'topic': 'The Future of AI in Everyday Life'
}

final_state = workflow.invoke(initial_state)

print("\n=== Generated Blog ===\n")
print("Title:\n", final_state['title'], "\n")
print("Short Story:\n", final_state['short_story'], "\n")
print("Plot Twist:\n", final_state['plot_twist'], "\n")
print("Conclusion:\n", final_state['conclusion'], "\n")
