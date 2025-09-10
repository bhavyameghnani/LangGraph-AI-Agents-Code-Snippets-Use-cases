# gemini-prompt-chaining.py
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# Import type hints for better code clarity
from typing import TypedDict

import os, getpass
from dotenv import load_dotenv
load_dotenv() 
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ------------------------------------------------------------------------------
# Initialize Gemini model
# ------------------------------------------------------------------------------
# model: which Gemini model to use (gemini-2.0-flash is fast and cost-effective)
# temperature: controls randomness (0.0 = deterministic, 1.0+ = more creative)
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
)

# ------------------------------------------------------------------------------
# Define the structure of the state object that flows through the graph
# ------------------------------------------------------------------------------
# TypedDict ensures the state has the expected keys and types
class BlogState(TypedDict):
    title: str    # The blog title
    outline: str  # The generated outline
    content: str  # The generated blog content

# ------------------------------------------------------------------------------
# Node function 1: create_outline
# Takes the title and generates a blog outline using the LLM
# ------------------------------------------------------------------------------
def create_outline(state: BlogState) -> BlogState:
    prompt = f"Generate a detailed outline for a blog on the topic: {state['title']}"
    response = model.invoke([("user", prompt)])
    state['outline'] = response.content
    return state

# ------------------------------------------------------------------------------
# Node function 2: create_blog
# Takes the title and outline, and generates the full blog content
# ------------------------------------------------------------------------------
def create_blog(state: BlogState) -> BlogState:
    prompt = f"Write a detailed blog titled '{state['title']}' using the following outline:\n{state['outline']}"
    response = model.invoke([("user", prompt)])
    state['content'] = response.content
    return state

# ------------------------------------------------------------------------------
# Build the LangGraph workflow
# ------------------------------------------------------------------------------
# Create a state graph specifying the state type (BlogState)
graph = StateGraph(BlogState)

# Add nodes to the graph, linking functions to step names
graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)

# Define the edges (flow of execution)
graph.add_edge(START, 'create_outline')   # Start → create_outline
graph.add_edge('create_outline', 'create_blog')  # create_outline → create_blog
graph.add_edge('create_blog', END)       # create_blog → End

# Compile the graph into an executable workflow
workflow = graph.compile()

# ------------------------------------------------------------------------------
# Execute the workflow with an initial state
# ------------------------------------------------------------------------------
initial_state = {'title': 'AI assistant for programmers'}

# Run the workflow
final_state = workflow.invoke(initial_state)

# ------------------------------------------------------------------------------
# Display the results
# ------------------------------------------------------------------------------
print(final_state['outline'])  # Print the generated outline
print(final_state['content'])  # Print the generated blog content
