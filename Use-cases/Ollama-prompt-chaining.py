from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama      
from typing import TypedDict

model = ChatOllama(model="phi3:latest")        

class BlogState(TypedDict):
    title: str
    outline: str
    content: str

def create_outline(state: BlogState) -> BlogState:
    title = state['title']
    prompt = f"Generate a detailed outline for a blog on the topic - {title}"
    print(prompt)
    response = model.invoke(prompt)
    state['outline'] = response.content if hasattr(response, "content") else response
    print(state)
    return state

def create_blog(state: BlogState) -> BlogState:
    title = state['title']
    outline = state['outline']
    print(outline)
    prompt = f"Write a detailed blog on the title - {title} using the following outline:\n{outline}"
    response = model.invoke(prompt)
    state['content'] = response.content if hasattr(response, "content") else response
    print(state)
    return state

graph = StateGraph(BlogState)

graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)
graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

workflow = graph.compile()

initial_state = {'title': 'Rise of AI in India'}
final_state = workflow.invoke(initial_state)

print(final_state)
print(final_state['outline'])
print(final_state['content'])
