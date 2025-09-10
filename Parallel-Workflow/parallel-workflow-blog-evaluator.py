from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator
import os

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
)

# Define structured output schema
class EvaluationSchema(BaseModel):
    feedback: str = Field(description='Detailed feedback for the blog')
    score: int = Field(description='Score out of 10', ge=0, le=10)

# Wrap model for structured output
structured_model = model.with_structured_output(EvaluationSchema)

# Define Blog evaluation state
class BlogEvalState(TypedDict):
    blog: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

# Evaluation nodes
def evaluate_language(state: BlogEvalState):
    prompt = f"Evaluate the language quality of the following blog and provide feedback and a score out of 10:\n\n{state['blog']}"
    output = structured_model.invoke(prompt)
    return {'language_feedback': output.feedback, 'individual_scores': [output.score]}

def evaluate_analysis(state: BlogEvalState):
    prompt = f"Evaluate the depth of analysis of the following blog and provide feedback and a score out of 10:\n\n{state['blog']}"
    output = structured_model.invoke(prompt)
    return {'analysis_feedback': output.feedback, 'individual_scores': [output.score]}

def evaluate_clarity(state: BlogEvalState):
    prompt = f"Evaluate the clarity of thought of the following blog and provide feedback and a score out of 10:\n\n{state['blog']}"
    output = structured_model.invoke(prompt)
    return {'clarity_feedback': output.feedback, 'individual_scores': [output.score]}

def final_evaluation(state: BlogEvalState):
    prompt = f"""Based on the following feedbacks, create a summarized feedback:
- Language feedback: {state['language_feedback']}
- Depth of analysis feedback: {state['analysis_feedback']}
- Clarity of thought feedback: {state['clarity_feedback']}
"""
    overall_feedback = model.invoke(prompt).content
    avg_score = sum(state['individual_scores']) / len(state['individual_scores'])
    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}

# Build workflow graph
graph = StateGraph(BlogEvalState)
graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_clarity', evaluate_clarity)
graph.add_node('final_evaluation', final_evaluation)

# Define edges (parallel evaluation → summary)
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_clarity')

graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_clarity', 'final_evaluation')

graph.add_edge('final_evaluation', END)

workflow = graph.compile()

# Example blog for testing
sample_blog = """
AI and the Future of Creativity
Artificial Intelligence is no longer a distant concept—it’s a daily companion in creative work. 
From writing code to composing music and generating art, AI systems are shaping how humans imagine and create. 
But will AI replace creativity or amplify it? The future may be a blend—machines accelerating ideas, while human intuition and emotion steer them. 
To keep creativity human-centered, we need ethical frameworks, transparency, and collaboration between creators and technologists.
"""

initial_state = {
    'blog': sample_blog
}

# Run the evaluation
final_state = workflow.invoke(initial_state)

print("=== Final Blog Evaluation ===\n")
print("Language Feedback:", final_state['language_feedback'], "\n")
print("Analysis Feedback:", final_state['analysis_feedback'], "\n")
print("Clarity Feedback:", final_state['clarity_feedback'], "\n")
print("Overall Feedback:", final_state['overall_feedback'], "\n")
print("Average Score:", final_state['avg_score'])
