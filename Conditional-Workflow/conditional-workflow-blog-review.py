from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# ---- SCHEMAS ----
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description='Sentiment of the blog feedback')

class DiagnosisSchema(BaseModel):
    issue_type: Literal["Clarity", "Engagement", "Structure", "Tone", "Other"] = Field(description='Type of issue found in the blog')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='Tone of the feedback')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent the improvement is')

# Structured models
structured_model = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)

# ---- STATE ----
class BlogFeedbackState(TypedDict):
    feedback: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str

# ---- NODES ----
def find_sentiment(state: BlogFeedbackState):
    """Determine sentiment of the blog feedback"""
    prompt = f"What is the sentiment (positive or negative) of this blog feedback:\n\n{state['feedback']}"
    sentiment = structured_model.invoke(prompt).sentiment
    return {'sentiment': sentiment}

def check_sentiment(state: BlogFeedbackState) -> Literal["positive_response", "run_diagnosis"]:
    """Decide next step based on sentiment"""
    return 'positive_response' if state['sentiment'] == 'positive' else 'run_diagnosis'

def positive_response(state: BlogFeedbackState):
    """Generate a warm thank-you message"""
    prompt = f"""Write a warm thank-you reply for this positive blog feedback:
    \n\n\"{state['feedback']}\"\n
Also, invite the user to share this blog with others."""
    response = model.invoke(prompt).content
    return {'response': response}

def run_diagnosis(state: BlogFeedbackState):
    """Diagnose what went wrong in the blog feedback"""
    prompt = f"""Analyze this negative blog feedback:\n\n{state['feedback']}\n
Return issue_type, tone, and urgency."""
    response = structured_model2.invoke(prompt)
    return {'diagnosis': response.model_dump()}

def negative_response(state: BlogFeedbackState):
    """Generate an empathetic improvement suggestion"""
    diagnosis = state['diagnosis']
    prompt = f"""You are a blog editor.
The user mentioned a '{diagnosis['issue_type']}' issue, tone is '{diagnosis['tone']}', urgency is '{diagnosis['urgency']}'.
Write an empathetic message acknowledging their feedback and suggest specific improvements."""
    response = model.invoke(prompt).content
    return {'response': response}

# ---- BUILD GRAPH ----
graph = StateGraph(BlogFeedbackState)
graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)

graph.add_edge(START, 'find_sentiment')
graph.add_conditional_edges('find_sentiment', check_sentiment)
graph.add_edge('positive_response', END)
graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END)

workflow = graph.compile()

# ---- TEST ----
initial_state = {
    'feedback': "The blog was too long and lacked a clear structure. I got lost halfway through and had to stop reading."
}

result = workflow.invoke(initial_state)
print("\n=== Blog Feedback Analysis Result ===\n")
print(result['response'])
