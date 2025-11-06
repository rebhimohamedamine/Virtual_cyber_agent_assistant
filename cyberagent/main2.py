from fastapi import FastAPI
from pydantic import BaseModel
import os
from typing import TypedDict, Optional
import json
import logging

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import StrOutputParser

# === Setup logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Env Variables ===
os.environ["LITELLM_PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# === FastAPI App ===
app = FastAPI()

# === Request Model ===
class Input(BaseModel):
    question: str

# === State Definition for LangGraph ===
class AgentSelectionState(TypedDict):
    question: str
    selected_agent: str
    requires_diagram: bool
    diagram_type: Optional[str]
    brief_description: Optional[str]

# === Prompt with strict JSON instruction and example ===
prompt_template = PromptTemplate.from_template("""
You are an expert agent selector.

Based on this user question: "{question}", choose the most relevant agent from the following list:

- Expert Data Gouvernance workflow: Handles data quality, compliance (e.g., GDPR, HIPAA), metadata management, and data lineage.
- Architecte Cybersécurité workflow: Specializes in threat modeling, vulnerability assessments, secure architectures, and incident response.
- Consultant IA workflow: Advises on AI strategy, model development, MLOps, explainability, and responsible AI implementation.
- Analyste FinOps workflow: Focuses on cloud cost optimization, usage monitoring, budget forecasting, and multi-cloud cost analysis.
- Expert Risk Management workflow: Identifies risks, performs regulatory gap analysis, and supports business continuity planning.
- Spécialiste Identity & Access Management workflow: Designs IAM architectures, including Zero Trust models and secure authentication flows.
- Ingénieur Privacy & Protection workflow: Ensures privacy by design, manages PETs (Privacy Enhancing Technologies), and audits data minimization and consent flows.
- Cloud Architect Workflow: Designs secure, scalable cloud infrastructure and supports hybrid or multi-cloud migration strategies.
- Compliance Officer Workflow: Maintains audit readiness, maps policies to regulations, and oversees compliance implementation.
- DevSecOps Expert Workflow: Integrates security into CI/CD pipelines, manages secrets, and uses SAST/DAST tools to secure codebases.

Additionally, answer this:

Does this question require a visual diagram, architecture, or schema (e.g., system design, network layout, dataflow)?

Respond ONLY with a valid JSON object in the exact format shown below, with no extra text or explanation:

{{
  "selected_agent": "string",
  "requires_diagram": true | false,
  "diagram_type": "string or empty",
  "brief_description": "string or empty"
}}

Example response:

{{
  "selected_agent": "Cloud Architect Workflow",
  "requires_diagram": true,
  "diagram_type": "network",
  "brief_description": "Shows secure communication paths between clouds and data centers"
}}
""")

# === Agent Selector Node ===
def select_agent_node(data: AgentSelectionState) -> AgentSelectionState:
    question = data["question"]
    chain = prompt_template | llm | StrOutputParser()
    output_str = chain.invoke({"question": question})

    logger.info(f"Raw LLM output: {output_str}")

    # Parse JSON output safely
    try:
        result = json.loads(output_str)
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON from LLM output. Returning fallback.")
        result = {
            "selected_agent": "Unknown",
            "requires_diagram": False,
            "diagram_type": "",
            "brief_description": ""
        }
    
    return {
        "question": question,
        "selected_agent": result.get("selected_agent", "").strip(),
        "requires_diagram": result.get("requires_diagram", False),
        "diagram_type": result.get("diagram_type", "").strip(),
        "brief_description": result.get("brief_description", "").strip()
    }

selector_node = RunnableLambda(select_agent_node)

# === LangGraph Definition ===
builder = StateGraph(AgentSelectionState)
builder.add_node("select_agent", selector_node)
builder.set_entry_point("select_agent")
builder.add_edge("select_agent", END)
graph = builder.compile()

# === API Routes ===

@app.get("/")
def root():
    return {"message": "LangGraph API running!"}

@app.post("/choose-agent")
async def choose_agent(input: Input):
    result = graph.invoke({"question": input.question})
    return {
        "query": input.question.strip(),
        "selected_agent": result["selected_agent"],
        "requires_diagram": result["requires_diagram"],
        "diagram_type": result["diagram_type"],
        "brief_description": result["brief_description"]
    }
