"""
Multi-Agent System for Deep Document Intelligence using LangGraph
Implements three specialized agents coordinated through a graph-based orchestration layer.
Now using OpenAI GPT-4.1 as the LLM backend.
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, TypedDict, Optional
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END

# LangChain imports for OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Configure OpenAI via LangChain
OPENAI_API_KEY = os.getenv("openai_api_key")
model_name = os.getenv("openai_model_name", "gpt-4.1")

# Initialize LangChain OpenAI model
llm = ChatOpenAI(
    model=model_name,
    api_key=OPENAI_API_KEY,
    temperature=0.3
)

# ============================================================================
# CACHING SYSTEM FOR AGENT OUTPUTS
# ============================================================================

# Cache file path for storing agent results based on file hash
AGENT_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_cache.json")


def calculate_content_hash(content: str) -> str:
    """
    Calculate SHA256 hash of the document content.
    This is used to identify if the same content has been processed before.
    
    Args:
        content: The text content to hash
        
    Returns:
        Hexadecimal string of the SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content.encode('utf-8'))
    return sha256_hash.hexdigest()


def load_agent_cache() -> Dict:
    """
    Load the agent cache from the JSON file.
    Returns an empty dict if the file doesn't exist or is invalid.
    """
    try:
        if os.path.exists(AGENT_CACHE_FILE):
            with open(AGENT_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load agent cache file: {e}")
    return {}


def save_agent_cache(cache: Dict):
    """
    Save the agent cache to the JSON file.
    """
    try:
        with open(AGENT_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Warning: Could not save agent cache file: {e}")


def get_cached_agent_result(content_hash: str) -> Optional[Dict]:
    """
    Check if a content hash exists in the cache and return the cached agent results.
    
    Args:
        content_hash: The SHA256 hash of the document content
        
    Returns:
        The cached result if found, None otherwise
    """
    cache = load_agent_cache()
    if content_hash in cache:
        print(f"✅ Cache hit! Using cached agent results for hash: {content_hash[:16]}...")
        return cache[content_hash]["result"]
    return None


def save_agent_result_to_cache(content_hash: str, result: Dict, document_preview: str = None):
    """
    Save agent processing results to the cache.
    
    Args:
        content_hash: The SHA256 hash of the document content
        result: The complete agent analysis results
        document_preview: Optional preview of the document for reference
    """
    cache = load_agent_cache()
    cache[content_hash] = {
        "result": result,
        "document_preview": document_preview[:500] if document_preview else None,
        "processed_at": datetime.now().isoformat(),
        "model": model_name
    }
    save_agent_cache(cache)
    print(f"💾 Agent results cached for hash: {content_hash[:16]}...")


def clear_agent_cache():
    """
    Clear all cached agent results.
    """
    if os.path.exists(AGENT_CACHE_FILE):
        os.remove(AGENT_CACHE_FILE)
        print("🗑️ Agent cache cleared.")
    else:
        print("Agent cache is already empty.")


def list_agent_cache():
    """
    List all cached agent results with their metadata.
    """
    cache = load_agent_cache()
    if not cache:
        print("Agent cache is empty.")
        return
    
    print(f"\n📦 Cached agent results ({len(cache)} entries):")
    print("-" * 60)
    for content_hash, data in cache.items():
        processed_at = data.get('processed_at', 'Unknown')
        model_used = data.get('model', 'Unknown')
        preview = data.get('document_preview', 'N/A')[:100]
        print(f"  Hash: {content_hash[:16]}...")
        print(f"  Model: {model_used}")
        print(f"  Processed: {processed_at}")
        print(f"  Preview: {preview}...")
        print("-" * 60)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    Shared state across all agents in the graph.
    This enables context sharing between agents.
    """
    # Input document
    document: str
    
    # Agent outputs
    summary: Optional[str]
    summary_chunks: Optional[List[str]]
    action_items: Optional[Dict]
    risks_analysis: Optional[Dict]
    
    # Metadata
    status: str
    current_agent: str
    errors: List[str]
    
    # Progress tracking
    progress: float
    progress_message: str


# ============================================================================
# AGENT PROMPTS
# ============================================================================

SUMMARY_SYSTEM_PROMPT = """You are a Context-Aware Summary Agent, an expert at analyzing documents and creating comprehensive summaries.

Your responsibilities:
1. Preserve the main intent and purpose of the document
2. Highlight critical decisions mentioned
3. Note any constraints or limitations stated
4. Keep important context and relationships
5. Be concise but thorough

Always structure your response with:
## Executive Summary
[2-3 paragraph overview]

## Key Points
- [Bullet points of main takeaways]

## Critical Decisions
- [Any decisions mentioned]

## Constraints & Requirements
- [Any limitations or requirements noted]
"""

ACTION_SYSTEM_PROMPT = """You are an Action & Dependency Extraction Agent, specialized in identifying actionable tasks from documents.

Extract every action item, task, or to-do mentioned or implied in the document.
For each action item, identify:
1. Task Description: Clear, actionable description
2. Owner: Person/team responsible (if mentioned, else "Unassigned")
3. Deadline: Due date/timeframe (if mentioned, else "Not specified")
4. Priority: High/Medium/Low (infer from context)
5. Dependencies: What must be completed first (if any)
6. Status: Pending/In Progress/Completed (if mentioned)

RESPOND IN THIS EXACT JSON FORMAT:
{
    "action_items": [
        {
            "id": 1,
            "task": "Description of the task",
            "owner": "Person or team name",
            "deadline": "Date or timeframe",
            "priority": "High/Medium/Low",
            "dependencies": ["List of dependent tasks"],
            "status": "Pending",
            "source_context": "Brief quote from document"
        }
    ],
    "dependency_graph": {
        "description": "Overview of task dependencies",
        "critical_path": ["Tasks in sequence"]
    },
    "unassigned_tasks_count": 0,
    "high_priority_count": 0
}

Return ONLY valid JSON, no additional text.
"""

RISK_SYSTEM_PROMPT = """You are a Risk & Open-Issues Agent, expert at identifying potential problems and unresolved questions.

Analyze the document for:
1. Open Questions: Unresolved questions or decisions pending
2. Missing Information: Data or details that should be present but aren't
3. Assumptions: Implicit or explicit assumptions being made
4. Risks: Potential problems, threats, or challenges
5. Ambiguities: Unclear statements that need clarification

RESPOND IN THIS EXACT JSON FORMAT:
{
    "open_questions": [
        {
            "id": 1,
            "question": "The unresolved question",
            "context": "Where this appears",
            "impact": "High/Medium/Low",
            "suggested_resolution": "Recommended next step"
        }
    ],
    "missing_information": [
        {
            "id": 1,
            "description": "What is missing",
            "importance": "Critical/Important/Nice-to-have",
            "impact_if_not_addressed": "Consequence"
        }
    ],
    "assumptions": [
        {
            "id": 1,
            "assumption": "The assumption",
            "risk_if_wrong": "What happens if incorrect",
            "validation_needed": "How to verify"
        }
    ],
    "risks": [
        {
            "id": 1,
            "risk": "Description of risk",
            "category": "Technical/Resource/Timeline/External/Other",
            "likelihood": "High/Medium/Low",
            "impact": "High/Medium/Low",
            "mitigation": "Suggested strategy"
        }
    ],
    "ambiguities": [
        {
            "id": 1,
            "statement": "The ambiguous statement",
            "clarification_needed": "What needs clarification"
        }
    ],
    "risk_score": {
        "overall": "High/Medium/Low",
        "rationale": "Explanation of assessment"
    }
}

Return ONLY valid JSON, no additional text.
"""


# ============================================================================
# AGENT NODE FUNCTIONS
# ============================================================================

def chunk_document(text: str, chunk_size: int = 8000) -> List[str]:
    """Split document into manageable chunks."""
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [text]


def summary_agent(state: AgentState) -> AgentState:
    """
    Context-Aware Summary Agent Node
    Generates concise summaries while preserving intent and critical decisions.
    """
    print("🔄 Running Summary Agent...")
    
    document = state["document"]
    chunks = chunk_document(document)
    chunk_summaries = []
    
    try:
        # Summarize each chunk
        for i, chunk in enumerate(chunks, 1):
            messages = [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=f"""
Summarize this document chunk ({i} of {len(chunks)}):

{chunk}

Provide a focused summary preserving key information.
""")
            ]
            response = llm.invoke(messages)
            chunk_summaries.append(response.content)
        
        # Generate final consolidated summary
        if len(chunks) > 1:
            combined = "\n\n".join([f"[Section {i+1}]:\n{s}" for i, s in enumerate(chunk_summaries)])
            final_messages = [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=f"""
Create a final consolidated summary from these section summaries:

{combined}

Provide a comprehensive executive summary following the required format.
""")
            ]
            final_response = llm.invoke(final_messages)
            final_summary = final_response.content
        else:
            final_summary = chunk_summaries[0] if chunk_summaries else "No summary generated."
        
        return {
            **state,
            "summary": final_summary,
            "summary_chunks": chunk_summaries,
            "current_agent": "summary_complete",
            "progress": 0.33,
            "progress_message": "Summary Agent completed"
        }
        
    except Exception as e:
        return {
            **state,
            "summary": f"Error generating summary: {str(e)}",
            "summary_chunks": [],
            "errors": state.get("errors", []) + [f"Summary Agent Error: {str(e)}"],
            "current_agent": "summary_error",
            "progress": 0.33,
            "progress_message": f"Summary Agent error: {str(e)}"
        }


def action_agent(state: AgentState) -> AgentState:
    """
    Action & Dependency Extraction Agent Node
    Extracts actionable tasks with dependencies, owners, and deadlines.
    """
    print("🔄 Running Action Extraction Agent...")
    
    document = state["document"]
    summary_context = state.get("summary", "")
    
    try:
        messages = [
            SystemMessage(content=ACTION_SYSTEM_PROMPT),
            HumanMessage(content=f"""
Analyze this document and extract ALL action items:

DOCUMENT:
{document[:15000]}

CONTEXT FROM SUMMARY AGENT:
{summary_context[:2000] if summary_context else "No summary available"}

Extract all tasks, action items, and to-dos with their metadata.
""")
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse JSON response
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            action_data = json.loads(response_text.strip())
        except json.JSONDecodeError:
            action_data = {
                "action_items": [],
                "raw_response": response_text,
                "parse_error": "Could not parse JSON response"
            }
        
        return {
            **state,
            "action_items": action_data,
            "current_agent": "action_complete",
            "progress": 0.66,
            "progress_message": "Action Extraction Agent completed"
        }
        
    except Exception as e:
        return {
            **state,
            "action_items": {"action_items": [], "error": str(e)},
            "errors": state.get("errors", []) + [f"Action Agent Error: {str(e)}"],
            "current_agent": "action_error",
            "progress": 0.66,
            "progress_message": f"Action Agent error: {str(e)}"
        }


def risk_agent(state: AgentState) -> AgentState:
    """
    Risk & Open-Issues Agent Node
    Identifies unresolved questions, missing data, assumptions, and risks.
    """
    print("🔄 Running Risk Analysis Agent...")
    
    document = state["document"]
    summary_context = state.get("summary", "")
    action_context = state.get("action_items", {})
    action_count = len(action_context.get("action_items", []))
    
    try:
        messages = [
            SystemMessage(content=RISK_SYSTEM_PROMPT),
            HumanMessage(content=f"""
Analyze this document for risks and open issues:

DOCUMENT:
{document[:15000]}

CONTEXT FROM PREVIOUS AGENTS:
- Summary available: {"Yes" if summary_context else "No"}
- Action items found: {action_count}

Identify all risks, open questions, assumptions, and missing information.
""")
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse JSON response
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            risk_data = json.loads(response_text.strip())
        except json.JSONDecodeError:
            risk_data = {
                "open_questions": [],
                "risks": [],
                "raw_response": response_text,
                "parse_error": "Could not parse JSON response"
            }
        
        return {
            **state,
            "risks_analysis": risk_data,
            "current_agent": "risk_complete",
            "status": "completed",
            "progress": 1.0,
            "progress_message": "All agents completed successfully!"
        }
        
    except Exception as e:
        return {
            **state,
            "risks_analysis": {"risks": [], "open_questions": [], "error": str(e)},
            "errors": state.get("errors", []) + [f"Risk Agent Error: {str(e)}"],
            "current_agent": "risk_error",
            "status": "completed_with_errors",
            "progress": 1.0,
            "progress_message": f"Completed with errors: {str(e)}"
        }


def compile_results(state: AgentState) -> AgentState:
    """
    Final node to compile and format all results.
    """
    print("📊 Compiling final results...")
    
    action_data = state.get("action_items", {})
    risk_data = state.get("risks_analysis", {})
    
    consolidated_report = {
        "executive_summary": state.get("summary", ""),
        "action_items": action_data.get("action_items", []),
        "action_metadata": {
            "total_tasks": len(action_data.get("action_items", [])),
            "high_priority": action_data.get("high_priority_count", 0),
            "unassigned": action_data.get("unassigned_tasks_count", 0)
        },
        "risks_and_issues": {
            "open_questions": risk_data.get("open_questions", []),
            "risks": risk_data.get("risks", []),
            "assumptions": risk_data.get("assumptions", []),
            "missing_info": risk_data.get("missing_information", []),
            "overall_risk_score": risk_data.get("risk_score", {})
        }
    }
    
    return {
        **state,
        "consolidated_report": consolidated_report,
        "status": "completed"
    }


# ============================================================================
# BUILD THE LANGGRAPH WORKFLOW
# ============================================================================

def build_agent_graph():
    """
    Build the LangGraph workflow for multi-agent document analysis.
    
    Graph Structure:
    
        [START]
           |
           v
    +-------------+
    | Summary     |
    | Agent       |
    +-------------+
           |
           v
    +-------------+
    | Action      |
    | Agent       |
    +-------------+
           |
           v
    +-------------+
    | Risk        |
    | Agent       |
    +-------------+
           |
           v
    +-------------+
    | Compile     |
    | Results     |
    +-------------+
           |
           v
         [END]
    """
    
    # Create the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("summary_agent", summary_agent)
    workflow.add_node("action_agent", action_agent)
    workflow.add_node("risk_agent", risk_agent)
    workflow.add_node("compile_results", compile_results)
    
    # Define the edges (flow between agents)
    workflow.set_entry_point("summary_agent")
    workflow.add_edge("summary_agent", "action_agent")
    workflow.add_edge("action_agent", "risk_agent")
    workflow.add_edge("risk_agent", "compile_results")
    workflow.add_edge("compile_results", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

# Create the compiled graph (singleton)
agent_graph = build_agent_graph()


def analyze_document(document: str, progress_callback=None, use_cache: bool = True) -> Dict:
    """
    Main entry point for document analysis using LangGraph.
    Includes caching support to skip re-processing identical documents.
    
    Args:
        document: Text content to analyze
        progress_callback: Optional callback for progress updates (message, progress)
        use_cache: Whether to use caching (default True). Set to False to force re-analysis.
    
    Returns:
        Complete analysis results from all agents
    """
    # Calculate content hash for caching
    content_hash = calculate_content_hash(document)
    print(f"📝 Document content hash: {content_hash}")
    
    # Check cache if enabled
    if use_cache:
        if progress_callback:
            progress_callback("Checking cache for existing analysis...", 0.02)
        
        cached_result = get_cached_agent_result(content_hash)
        if cached_result is not None:
            if progress_callback:
                progress_callback("✅ Retrieved from cache!", 1.0)
            
            # Add cache info to result
            cached_result["from_cache"] = True
            cached_result["content_hash"] = content_hash
            return cached_result
    
    # Not in cache, proceed with agent processing
    print("🔄 Processing document with agents...")
    
    # Initialize state
    initial_state = {
        "document": document,
        "summary": None,
        "summary_chunks": None,
        "action_items": None,
        "risks_analysis": None,
        "status": "processing",
        "current_agent": "starting",
        "errors": [],
        "progress": 0.0,
        "progress_message": "Starting analysis..."
    }
    
    if progress_callback:
        progress_callback("Starting multi-agent analysis with OpenAI GPT-4.1...", 0.05)
    
    try:
        # Run the graph
        final_state = None
        
        # Stream through the graph to get intermediate states
        for state in agent_graph.stream(initial_state):
            # Get the current node's output
            for node_name, node_state in state.items():
                final_state = node_state
                
                if progress_callback and "progress_message" in node_state:
                    progress_callback(
                        node_state.get("progress_message", "Processing..."),
                        node_state.get("progress", 0.5)
                    )
        
        if final_state is None:
            raise Exception("No state returned from graph")
        
        # Format the results
        result = {
            "status": final_state.get("status", "completed"),
            "agents_results": {
                "summary": {
                    "agent": "Context-Aware Summary Agent",
                    "summary": final_state.get("summary", ""),
                    "chunk_count": len(final_state.get("summary_chunks", []) or [])
                },
                "actions": {
                    "agent": "Action & Dependency Extraction Agent",
                    "extracted_data": final_state.get("action_items", {})
                },
                "risks": {
                    "agent": "Risk & Open-Issues Agent",
                    "analysis": final_state.get("risks_analysis", {})
                }
            },
            "consolidated_report": final_state.get("consolidated_report", {}),
            "errors": final_state.get("errors", []),
            "from_cache": False,
            "content_hash": content_hash
        }
        
        # Save to cache for future use
        if use_cache:
            save_agent_result_to_cache(content_hash, result, document)
        
        return result
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error: {str(e)}", 1.0)
        
        return {
            "status": "error",
            "error": str(e),
            "agents_results": {},
            "consolidated_report": {},
            "from_cache": False,
            "content_hash": content_hash
        }


