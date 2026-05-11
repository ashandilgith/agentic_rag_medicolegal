from langgraph.graph import StateGraph, END
from state import AgentState
# NEW: Import the extract_node alongside the others
from nodes import retrieve_node, extract_node, draft_node, critique_node, refine_node

def should_continue(state: AgentState):
    """Determines if the draft is approved or needs more refinement."""
    # If the Senior Auditor node outputs 'PASS', or we hit our loop limit, end the graph.
    if "PASS" in state["critique"] or state.get("revision_count", 0) >= 3:
        return "end"
    return "refine"

def build_graph(retriever):
    """Compiles the agentic workflow."""
    workflow = StateGraph(AgentState)

    # 1. Define the Nodes (The "Departments" of your AI agent)
    workflow.add_node("retrieve", lambda state: retrieve_node(state, retriever))
    workflow.add_node("extract", extract_node)  # NEW: The Forensic Accountant
    workflow.add_node("draft", draft_node)      # The Technical Writer
    workflow.add_node("critique", critique_node) # The Senior Auditor
    workflow.add_node("refine", refine_node)     # The Remediator

    # 2. Define the Linear Edges (The standard assembly line)
    workflow.set_entry_point("retrieve")
    
    # NEW: The retrieved documents now go to the extractor first
    workflow.add_edge("retrieve", "extract")
    
    # NEW: The extracted bullet points and context then go to the drafter
    workflow.add_edge("extract", "draft")
    
    # The drafted text goes to the auditor for review
    workflow.add_edge("draft", "critique")

    # 3. Define the Conditional Routing (The feedback loop)
    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {
            "refine": "refine",
            "end": END
        }
    )

    # If routed to refine, send the fixed draft back to the auditor for a second look
    workflow.add_edge("refine", "critique") 
    
    return workflow.compile()