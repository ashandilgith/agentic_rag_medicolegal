from typing import TypedDict, List
from langchain_core.documents import Document

class AgentState(TypedDict):
    query: str
    context: List[Document]
    extracted_facts: str  # NEW: Holds the hard data before drafting begins
    draft: str
    critique: str
    revision_count: int
    final_output: str