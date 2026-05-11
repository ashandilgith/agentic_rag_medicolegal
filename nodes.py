from langchain_core.prompts import ChatPromptTemplate
from config import llm
from state import AgentState

def retrieve_node(state: AgentState, retriever):
    """Retrieves relevant case files and medical records."""
    print(f"Retrieving context for: {state['query']}")
    docs = retriever.invoke(state["query"]) 
    return {"context": docs}


def extract_node(state: AgentState):
    """Isolates hard facts and constraints before any drafting occurs."""
    print("Extracting hard facts and constraints...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a forensic data extractor. Your objective is to review the Source Context and extract all immutable facts relevant to the User Query.
        
        Rules:
        1. Extract exact dates, times, financial figures, physical measurements, and names.
        2. Identify Demographic/Contextual Limits: Note who or what the source document applies to (e.g., "This study applies to children," "This law applies to California").
        3. Do not write a summary. Output ONLY a strict, categorized bulleted list of facts.
        4. If a piece of information is requested but missing, explicitly write "DATA MISSING: [Topic]".
        """),
        ("human", "Source Context: {context}\n\nUser Query: {query}")
    ])
    chain = prompt | llm
    
    formatted_context = "\n\n".join(doc.page_content for doc in state["context"])
    response = chain.invoke({"context": formatted_context, "query": state["query"]})
    return {"extracted_facts": response.content}



def draft_node(state: AgentState):
    """Generates the initial draft using the extracted facts as a scaffold."""
    print("Drafting initial response...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert technical drafter. Create a comprehensive, highly professional response to the User Query.
        
        Rules:
        1. Foundation: You MUST incorporate every relevant fact from the 'Extracted Facts' list.
        2. Strict Grounding: Do not invent, infer, or hallucinate information. 
        3. No Filler: Do not use boilerplate language or generic platitudes to pad the response. If information is missing, explicitly state "The provided documentation does not specify..."
        4. Tone: Objective, clinical, and analytical.
        """),
        ("human", "Extracted Facts:\n{facts}\n\nSource Context:\n{context}\n\nUser Query: {query}")
    ])
    chain = prompt | llm
    
    formatted_context = "\n\n".join(doc.page_content for doc in state["context"])
    response = chain.invoke({
        "facts": state["extracted_facts"],
        "context": formatted_context, 
        "query": state["query"]
    })
    return {"draft": response.content}




def critique_node(state: AgentState):
    """The ruthless firewall that checks for logic, relevance, and grounding."""
    print("Critiquing draft...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a ruthless, highly-scrutinizing Senior Auditor. Your job is to evaluate the Draft against the Extracted Facts and Source Context.
        
        Perform the following checks:
        1. RELEVANCE GATE: Does the Source Context logically apply to the User Query? (e.g., Do not apply pediatric medical data to an adult; do not apply irrelevant case law). If it does not apply, flag this as a FATAL ERROR.
        2. FACTUAL INTEGRITY: Are all numbers, dates, and claims in the Draft perfectly matched to the Extracted Facts?
        3. HALLUCINATION CHECK: Did the Draft invent any information or use vague filler language?
        
        Output Format:
        If the Draft passes all checks flawlessly, output exactly and only: "PASS".
        If the Draft fails ANY check, provide a bulleted list of strict directives on how to fix it. Do not be polite. Be highly specific.
        """),
        ("human", "User Query: {query}\n\nExtracted Facts:\n{facts}\n\nDraft:\n{draft}")
    ])
    chain = prompt | llm
    
    response = chain.invoke({
        "query": state["query"],
        "facts": state["extracted_facts"], 
        "draft": state["draft"]
    })
    return {"critique": response.content}


def refine_node(state: AgentState):
    """Executes the Auditor's critique to fix the draft."""
    print("Refining draft based on critique...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert remediator. Your job is to fix the Draft by strictly following the Auditor's Critique.
        
        Rules:
        1. Address every single point raised in the Critique.
        2. If the Critique identifies a 'FATAL ERROR' regarding relevance, rewrite the section to explicitly state that the provided source material is inapplicable to the query.
        3. Maintain the objective, analytical tone.
        """),
        ("human", "Auditor Critique:\n{critique}\n\nExtracted Facts:\n{facts}\n\nOriginal Draft:\n{draft}")
    ])
    chain = prompt | llm
    
    response = chain.invoke({
        "critique": state["critique"], 
        "facts": state["extracted_facts"],
        "draft": state["draft"]
    })
    
    current_count = state.get("revision_count", 0)
    return {"draft": response.content, "revision_count": current_count + 1}


