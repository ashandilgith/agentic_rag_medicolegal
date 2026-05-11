import streamlit as st
import os
# CHANGED: Import the connection function instead of the build function
from ingest import get_existing_retriever
from graph import build_graph

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Undisclosed Seattle Personal Injury Law Firm - Agentic RAG", layout="wide")
st.title("Legal/Medical Intelligence Engine")
st.markdown("Test environment for the Agentic 'Critic-Refiner' drafting loop.")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
# We store the graph in session state so it doesn't rebuild on every UI click
if "agent_app" not in st.session_state:
    st.session_state.agent_app = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# SIDEBAR: SYSTEM INITIALIZATION
# ==========================================
with st.sidebar:
    st.header("System Controls")
    
    # CHANGED: Button text and logic reflect connecting, not ingesting
    if st.button("Connect to Qdrant Database"):
        with st.spinner("Connecting to vector store and compiling graph..."):
            retriever = get_existing_retriever()
            st.session_state.agent_app = build_graph(retriever)
            st.success("System Connected & Ready!")
    
    st.markdown("---")
    # CHANGED: Updated instructions
    st.markdown("**How to test:**\n1. Click 'Connect to Qdrant Database'.\n2. Ask a query.")

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Enter your query (e.g., 'Draft a summary of the patient's symptoms...')"):
    
    if st.session_state.agent_app is None:
        st.error("Please connect the system from the sidebar first.")
        st.stop()

    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Process through LangGraph Agent
    with st.chat_message("assistant"):
        status_container = st.empty()
        
        initial_state = {
            "query": prompt,
            "revision_count": 0
        }
        
        # Stream the graph execution to show the "thought process"
        status_text = "Starting Agentic Loop...\n"
        status_container.code(status_text)
        
        for event in st.session_state.agent_app.stream(initial_state):
            for node_name, state_update in event.items():
                if node_name != "__end__":
                    status_text += f"✓ Completed Node: {node_name.upper()}\n"
                    status_container.code(status_text)
        
        # Fetch the final state to display the drafted response
        final_state = st.session_state.agent_app.invoke(initial_state)
        final_draft = final_state["draft"]
        
        # Clear the status block and show the final markdown
        status_container.empty()
        st.markdown(final_draft)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": final_draft})