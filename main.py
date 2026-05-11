import os
from ingest import build_vector_store
from graph import build_graph

if __name__ == "__main__":
    # Create a mock directory if it doesn't exist for Codespace testing
    test_dir = "sample_docs"
    os.makedirs(test_dir, exist_ok=True)
    
    print("--- Initializing System ---")
    # In a real environment, you'd ensure PDFs are in 'sample_docs'
    retriever = build_vector_store(test_dir) 
    agent_app = build_graph(retriever)
    
    test_query = "Identify every mention of hull integrity warnings sent at least 30 days prior to the collision in the maritime logs."
    
    initial_state = {
        "query": test_query,
        "revision_count": 0
    }
    
    print(f"\n--- Processing Query: '{test_query}' ---")
    for event in agent_app.stream(initial_state):
        for k, v in event.items():
            if k != "__end__":
                print(f"--> Node Completed: {k}")
                
    final_state = agent_app.invoke(initial_state)
    
    print("\n--- FINAL OUTPUT ---")
    print(final_state["draft"])