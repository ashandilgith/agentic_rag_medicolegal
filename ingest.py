from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from config import embeddings, QDRANT_URL, QDRANT_API_KEY

# ==========================================
# WRITE MODE: For adding new files
# ==========================================
def build_vector_store(directory_path: str):
    """
    Reads PDFs, chunks them, and loads them into a persistent Qdrant Cloud cluster.
    Run this manually via terminal ONLY when new files are added to the directory.
    """
    print(f"Loading documents from {directory_path}...")
    loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    # Medical records require larger chunks to maintain context across sections.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    print("Syncing vectors to Qdrant Cloud...")
    
    # .from_documents() processes the text chunks and writes them to the database.
    vector_store = QdrantVectorStore.from_documents(
        splits,
        embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name="baker_legal_universe",
    )
    
    # We update k=15 here as well, just in case this function is used for a quick terminal test.
    return vector_store.as_retriever(search_kwargs={"k": 15})


# ==========================================
# READ MODE: For the Streamlit Dashboard
# ==========================================
def get_existing_retriever():
    """
    Connects to the persistent Qdrant Cloud cluster without uploading new documents.
    This is what the Streamlit app calls to instantly access the 2.5TB universe.
    """
    print("Connecting to existing Qdrant cluster...")
    
    # .from_existing_collection() simply shakes hands with the database. 
    # It does not cost compute or API tokens to run this connection.
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="baker_legal_universe",
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    
    # k=15 ensures the retriever pulls enough chunks to capture all dense facts, 
    # preventing the "Lossy Compression" omission of specific damages and dates.
    return vector_store.as_retriever(search_kwargs={"k": 15})