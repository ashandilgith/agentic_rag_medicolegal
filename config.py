import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ==========================================
# ENVIRONMENT VARIABLES
# ==========================================
# Load variables from the .env file into the environment
load_dotenv()

# Securely retrieve the keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Fail fast if the environment isn't set up correctly
if not OPENAI_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Missing API keys. Please check your .env file.")

# ==========================================
# MODEL CONFIGURATION
# ==========================================
# ON-PREM: Swap `ChatOpenAI` for your local reasoning engine.
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# ON-PREM: Swap for `HuggingFaceEmbeddings` using `BGE-M3`.
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)