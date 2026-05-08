from langchain_community.embeddings import HuggingFaceEmbeddings

# Modelo multilingüe ligero, sin coste de API y apto para Streamlit Cloud (1 GB RAM).
# Alternativa con Gemini Embeddings: GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
