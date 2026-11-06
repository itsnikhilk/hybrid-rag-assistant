import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

    folder_path = project_dir / "knowledge_base"
    
    if not folder_path.is_dir():
        print(f"Directory {folder_path} does not exist. Please create it and add text files.")
        return

    raw_text_string = ""
    for file in folder_path.glob("*.txt"):
        content = file.read_text(encoding="utf-8")
        raw_text_string += content + "\n"

    if not raw_text_string.strip():
        print("No text found in the knowledge base directory.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=70
    )
    chunks = splitter.split_text(raw_text_string)

    llm = ChatOpenRouter(
        model="nvidia/nemotron-3.5-super-vl:free",
        temperature=0.5,
        api_key=api_key
    )

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding
    )

    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    # Fixed: passed chunks instead of raw_text_string
    keyword_retriever = BM25Retriever.from_texts(chunks)
    
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )

    print("\n--- Knowledge Base Ready ---")
    print("Example topics: AI, Cybersecurity, Cloud Computing, Blockchain")
    print("Sample query: 'What is cloud computing?'\n")

    query = input("Enter your query: ")
    retrieved_docs = hybrid_retriever.invoke(query)

    prompt = f"""Context: {retrieved_docs}

Question: {query}

Answer only using the context. If the answer isn't supported, say: "I don't have enough information in my knowledge base." Be concise and don't invent facts."""

    output = llm.invoke(prompt)
    print("Output:", output.content)

if __name__ == "__main__":
    main()