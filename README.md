# Hybrid RAG Knowledge Assistant

An open-source Retrieval-Augmented Generation (RAG) tool built with **LangChain**, featuring hybrid search combining dense vector embeddings with sparse keyword-based search (`BM25`). Powered by HuggingFace embeddings and OpenRouter LLMs.

---

## Features

- **Hybrid Retrieval:** Combines semantic search (`ChromaDB` + `all-MiniLM-L6-v2`) with lexical keyword matching (`BM25Retriever`) via `EnsembleRetriever`.
- **Custom Knowledge Base:** Ingests local `.txt` documents dynamically.
- **Strict Grounding:** Custom prompt instructions prevent hallucinations by strictly limiting responses to the retrieved context.
- **Flexible LLMs:** Integrated with OpenRouter to easily swap underlying models.

---

## Project Structure

```text
.
├── knowledge_base/        # Place your source .txt files here
│   └── sample.txt
├── .env.example           # Environment variable template
├── .gitignore             # Standard git ignore file
├── main.py                # Main script execution
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
```

---

## Quickstart

### 1. Prerequisites
- Python 3.9+
- An API Key from [OpenRouter](https://openrouter.ai/)

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and add your OpenRouter key:

```bash
cp .env.example .env
```

Inside `.env`:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Running the Assistant

1. Add your `.txt` files into the `knowledge_base/` directory.
2. Run the application:

```bash
python main.py
```

3. Enter your query when prompted.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.