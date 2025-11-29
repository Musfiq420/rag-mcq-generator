# 📘 RAG-Based MCQ Generator (NCTB Books)

An interactive **MCQ generator** built with **Streamlit**, **Sentence Transformers**, **Pinecone Vector Database**, and **Groq LLM**.  
This application retrieves relevant content from the **NCTB Physics curriculum** using semantic search and generates **Bengali MCQs with explanations**.

---

## 🚀 Features

- 🔍 **RAG Pipeline** using Sentence Transformers + Pinecone v3  
- 🧠 **Groq Llama 3.1** for high-quality MCQ generation  
- 🌐 **Streamlit Web Interface**  
- 📝 **Interactive Bengali MCQs** (clickable, shows correct answers)  
- ⚡ Fast & lightweight inference  

---

## 🧰 Tech Stack

| Feature | Technology |
|--------|------------|
| Web UI | Streamlit |
| Embeddings | SentenceTransformer |
| Vector DB | Pinecone v3 |
| LLM | Groq (Llama 3.1) |
| Language | Python |

---

## 📦 Installation

**Clone the repository:**

```bash
git clone https://github.com/Musfiq420/rag-mcq-generator.git
cd rag-mcq-generator
```

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔐 Environment Variables

Create a file:  
`/.streamlit/secrets.toml`

Add:

```toml
PINECONE_API_KEY = "your_pinecone_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

## ▶️ Run the App

```bash
streamlit run mcq_streamlit_app.py
```

Then visit:
http://localhost:8501




## 🧠 How It Works

1. User inputs **chapter**, **topic**, and **number of MCQs**  
2. Query is embedded using **SentenceTransformer**  
3. **Pinecone** performs semantic vector search  
4. Retrieved context is injected into an **MCQ generation prompt**  
5. **Groq LLM** generates Bengali MCQs + explanations  
6. **Streamlit** renders the MCQs using HTML + JS
