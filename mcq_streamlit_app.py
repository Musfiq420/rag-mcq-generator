import streamlit as st
from streamlit.components.v1 import html
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# ---------------- STREAMLIT SECRETS -------------------------
# ============================================================

pinecone_api_key = st.secrets["PINECONE_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# ============================================================
# ---------------- INITIALIZE EMBEDDING MODEL ----------------
# ============================================================

embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ============================================================
# ---------------- INITIALIZE PINECONE (v3) ------------------
# ============================================================

pc = Pinecone(api_key=pinecone_api_key)

# Connect to existing index
index_name = "nctb-physics"
index = pc.Index(index_name)

# ============================================================
# ---------------- SEMANTIC SEARCH FUNCTION ------------------
# ============================================================

def semantic_search(query, top_k=5):
    qvec = embed_model.encode(query).tolist()
    
    res = index.query(
        vector=qvec,
        top_k=top_k,
        include_metadata=True
    )

    # Pinecone v3 returns dictionary response
    matches = res.get("matches", [])
    return [m["metadata"].get("text", "") for m in matches]

# ============================================================
# ---------------- MCQ GENERATION PROMPT ---------------------
# ============================================================

MCQ_PROMPT = """
You must output ONLY valid HTML code.

Generate EXACTLY {n} Bengali MCQs using the reference context below.
Output must contain the following structure exactly (interactive MCQs):

<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>MCQ</title>
<style>
.mcq-list {{ list-style: none; padding: 0; }}
.mcq-item {{ margin-bottom: 20px; }}
.question {{ font-weight: bold; margin-bottom: 10px; }}
.options {{ list-style: none; padding: 0; }}
.options li {{ margin-bottom: 8px; padding: 6px; cursor: pointer; }}
.options li:hover {{ background: #ddd; }}
.feedback {{ display: none; margin-top: 10px; padding: 10px; background: #f0f0f0; }}
</style>
</head>
<body>

<ol class="mcq-list">
  <li class="mcq-item">
    <div class="question">QUESTION_TEXT</div>
    <ul class="options">
      <li onclick="reveal(this, 'a')">a) OPTION_A</li>
      <li onclick="reveal(this, 'b')">b) OPTION_B</li>
      <li onclick="reveal(this, 'c')">c) OPTION_C</li>
      <li onclick="reveal(this, 'd')">d) OPTION_D</li>
    </ul>
    <div class="feedback" data-answer="b">
      <strong>সঠিক উত্তর:</strong> b <br>
      <span class="explanation">EXPLANATION_TEXT</span>
    </div>
  </li>
</ol>

<script>
function reveal(element, option) {{
    let item = element.closest('.mcq-item');
    let feedback = item.querySelector('.feedback');
    let correct = feedback.getAttribute('data-answer');
    let opts = item.querySelectorAll('.options li');

    opts.forEach(li => li.style.background = '#eee');

    if(option === correct) {{
        element.style.background = '#c8f7c5';
    }} else {{
        element.style.background = '#f7c5c5';
    }}
    feedback.style.display = 'block';
}}
</script>

</body>
</html>

### NOW GENERATE:
Replace QUESTION_TEXT, OPTIONS, and EXPLANATION_TEXT for exactly {n} MCQs.
Use this reference context:
{context}
"""

prompt = PromptTemplate(template=MCQ_PROMPT, input_variables=["n", "context"])
parser = StrOutputParser()

# ============================================================
# ---------------- MCQ GENERATION FUNCTION -------------------
# ============================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=groq_api_key
)

def generate_mcqs_html(chapter, topic, n_mcq, top_k=5):
    query = f"{chapter} {topic}"
    context_list = semantic_search(query, top_k=top_k)
    context = "\n\n".join(context_list)

    chain = prompt | llm | parser
    return chain.invoke({"n": n_mcq, "context": context})

# ============================================================
# ----------------------- STREAMLIT UI ------------------------
# ============================================================

st.title("📘 RAG-Based MCQ Generator (NCTB Physics)")

chapter = st.text_input("Chapter Name")
topic = st.text_input("Topic Name")
n = st.number_input("Number of MCQs", min_value=1, max_value=20, value=5)

if st.button("Generate MCQs"):
    if not chapter or not topic:
        st.error("Please enter both chapter and topic.")
    else:
        with st.spinner("Generating MCQs..."):
            html_code = generate_mcqs_html(chapter, topic, n)
        html(html_code, height=900, scrolling=True)
