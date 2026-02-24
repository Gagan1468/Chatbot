# app.py (run karne ke liye: streamlit run app.py)
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ────────────────────────────────────────────────
# Apna website ka data yahan daal do (text mein)
# Zyada pages hain to file se load kar sakte ho
# ────────────────────────────────────────────────
documents = [
    "KIROS is an AI-powered nutrition scanner and personalized fitness coach by UDAAN TECH SOLUTIONS.",
    "Founder Gagan is a Mumbai-based AI developer and fitness enthusiast.",
    "KIROS features real-time food scanning, nutrition analysis, adaptive workout plans.",
    "Contact: gamerhustle12@gmail.com or rao14685448@gmail.com",
    "Follow on Instagram: https://www.instagram.com/tech.gagan26/",
    "X (Twitter): https://x.com/hustlegame92691",
    "LinkedIn: www.linkedin.com/in/gagan-yadav-1138b5367",
    # Aur jitna data hai sab yahan add kar dena...
]

# Embeddings + Vector Store (local & free)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Mistral-7B-Instruct-v0.3 with YOUR token
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token="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",  # ← Tera token
    temperature=0.7,
    max_new_tokens=512,
    repetition_penalty=1.1
)

# Prompt template (KIROS personality)
prompt = PromptTemplate.from_template(
    """You are KIROS, an AI assistant from UDAAN TECH SOLUTIONS. 
    Be helpful, friendly, and focused on fitness, nutrition, and KIROS features.
    Use the context below to answer accurately.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
)

# RAG Chain
rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ────────────────────────────────────────────────
# Streamlit Chat UI
# ────────────────────────────────────────────────

st.title("KIROS Chatbot – Powered by Mistral-7B")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about KIROS, fitness, or nutrition..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("KIROS thinking..."):
            response = rag_chain.invoke(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
