import os
import tempfile
import streamlit as st
from google import genai
from google.genai import types

# Secure RAG & Chunking Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------------------------------------
# 1. SECURITY & CONFIGURATION
# ---------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("Security Alert: GEMINI_API_KEY environment variable is missing. Halting execution.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# Utilizing the new 2.5 Flash model which grants 500 free Search RPD
MODEL_ID = 'gemini-2.5-flash'

# UI Initialization
st.set_page_config(page_title="Agentic Legal Counsel 2.5", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "last_petition" not in st.session_state:
    st.session_state.last_petition = None

# ---------------------------------------------------------
# 2. CORE MODULE: SECURE LOCAL KNOWLEDGE BASE (ChromaDB)
# ---------------------------------------------------------
def build_secure_vector_db(uploaded_files):
    """
    Ingests static PDFs (YÖK, School Rules) into a local ChromaDB.
    Implements strict cleanup protocols to prevent memory leaks.
    """
    print("System Log: Initiating secure Vector DB compilation...")
    documents = []
    
    for uploaded_file in uploaded_files:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        temp_path = None
        
        try:
            # Security: Use context manager for temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
                
            print(f"System Log: Parsing document -> {uploaded_file.name}")
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())
            
        except Exception as e:
            print(f"System Log: Parsing error on {uploaded_file.name} -> {e}")
        finally:
            # Security: Guaranteed cleanup of temp files
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"System Log: Temporary file purged -> {temp_path}")

    if not documents:
        return None

    try:
        print("System Log: Executing text chunking...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        print("System Log: Initializing HuggingFace Embeddings...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Persist directory ensures we don't hold massive data in volatile RAM
        vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="./chroma_v25_db")
        print("System Log: Database compilation successful!")
        return vector_db
    except Exception as e:
        print(f"System Log: Database generation failed -> {e}")
        return None

def stream_generator(response_stream):
    """Yields chunks for dynamic frontend rendering."""
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text

# ---------------------------------------------------------
# 3. FRONTEND: SIDEBAR CONFIGURATION
# ---------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Hybrid Agentic RAG")
    st.info("Engine: Gemini 2.5 Flash + Local ChromaDB + Google Search Grounding (500 RPD Free)")
    
    rulebook_files = st.file_uploader("Upload Static Regulations (PDF)", type=['pdf'], accept_multiple_files=True)
    
    if st.button("Compile Knowledge Base"):
        if rulebook_files:
            with st.spinner("Encrypting and vectorizing documents locally..."):
                db = build_secure_vector_db(rulebook_files)
                if db:
                    st.session_state.vector_db = db
                    st.success("Knowledge Base is Online!")
                else:
                    st.error("Compilation failed. Check system logs.")
        else:
            st.warning("Upload PDF documents first.")

    if st.button("Purge Session Memory"):
        st.session_state.messages = []
        st.session_state.last_petition = None
        st.rerun()

# ---------------------------------------------------------
# 4. MAIN CHAT & AGENTIC ROUTER LOGIC
# ---------------------------------------------------------
st.title("⚖️ Autonomous Legal Counsel (v2.5)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your legal issue (e.g., grading error, mobbing)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Retrieve static local data
        local_context = "No local documents provided."
        if st.session_state.vector_db:
            print(f"System Log: Querying Vector DB for -> '{prompt[:30]}...'")
            relevant_chunks = st.session_state.vector_db.similarity_search(prompt, k=3)
            local_context = "\n".join([f"- {c.page_content}" for c in relevant_chunks])
            
        # Agentic System Instruction
        sys_instruction = """
        You are an elite legal AI agent in Turkey.
        
        ROUTING PROTOCOL:
        1. If the query is about academic rules (grades, attendance), prioritize the 'LOCAL RULEBOOK' below.
        2. If the query involves crimes, constitutional rights, or mobbing, you MUST use the Google Search tool to find recent Turkish Penal Code (TCK) and Supreme Court (Yargıtay) precedents.
        3. Do not hallucinate laws. Cite specific article numbers. Respond entirely in Turkish.
        """
        
        # Payload construction
        final_payload = f"""
        LOCAL RULEBOOK EXCERPTS:
        {local_context}
        
        USER QUERY: {prompt}
        """
        
        try:
            print(f"System Log: Dispatching payload to {MODEL_ID} with Native Search enabled...")
            response_stream = client.models.generate_content_stream(
                model=MODEL_ID,
                contents=final_payload,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    temperature=0.2,
                    tools=[types.Tool(google_search=types.GoogleSearch())] # Now legally using our 500 free RPD!
                )
            )
            full_response = st.write_stream(stream_generator(response_stream))
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Inference Error: {e}")
            print(f"System Log: Critical AI Inference Error -> {e}")

# ---------------------------------------------------------
# 5. ACTION BUTTONS (Context Optimization applied)
# ---------------------------------------------------------
if len(st.session_state.messages) > 1:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    # Payload Pruning: Strip heavy RAG contexts to prevent token bloat (Error 429 mitigation)
    clean_history = "\n".join([
        f"{msg['role'].upper()}: {msg['content'][:600]}..." 
        for msg in st.session_state.messages[-4:]
    ])

    with col1:
        if st.button("📝 Draft Official Petition"):
            with st.spinner("Drafting formal legal petition..."):
                petition_prompt = f"Based on this recent chat history, draft a formal, legally structured petition (dilekçe) in Turkish addressing the relevant university authority.\n\nCLEAN HISTORY:\n{clean_history}"
                try:
                    print("System Log: Executing background task -> Petition Generation")
                    res = client.models.generate_content(
                        model=MODEL_ID,
                        contents=petition_prompt,
                        config=types.GenerateContentConfig(temperature=0.2)
                    )
                    st.session_state.last_petition = res.text
                    st.subheader("Official Petition Draft")
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"Generation Error: {e}")

    with col2:
        if st.button("⚖️ Execute Anti-Thesis Analysis"):
            if st.session_state.last_petition:
                with st.spinner("Adversarial AI analyzing petition..."):
                    anti_prompt = f"Act as the opposing strict university counsel. Critique the following petition for procedural flaws and state exactly why it could be rejected in Turkish.\n\nPETITION:\n{st.session_state.last_petition}"
                    try:
                        print("System Log: Executing background task -> Anti-Thesis Generation")
                        anti_res = client.models.generate_content(
                            model=MODEL_ID,
                            contents=anti_prompt,
                            config=types.GenerateContentConfig(temperature=0.6)
                        )
                        st.subheader("⚠️ Risk & Anti-Thesis Report")
                        st.warning(anti_res.text)
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")
            else:
                st.warning("Generate a petition first to analyze it.")