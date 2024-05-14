import streamlit as st
from PyPDF2 import PdfReader
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ctransformers import AutoModelForCausalLM

# Initialize encoder and language model
encoder = SentenceTransformer('jinaai/jina-embedding-b-en-v1')
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q3_K_S.gguf",
    model_type="llama",
    temperature=0.2,
    repetition_penalty=1.5,
    max_new_tokens=300,
)

# Initialize Qdrant client
client = QdrantClient(path="./db")

def setup_database(files):
    all_chunks = []
    for file in files:
        pdf_path = file
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50, length_function=len)
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    
    # Setup collection
    client.recreate_collection(
        collection_name="my_facts",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    
    # Upload documents
    for idx, chunk in enumerate(all_chunks):
        client.upload_record(
            collection_name="my_facts",
            record=models.Record(
                id=idx,
                vector=encoder.encode(chunk).tolist(),
                payload={"text": chunk}
            )
        )

def get_answer(question):
    hits = client.search(
        collection_name="my_facts",
        query_vector=encoder.encode(question).tolist(),
        limit=3
    )
    
    context = " ".join(hit.payload["text"] for hit in hits)
    prompt = f"Context: {context}\nQuestion: {question}"
    response = llm(prompt)
    return response

# Streamlit interface
st.title("PDF Document Q&A System")

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])
if uploaded_files:
    setup_database(uploaded_files)
    st.session_state['database_initialized'] = True

if st.session_state.get('database_initialized', False):
    # Input for user question
    user_question = st.text_input("Ask a question based on the uploaded PDFs:")

    if user_question:
        # Generate answer
        answer = get_answer(user_question)
        # Display conversation
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
        st.session_state.conversation.append(f"User: {user_question}")
        st.session_state.conversation.append(f"Bot: {answer}")
        for line in st.session_state.conversation:
            st.text(line)
else:
    st.write("Please upload some PDF files to initialize the database.")
