import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from ctransformers import AutoModelForCausalLM

# Initialize the encoder and the chatbot
encoder = SentenceTransformer('jinaai/jina-embedding-b-en-v1')
client = QdrantClient(path="./db2")
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q3_K_S.gguf", model_type="llama")

def process_and_index_pdf(file):
    reader = PdfReader(file)
    all_text = "".join(page.extract_text() or "" for page in reader.pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50, length_function=len)
    chunks = text_splitter.split_text(all_text)

    # Ensure the collection is prepared for new data
    client.recreate_collection(
        collection_name="documents",
        vector_size=encoder.get_sentence_embedding_dimension(),
        distance='Cosine',
    )

    # Indexing documents
    records = [
        {
            "id": idx,
            "vector": encoder.encode(chunk).tolist(),
            "payload": {"text": chunk}
        } for idx, chunk in enumerate(chunks)
    ]
    client.upload_records(collection_name="documents", records=records)

def admin_interface():
    st.subheader("Upload and Index New PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=['pdf'])
    if st.button("Process and Index PDFs"):
        for uploaded_file in uploaded_files:
            process_and_index_pdf(uploaded_file)
        st.success("PDFs processed and indexed successfully.")

def answer_question(question):
    query_vector = encoder.encode(question).tolist()
    hits = client.search(collection_name="documents", query_vector=query_vector, top_k=3)
    context = " ".join(hit['payload']['text'] for hit in hits)
    
    system_prompt = f"You are a helpful co-worker, you will use the provided context to answer user questions. Read the given context before answering questions. Context: {context} User question: {question}"
    
    answer = llm.generate(system_prompt, max_length=500)
    return answer

def chat_interface():
    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your question:")
    if user_question and st.button("Submit"):
        response = answer_question(user_question)
        st.write(response)

def main():
    st.title("School PDF Knowledge Base and Chatbot")
    if st.sidebar.selectbox("Mode", ["Admin", "User"]) == "Admin":
        admin_interface()
    else:
        chat_interface()

if __name__ == "__main__":
    main()
