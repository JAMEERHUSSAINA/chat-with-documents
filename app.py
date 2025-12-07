import streamlit as st
import os
from dotenv import load_dotenv
from io import BytesIO
from PyPDF2 import PdfReader
# from docx import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from groq import Groq

# ----------------------------- Load ENV -----------------------------
load_dotenv()
# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ----------------------------- File Processing -----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_bytes = pdf.read()
        reader = PdfReader(BytesIO(pdf_bytes))
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        pdf.seek(0)
    return text



# ----------------------------- Text Splitter -----------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n"]
    )
    return splitter.split_text(text)

# ----------------------------- Vectorstore -----------------------------
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(text_chunks, embedding=embeddings)


# ----------------------------- Conversation Chain -----------------------------
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


# ----------------------------- Streamlit App -----------------------------
def main():
    st.set_page_config(page_title="Chat with Documents", page_icon="📚")
    st.header("📚 Chat with Documents")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    user_question = st.text_input("Ask a question about your documents...")
    if user_question and st.session_state.conversation:
        response = st.session_state.conversation({"question": user_question})
        st.write(response["answer"])

    with st.sidebar:
        st.subheader("Upload Your Documents")

        pdf_docs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        # docx_docs = st.file_uploader("Upload Word files", type="docx", accept_multiple_files=True)
        # txt_docs = st.file_uploader("Upload Text files", type="txt", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one file!")
            else:
                with st.spinner("Extracting text..."):
                    text = get_pdf_text(pdf_docs)

                with st.spinner("Splitting text..."):
                    chunks = get_text_chunks(text)

                with st.spinner("Creating vector database..."):
                    vectorstore = get_vectorstore(chunks)

                with st.spinner("Preparing model..."):
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Document processed successfully! 🚀")

# def get_docx_text(docx_file):
#     doc = Document(docx_file)
#     text = ""
#     for para in doc.paragraphs:
#         text += para.text + "\n"
#     return text


# def get_txt_text(txt_file):
#     raw = txt_file.read()
#     txt_file.seek(0)
#     return raw.decode("utf-8")


# def extract_all_text(pdf_docs, docx_docs, txt_docs):
#     text = ""
#     if pdf_docs:
#         text += get_pdf_text(pdf_docs)
#     if docx_docs:
#         for d in docx_docs:
#             text += get_docx_text(d)
#     if txt_docs:
#         for t in txt_docs:
#             text += get_txt_text(t)
#     return text
if __name__ == "__main__":
    main()
