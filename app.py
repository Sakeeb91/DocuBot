# Import required libraries
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, embedding_model):
    if embedding_model == "OpenAI":
        embeddings = OpenAIEmbeddings()
    elif embedding_model == "HuggingFace":
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    else:
        raise ValueError("Invalid language model choice. Supported options: 'OpenAI' and 'HuggingFace'.")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, conversational_model, model_temperature=0.5):
    if conversational_model == "OpenAI":
        llm = ChatOpenAI(temperature=model_temperature)
    elif conversational_model == "HuggingFace":
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    # code for PDF upload and model selection

    pdf_docs = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
    if pdf_docs:
        text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(text)
        embedding_model = st.selectbox("Choose an embedding model", ["OpenAI", "HuggingFace"])
        vectorstore = get_vectorstore(text_chunks, embedding_model)
        conversational_model = st.selectbox("Choose a conversational model", ["OpenAI", "HuggingFace"])
        model_temperature = st.slider("Choose a model temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        conversation_chain = get_conversation_chain(vectorstore, conversational_model, model_temperature)
        st.session_state.conversation = conversation_chain
        st.session_state.chat_history = None
        st.write("You can now ask questions about your documents.")

    # code for chat history display
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

    # code for user input
    if st.button("Ask"):
        user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
