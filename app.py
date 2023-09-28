from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
import streamlit as st
import os
import fitz
from PIL import Image

st.title("PDF Chatbot")

def process_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_response(chain, history, query):
    result = chain(
        {"question": query, 'chat_history': history}, return_only_outputs=True)
    return result["answer"]

def main():
    st.write("Provide your OpenAI API Key:")
    api_key = st.text_input("OpenAI API Key:", type="password")  # Creates a password input box for the API key

    if api_key:  # Checks if api_key is not empty
        os.environ['OPENAI_API_KEY'] = api_key  # Sets the API key from user input
        st.write("Upload a PDF file:")
        pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
        query = st.text_input("Enter a question:", "")

        if pdf_file is not None:
            text = process_pdf(pdf_file)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(text)
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
            chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3),
                                                         retriever=vectorstore.as_retriever(),
                                                         memory=memory)
            history = []
            for chunk in chunks:
                response = generate_response(chain, history, chunk)
                history.append(chunk)
                history.append(response)
                st.write(response)

if __name__ == '__main__':
    main()
