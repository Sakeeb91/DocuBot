# DocuBot

This project focuses on developing a conversational chatbot capable of answering queries based on multiple uploaded PDF documents. Utilizing conversational retrieval techniques, the chatbot harnesses the power of Artificial Intelligence to provide accurate and contextual answers. The Streamlit application is designed to be customizable according to user needs, with a conversation memory feature that enables a seamless and continuous chat interaction between the user and the tool.

## Pre-requisites

Before diving into the code, ensure that your system has the following requirements installed:

- Python 3.7 or higher
- Streamlit
- PyPDF2
- HuggingFace API
- OpenAI API
- FAISS

## Installation

You can install the necessary libraries using pip:

```bash
pip install streamlit PyPDF2 faiss-gpu
```

## Getting Started

**1. Cloning the Repository:**

```bash
git clone https://github.com/your-repository-link
cd your-repository-folder
```

**2. Running the Application:**

```bash
streamlit run app.py
```

After executing the above command, a local server will start, and you can access the application via your web browser at `http://localhost:8501`.

## Usage

Upon launching the application, you can upload the PDF documents you wish the chatbot to reference. Enter your queries in the chat interface, and the chatbot will retrieve relevant information from the provided documents to answer your questions.

## Features

- Conversational Retrieval: Provides accurate answers by searching through the uploaded PDF documents based on your queries.
- Conversation Memory: Retains the context of the conversation to ensure a coherent and continuous interaction.
- Customizable Interface: The Streamlit application interface is designed to be user-friendly and customizable to meet your needs.

## Contribution

Feel free to fork the project, submit issues, and contribute to its continuous improvement!

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

---

Happy coding, and enjoy building your conversational chatbot with Langchain and large language models!
