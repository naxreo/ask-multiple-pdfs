import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import requests


def get_url_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None


def get_pdf_text(pdf_docs):
    text_chunks = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text_chunks.append(page.extract_text())
    return '\n'.join(text_chunks)

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if 'conversation' not in st.session_state:
        st.session_state.conversation = {}
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    chat_messages = []
    for i, message in enumerate(st.session_state.chat_history[-2:]):
        if i % 2 == 0:
            pass
            # chat_messages.append(user_template.replace("{{MSG}}", message.content))
        else:
            chat_messages.append(bot_template.replace("{{MSG}}", message.content))

    st.markdown(''.join(chat_messages), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )
        
        url = st.text_input("Enter URL:")
        
        if st.button("Process"):
            with st.spinner("Processing"):
                if url:
                    # Fetch text content from the specified URL
                    text = get_url_text(url)
                    if text:
                        text_chunks = get_text_chunks(text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    else:
                        st.error("Failed to fetch text from the URL.")
                elif pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
