import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Phase 2 imports
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3 imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

@st.cache_data  # not to store vectors every time and store in cache
def get_vectorstore(file):
    folder = 'tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = f'./{folder}/{file.name}'
    with open(file_path, 'wb') as f:
            f.write(file.getvalue())
    loaders=[PyPDFLoader(file_path)] 
    # Create chunks/vectors using chromedb
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-miniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore

# Set up the page
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("PDF RAG Chatbot")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    pdf_upload = st.file_uploader("Upload your PDF Here", type=['pdf'])
    if pdf_upload is not None:   
        st.success(f"Uploaded: {pdf_upload.name}")
        with st.expander("View PDF"):
            pdf_data = pdf_upload.read()
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=pdf_upload.name,
                mime="application/pdf"
            )
            vectorstore = get_vectorstore(pdf_upload)
    st.button("Clear Chat History", on_click=lambda: st.session_state.messages.clear())
    model = st.selectbox("Select Model", ["llama3-8b-8192","gemma2-9b-it","other-model"], key="model_choice")

# Set a session state variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the history with chat bubbles
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.chat_message("User").markdown(message['content'])
    else:
        st.chat_message("Assistant").markdown(message['content'])

# Chat input with placeholder text
prompt = st.chat_input("Type your question here...")

if prompt:
    st.chat_message("User").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Define system prompt
    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are very smart at everything. Always give the best and  most accurate. Answer the following question: {user_prompt}.
        Start the answer directly. No small talk, please."""
    )

    # Initialize the model
    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )

    try:
        if pdf_upload is not None:
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True
            )
            result = chain({"query": prompt})
            response = result["result"]
            # # Display sources
            # st.write("Sources:")
            # for doc in result["source_documents"]:
            #     st.write(f"Source: {doc.metadata['source']}")
        else:
            st.error("Failed to Load PDF, Assistant might search in web..")
            response = ""
            chain = groq_sys_prompt | groq_chat | StrOutputParser()
            response = chain.invoke({'user_prompt': prompt})

        st.chat_message("Assistant").markdown(response)
        st.session_state.messages.append({'role': 'Assistant', 'content': response})
    except Exception as e:
        st.error(f"Error: [{str(e)}]")

# Add download button for chat history
if st.button("Download Chat History"):
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    st.download_button("Download", history, file_name="chat_history.txt")

# Add styles for better UI
st.markdown(
    """
    <style>
        .stChatMessage { margin-bottom: 15px; }
        .stSpinner > div { border-color: #4CAF50 transparent transparent; }
    </style>
    """,
    unsafe_allow_html=True
)
