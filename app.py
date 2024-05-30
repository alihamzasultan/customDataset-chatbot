import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile

# Replace with your OpenAI API key
HARD_CODED_API_KEY = "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"

# Hardcoded path to the CSV file
HARD_CODED_CSV_PATH = "dataset/AI.csv"

# Load the CSV file
loader = CSVLoader(file_path=HARD_CODED_CSV_PATH, encoding="utf-8")
data = loader.load()

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=HARD_CODED_API_KEY)
vectors = FAISS.from_documents(data, embeddings)

# Create the Conversational Retrieval Chain
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=HARD_CODED_API_KEY),
    retriever=vectors.as_retriever()
)

# Function to handle conversational chat
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask me anything about the QR-Code 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

# Container for the chat history
response_container = st.container()
# Container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk about your queries here ", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
