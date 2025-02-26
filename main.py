import streamlit as st
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Import category modules
import consumer
import criminal
import cyber
import property
import KYR

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="LawGPT")

# Dictionary of categories
categories = {
    "📖 Know Your Rights": KYR,
    "⚖️ Criminal Law": criminal,
    "🖥️ Cyber Law": cyber,
    "🏡 Property Law": property,
    "📜 Consumer Law": consumer,
}

# Initialize session state
if "selected_category" not in st.session_state:
    st.session_state.selected_category = list(categories.keys())[0]

if "messages_per_category" not in st.session_state:
    st.session_state.messages_per_category = {category: [] for category in categories.keys()}

# Enhanced Sidebar UI Styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        width: 280px !important;
        background: linear-gradient(135deg, #18182b, #25253c) !important;
        color: white;
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .sidebar-title {
        font-size: 26px;
        font-weight: bold;
        text-align: center;
        color: white;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
    }

    .sidebar-content {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .sidebar-footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-size: 14px;
        margin-top: 20px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar UI
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚖️ LawGPT</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        for idx, category in enumerate(categories.keys()):  # Ensure unique keys
            if st.button(category, use_container_width=True, key=f"button_{idx}"):
                st.session_state.selected_category = category
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-footer">🛡️ Stay legally informed.</div>', unsafe_allow_html=True)


# Function to run chatbot
def run_chatbot():
    current_category = st.session_state.selected_category

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

    # Initialize embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Define prompt template
    prompt_template = """
    <s>[INST]This is a chat template. As a legal chatbot, your primary objective is to provide accurate and concise information based on the user's questions. 
    CONTEXT: {context}
    CHAT HISTORY: {chat_history}
    QUESTION: {question}
    ANSWER:
    </s>[INST]
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

    # Initialize LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    # Set up QA chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    # Display previous messages for the selected category
    for message in st.session_state.messages_per_category[current_category]:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))

    # Chat input
    input_prompt = st.chat_input("Ask a legal question...")
    if input_prompt:
        with st.chat_message("user"):
            st.write(input_prompt)
        st.session_state.messages_per_category[current_category].append({"role": "user", "content": input_prompt})

        with st.chat_message("assistant"):
            with st.status("Thinking 💡...", expanded=True):
                result = qa.invoke(input=input_prompt)
                message_placeholder = st.empty()
                full_response = ""

                for chunk in result["answer"]:
                    full_response += chunk
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " ▌")

        st.session_state.messages_per_category[current_category].append({"role": "assistant", "content": result["answer"]})

# Call the show function of the selected category
categories[st.session_state.selected_category].show(run_chatbot)
