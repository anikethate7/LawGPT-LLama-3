import streamlit as st
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

st.set_page_config(page_title="NyayGURU")

# Dictionary of categories
categories = {
    "📚 Know Your Rights": KYR,
    "⚖️ Criminal Law": criminal,
    "🖥️ Cyber Law": cyber,
    "🏡 Property Law": property,
    "📛 Consumer Law": consumer,
}

# Supported languages
languages = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
}

# Initialize session state
if "selected_category" not in st.session_state:
    st.session_state.selected_category = list(categories.keys())[0]

if "messages_per_category" not in st.session_state:
    st.session_state.messages_per_category = {category: [] for category in categories.keys()}

if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"

if "loading" not in st.session_state:
    st.session_state.loading = False

# Ensure sidebar remains the same in both light and dark mode
st.markdown("""
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        width: 280px !important;
        background: #18182b !important;  /* Dark background */
        color: white !important;
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* Sidebar Title */
    .sidebar-title {
        font-size: 26px;
        font-weight: bold;
        text-align: center;
        color: white !important;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
    }

    /* Sidebar buttons */
    .stButton>button {
        width: 100% !important;
        background-color: #25253c !important; /* Dark gray */
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 10px !important;
        padding: 10px !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }

    /* Hover effect */
    .stButton>button:hover {
        background-color: #303048 !important; /* Slightly lighter gray */
        color: white !important;
    }

    /* Sidebar Footer */
    .sidebar-footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 14px;
        margin-top: 20px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Language selector */
    .language-selector {
        margin-top: 20px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Ensure consistency in light mode */
    html[theme="light"] [data-testid="stSidebar"] {
        background: #18182b !important;
        color: white !important;
    }
    
    /* Custom loader */
    .custom-loader {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }
    
    .loader-dot {
        width: 12px;
        height: 12px;
        margin: 0 5px;
        border-radius: 50%;
        background-color: #18182b;
        display: inline-block;
        animation: bounce 1.5s infinite ease-in-out;
    }
    
    .loader-dot:nth-child(1) {
        animation-delay: 0s;
    }
    
    .loader-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .loader-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes bounce {
        0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Adjust loader in dark mode */
    html[theme="dark"] .loader-dot {
        background-color: #f0f2f6;
    }
    
    /* Legal balanced scale loader */
    .scale-loader {
        width: 100px;
        height: 100px;
        position: relative;
        margin: 20px auto;
    }
    
    .scale-bar {
        width: 100px;
        height: 8px;
        background-color: #18182b;
        position: absolute;
        top: 50%;
        left: 0;
        transform-origin: center;
        animation: balance 3s infinite ease-in-out;
        border-radius: 4px;
    }
    
    .scale-left, .scale-right {
        width: 40px;
        height: 40px;
        background-color: rgba(24, 24, 43, 0.7);
        border-radius: 4px;
        position: absolute;
        bottom: -25px;
    }
    
    .scale-left {
        left: 0;
    }
    
    .scale-right {
        right: 0;
    }
    
    .scale-stand {
        width: 8px;
        height: 70px;
        background-color: #18182b;
        position: absolute;
        bottom: -55px;
        left: 50%;
        transform: translateX(-50%);
        border-radius: 4px;
    }
    
    .scale-base {
        width: 60px;
        height: 8px;
        background-color: #18182b;
        position: absolute;
        bottom: -60px;
        left: 50%;
        transform: translateX(-50%);
        border-radius: 4px;
    }
    
    @keyframes balance {
        0%, 100% {
            transform: rotate(-10deg);
        }
        50% {
            transform: rotate(10deg);
        }
    }
    
    /* Adjust scale loader in dark mode */
    html[theme="dark"] .scale-bar,
    html[theme="dark"] .scale-stand,
    html[theme="dark"] .scale-base {
        background-color: #f0f2f6;
    }
    
    html[theme="dark"] .scale-left,
    html[theme="dark"] .scale-right {
        background-color: rgba(240, 242, 246, 0.7);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar UI
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚖️ NyayGURU</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        for idx, category in enumerate(categories.keys()):
            if st.button(category, use_container_width=True, key=f"button_{idx}"):
                st.session_state.selected_category = category
        st.markdown('</div>', unsafe_allow_html=True)

    # Language selector
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    st.write("🌐 Select Language:")
    selected_language = st.selectbox(
        label="Language",
        options=list(languages.keys()),
        index=list(languages.keys()).index(st.session_state.selected_language),
        label_visibility="collapsed"
    )
    if selected_language != st.session_state.selected_language:
        st.session_state.selected_language = selected_language
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-footer">🛡️ Stay legally informed.</div>', unsafe_allow_html=True)

# Custom loader components
def show_scale_loader():
    st.markdown("""
        <div class="scale-loader">
            <div class="scale-bar"></div>
            <div class="scale-left"></div>
            <div class="scale-right"></div>
            <div class="scale-stand"></div>
            <div class="scale-base"></div>
        </div>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource(show_spinner=False)
def load_vector_store(_embeddings):
    # The embeddings are not hashable, so we prevent Streamlit from hashing them.
    return FAISS.load_local("my_vector_store", _embeddings, allow_dangerous_deserialization=True)

@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

@st.cache_data(show_spinner=False)
def translate_text(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    translation_prompt = f"""
    You are a professional legal translator with expertise in accurately translating legal documents while preserving their precise meaning and intent.

Task:
Translate the following text from {source_lang} to {target_lang}, ensuring that the legal terminology, nuances, and context remain intact. The translation must be legally accurate and convey the original intent without any misinterpretation or ambiguity.

Guidelines:

Legal Precision: Maintain the original legal meaning, terminology, and structure. Avoid altering or omitting any critical legal details.
Clarity & Readability: Ensure the translation is clear, concise, and easy to understand while preserving formal legal language.
Structured Formatting: Use short paragraphs, bullet points, and proper spacing to enhance readability.
No Additional Content: Provide only the translated text without any introductory phrases, explanations, or extra notes.
Output Format:

Strictly translated legal text without any personal interpretation.
Well-organized and properly formatted text to ensure readability and professional presentation.
    {text}
    """
    response = llm.invoke(translation_prompt)
    return response.content


# Function to run chatbot
def run_chatbot():
    current_category = st.session_state.selected_category
    current_language = st.session_state.selected_language
    language_code = languages[current_language]  # Get the language code

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=2, memory_key="chat_history", return_messages=True
        )

    # Initialize embeddings and vector store
    embeddings = load_embeddings()
    db = load_vector_store(embeddings)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Initialize LLM
    llm = load_llm()

    # Set up QA chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever
    )

    # Display previous messages for the selected category
    for message in st.session_state.messages_per_category[current_category]:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))

    # Chat input
    input_placeholder = f"Ask a legal question in {current_language}..."
    input_prompt = st.chat_input(input_placeholder)

    if input_prompt:
        # Display user message
        with st.chat_message("user"):
            st.write(input_prompt)

        # Store original user message
        st.session_state.messages_per_category[current_category].append({"role": "user", "content": input_prompt})

        # Process the query
        with st.chat_message("assistant"):
            # Set loading state
            st.session_state.loading = True
            
            # Show custom loader
            loader_container = st.empty()
            with loader_container.container():
                show_scale_loader()
                st.markdown("<p style='text-align:center;'></p>", unsafe_allow_html=True)
            
            # Use the ConversationalRetrievalChain with the original query
            result = qa.invoke({"question": input_prompt})

            # Get the response
            english_response = result["answer"]

            # Translate the response to the target language
            if current_language != "English":
                final_response = translate_text(english_response, "English", current_language)
            else:
                final_response = english_response

            # Clear the loader
            loader_container.empty()
            
            # Display with typing effect
            message_placeholder = st.empty()
            full_response = ""

            # Process character by character for the typing effect
            for char in final_response:
                full_response += char
                time.sleep(0.01)  # Slightly faster typing effect
                message_placeholder.markdown(full_response + " ▌")

            message_placeholder.markdown(full_response)
            
            # Reset loading state
            st.session_state.loading = False

        # Store response
        st.session_state.messages_per_category[current_category].append({"role": "assistant", "content": final_response})

# Call the show function of the selected category
categories[st.session_state.selected_category].show(run_chatbot)
