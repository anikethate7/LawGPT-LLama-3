import streamlit as st

def show(run_chatbot):
    st.title("🖥️ Cyber Law")
    st.write("Welcome to the Cyber Law section. Ask your queries below.")
    run_chatbot()  # Call the chatbot function
