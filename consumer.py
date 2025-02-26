import streamlit as st

def show(run_chatbot):
    st.title("📜 Consumer Law")
    st.write("Welcome to the Consumer Law section. Ask your queries below.")
    run_chatbot()  # Call the chatbot function
