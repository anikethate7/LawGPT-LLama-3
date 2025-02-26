import streamlit as st

def show(run_chatbot):
    st.title("⚖️ Criminal Law")
    st.write("Welcome to the Criminal Law section. Ask your queries below.")
    run_chatbot()  # Call the chatbot function
