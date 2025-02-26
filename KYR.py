import streamlit as st

def show(run_chatbot):
    st.title("📖 Know Your Rights")
    st.write("Welcome to the Know Your Rights section. Ask your queries below.")
    run_chatbot()  # Call the chatbot function
