import streamlit as st

def show(run_chatbot):
    st.title("🏡 Property Law")
    st.write("Welcome to the Property Law section. Ask your queries below.")
    run_chatbot()  # Call the chatbot function
