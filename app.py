import streamlit as st

pages = {
    "Анализ и модель": "analysis_and_model.py",
    "Презентация": "presentation.py"
}

st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу", list(pages.keys()))

with open(pages[page], "r", encoding="utf-8") as file:
    exec(file.read())
