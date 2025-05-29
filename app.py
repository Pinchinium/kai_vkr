import streamlit as st

# Настройка навигации
pages = {
    "Анализ и модель": "analysis_and_model.py",
    "Презентация": "presentation.py"
}

st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти к:", list(pages.keys()))

if page == "Анализ и модель":
    from analysis_and_model import analysis_and_model_page
    analysis_and_model_page()
elif page == "Презентация":
    from presentation import presentation_page
    presentation_page()