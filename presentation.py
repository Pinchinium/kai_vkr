import streamlit as st

def main():
    st.title("Презентация проекта")
    st.markdown("""
    ## Прогнозирование отказов оборудования
    - **Цель:** Предсказать отказ оборудования.
    - **Методы:** Random Forest, Logistic Regression.
    - **Результаты:** Accuracy > 95%.
    """)

if __name__ == "__main__":
    main()