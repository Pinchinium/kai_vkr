import streamlit as st
import reveal_slides as rs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    st.title("Анализ данных и модель")
    
    presentation_markdown = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    - Описание задачи и датасета.
    - Цель: предсказать отказ оборудования (Target = 1) или его отсутствие
    (Target = 0).
    ---
    ## Этапы работы
    1. Загрузка данных.
    2. Предобработка данных.
    3. Обучение модели.
    4. Оценка модели.
    5. Визуализация результатов.
    ---
    ## Streamlit-приложение
    - Основная страница: анализ данных и предсказания.
    - Страница с презентацией: описание проекта.
    ---
    ## Заключение
    - Итоги и возможные улучшения.
    """

    uploaded_file = st.file_uploader("data/ai4i2020.csv", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Первые 5 строк данных:", data.head())
        
        # Предобработка данных
        data = data.drop(columns=['UDI', 'Product ID'])
        data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})
        
        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Обучение модели
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Оценка
        y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
        st.write("Classification Report:", classification_report(y_test, y_pred))

    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige",
        "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave",
        "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex",
        "mathjax2", "mathjax3", "notes", "search", "zoom"], [])
    # Отображение презентации
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},)
if __name__ == "__main__":
    main()