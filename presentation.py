import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    # Содержание презентации в формате Markdown
    presentation_markdown = """
# Прогнозирование отказов оборудования
---
## Введение
- Задача: предсказать отказ оборудования.
- Датасет с параметрами производственного процесса.
- Цель: определить, произойдёт ли отказ (Target = 1) или нет (Target = 0).
---
## Этапы работы
1. Загрузка и анализ данных.
2. Очистка и предобработка.
3. Выбор моделей и обучение.
4. Оценка качества: точность, полнота, F1-мера.
5. Визуализация и выводы.
---
## Методы и результаты
- Random Forest
- Logistic Regression  
- XGBoost
**Результаты:** Accuracy > 95%
**Лучшая точность:** XGBoost
---
## Streamlit-приложение
- Основная страница: ввод параметров и предсказание.
- Страница анализа и визуализации.
- Страница презентации проекта.
---
## Заключение
- Проект успешно решает задачу прогнозирования отказов.
- Возможные улучшения:
    - Гиперпараметрическая оптимизация.
    - Дополнительные алгоритмы.
    - Использование временных рядов.
    """

    # Настройки презентации в сайдбаре
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

    # Отображение презентации
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )
